"""
src/inference/retriever.py
pgvector-based ANN retrieval + in-memory fallback.
User tower produces a query embedding; pgvector returns top-K candidates.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PgVectorRetriever:
    """
    Retrieves candidate items from PostgreSQL + pgvector.
    Falls back to in-memory numpy search if DB unavailable.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        table_name: str = "item_embeddings",
        top_k: int = 500,
    ):
        self.db_url     = db_url or os.getenv("DATABASE_URL")
        self.table_name = table_name
        self.top_k      = top_k
        self._conn      = None

        # In-memory fallback
        self._emb_matrix: Optional[np.ndarray] = None
        self._item_ids:   Optional[np.ndarray]  = None

        self._try_connect()

    def _try_connect(self):
        if not self.db_url:
            logger.info("No DATABASE_URL — using in-memory retrieval")
            return
        try:
            import psycopg2
            self._conn = psycopg2.connect(self.db_url)
            logger.info("Connected to pgvector")
        except Exception as e:
            logger.warning(f"pgvector connection failed: {e} — using in-memory fallback")
            self._conn = None

    def load_embeddings_to_memory(self, emb_matrix: np.ndarray, item_ids: np.ndarray):
        """Load embeddings for in-memory cosine search (fallback)."""
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        self._emb_matrix = emb_matrix / np.maximum(norms, 1e-8)
        self._item_ids   = item_ids
        logger.info(f"Loaded {len(item_ids)} item embeddings into memory")

    def index_items(self, emb_matrix: np.ndarray, item_ids: List[int]):
        """Write item embeddings into pgvector table."""
        self.load_embeddings_to_memory(emb_matrix, np.array(item_ids))
        if self._conn is None:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        item_id INTEGER PRIMARY KEY,
                        embedding vector({emb_matrix.shape[1]})
                    )
                """)
                cur.execute(f"TRUNCATE {self.table_name}")
                from psycopg2.extras import execute_values
                rows = [(int(iid), emb_matrix[i].tolist())
                        for i, iid in enumerate(item_ids)]
                execute_values(
                    cur,
                    f"INSERT INTO {self.table_name} (item_id, embedding) VALUES %s",
                    rows,
                    template="(%s, %s::vector)",
                )
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_hnsw_idx
                    ON {self.table_name}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m=16, ef_construction=64)
                """)
            self._conn.commit()
            logger.info(f"Indexed {len(item_ids)} items in pgvector")
        except Exception as e:
            logger.warning(f"pgvector indexing failed: {e}")
            self._conn.rollback()

    def retrieve(
        self,
        query_emb: np.ndarray,
        k: Optional[int] = None,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Return list of (item_id, score) sorted by descending score."""
        k = k or self.top_k

        if self._conn is not None:
            try:
                return self._pg_retrieve(query_emb, k, exclude_ids)
            except Exception as e:
                logger.warning(f"pgvector query failed: {e} — falling back to memory")

        return self._memory_retrieve(query_emb, k, exclude_ids)

    def _pg_retrieve(self, query_emb, k, exclude_ids):
        query_emb = query_emb / np.maximum(np.linalg.norm(query_emb), 1e-8)
        q = query_emb.tolist()

        if exclude_ids:
            sql = f"""
                SELECT item_id, 1 - (embedding <=> %s::vector) AS score
                FROM {self.table_name}
                WHERE item_id <> ALL(%s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [q, list(exclude_ids), q, k]
        else:
            sql = f"""
                SELECT item_id, 1 - (embedding <=> %s::vector) AS score
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [q, q, k]

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def _memory_retrieve(self, query_emb, k, exclude_ids):
        if self._emb_matrix is None:
            return []
        q = query_emb / np.maximum(np.linalg.norm(query_emb), 1e-8)
        scores = self._emb_matrix @ q
        if exclude_ids is not None and len(exclude_ids) > 0:
            excl = set(exclude_ids)
            for i, iid in enumerate(self._item_ids):
                if int(iid) in excl:
                    scores[i] = -1.0
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(self._item_ids[i]), float(scores[i])) for i in top_idx]
