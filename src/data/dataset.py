"""
src/data/dataset.py
PyTorch Dataset classes for all training stages:
  - BERT4RecDataset    : masked item prediction (self-supervised pre-training)
  - TwoTowerDataset    : contrastive retrieval training
  - RankingDataset     : BPR pairwise ranking
  - ContinualDataset   : streaming incremental fine-tuning
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_ID  = 0
MASK_ID = 1
CLS_ID  = 2


class BERT4RecDataset(Dataset):
    """
    Self-supervised masked item prediction.
    For each user sequence, randomly replace 15% of items with MASK_ID.
    The model must predict the original item at masked positions.
    """

    def __init__(
        self,
        sequences: Dict[int, List[int]],
        n_items: int,
        max_seq_len: int = 200,
        mask_prob: float = 0.15,
        augment_seqs: Optional[List[List[int]]] = None,
    ):
        self.sequences   = list(sequences.values())
        self.n_items     = n_items
        self.max_seq_len = max_seq_len
        self.mask_prob   = mask_prob
        # Optionally include LLM-augmented synthetic sequences
        if augment_seqs:
            self.sequences.extend(augment_seqs)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx][-self.max_seq_len:]  # truncate
        seq_len = len(seq)

        # Build masked input and labels
        input_ids = []
        labels    = []
        for item in seq:
            if random.random() < self.mask_prob:
                input_ids.append(MASK_ID)
                labels.append(item)      # predict original
            else:
                input_ids.append(item)
                labels.append(PAD_ID)    # ignore in loss

        # Pad to max_seq_len
        pad_len = self.max_seq_len - seq_len
        input_ids = [PAD_ID] * pad_len + input_ids
        labels    = [PAD_ID] * pad_len + labels
        attn_mask = [False] * pad_len + [True] * seq_len  # True = real token

        return {
            "input_ids":   torch.tensor(input_ids,  dtype=torch.long),
            "labels":      torch.tensor(labels,      dtype=torch.long),
            "attn_mask":   torch.tensor(attn_mask,   dtype=torch.bool),
        }


class TwoTowerDataset(Dataset):
    """
    Contrastive retrieval training with in-batch negatives.
    Returns (user_sequence, positive_item_id).
    Negatives are handled inside the training loop (all other positives in batch).
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[int]],
        val_targets: Dict[int, int],
        max_seq_len: int = 200,
        split: str = "train",
    ):
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[List[int], int]] = []

        if split == "train":
            for uid, seq in train_seqs.items():
                # Create a sample for every item in the training history
                # (use all but the last item as context, last as positive)
                if len(seq) >= 2:
                    self.samples.append((seq[:-1], seq[-1]))
        else:  # val
            for uid, target in val_targets.items():
                if uid in train_seqs:
                    self.samples.append((train_seqs[uid], target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, pos_item = self.samples[idx]
        seq = seq[-self.max_seq_len:]
        seq_len = len(seq)
        pad_len = self.max_seq_len - seq_len
        input_ids = [PAD_ID] * pad_len + list(seq)
        attn_mask = [False] * pad_len + [True] * seq_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),
            "pos_item":  torch.tensor(pos_item,  dtype=torch.long),
        }


class RankingDataset(Dataset):
    """
    BPR pairwise ranking dataset.
    Returns (user_sequence, positive_item, negative_item, ips_weight).
    Negative sampling: 50% uniform + 50% popularity-weighted.
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[int]],
        n_items: int,
        ips_weights: Optional[np.ndarray] = None,
        popularity: Optional[np.ndarray] = None,
        max_seq_len: int = 200,
        n_negatives: int = 1,
    ):
        self.max_seq_len  = max_seq_len
        self.n_items      = n_items
        self.n_negatives  = n_negatives
        self.ips_weights  = ips_weights
        self.samples: List[Tuple[List[int], int]] = []

        for uid, seq in train_seqs.items():
            if len(seq) >= 2:
                self.samples.append((seq[:-1], seq[-1]))

        # Popularity-weighted negative sampling distribution
        if popularity is not None:
            pop = popularity[3:]  # skip PAD/MASK/CLS
            pop = pop ** 0.75    # smooth
            pop = pop / pop.sum()
            self.pop_probs = pop
        else:
            self.pop_probs = None

        # Build per-user item sets for fast negative checking
        self.user_item_sets = {
            i: set(seq) for i, (seq, _) in enumerate(self.samples)
        }

    def _sample_negative(self, sample_idx: int) -> int:
        user_items = self.user_item_sets[sample_idx]
        for _ in range(100):
            if self.pop_probs is not None and random.random() < 0.5:
                neg = int(np.random.choice(len(self.pop_probs), p=self.pop_probs)) + 3
            else:
                neg = random.randint(3, self.n_items + 2)
            if neg not in user_items:
                return neg
        return random.randint(3, self.n_items + 2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, pos_item = self.samples[idx]
        seq = seq[-self.max_seq_len:]
        seq_len = len(seq)
        pad_len = self.max_seq_len - seq_len
        input_ids = [PAD_ID] * pad_len + list(seq)
        attn_mask = [False] * pad_len + [True] * seq_len

        neg_item = self._sample_negative(idx)

        ips_w = 1.0
        if self.ips_weights is not None and pos_item < len(self.ips_weights):
            ips_w = float(self.ips_weights[pos_item])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),
            "pos_item":  torch.tensor(pos_item,  dtype=torch.long),
            "neg_item":  torch.tensor(neg_item,  dtype=torch.long),
            "ips_weight": torch.tensor(ips_w,    dtype=torch.float),
        }


class ContinualDataset(Dataset):
    """
    Streaming dataset for continual / incremental fine-tuning.
    Combines new interactions (80%) with experience replay from historical data (20%).
    """

    def __init__(
        self,
        new_sequences: Dict[int, List[int]],
        historical_sequences: Optional[Dict[int, List[int]]] = None,
        replay_ratio: float = 0.2,
        max_seq_len: int = 200,
    ):
        self.max_seq_len = max_seq_len
        new_samples = [(seq[:-1], seq[-1]) for seq in new_sequences.values() if len(seq) >= 2]

        # Experience replay: randomly sample from historical data
        replay_samples = []
        if historical_sequences and replay_ratio > 0:
            n_replay = int(len(new_samples) * replay_ratio / (1 - replay_ratio))
            hist_list = [(seq[:-1], seq[-1]) for seq in historical_sequences.values() if len(seq) >= 2]
            if hist_list:
                replay_samples = random.choices(hist_list, k=min(n_replay, len(hist_list)))

        self.samples = new_samples + replay_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, pos_item = self.samples[idx]
        seq = seq[-self.max_seq_len:]
        seq_len = len(seq)
        pad_len = self.max_seq_len - seq_len
        input_ids = [PAD_ID] * pad_len + list(seq)
        attn_mask = [False] * pad_len + [True] * seq_len

        # Simple uniform negative for continual training
        neg_item = random.randint(3, 50000)  # approximate

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),
            "pos_item":  torch.tensor(pos_item,  dtype=torch.long),
            "neg_item":  torch.tensor(neg_item,  dtype=torch.long),
            "ips_weight": torch.tensor(1.0,       dtype=torch.float),
        }
