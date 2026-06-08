import React, { useState } from 'react';
import { api } from '../api';

const MODES = [
  { value: 'neural', label: 'Neural' },
  { value: 'llm',    label: 'LLM Re-rank' },
  { value: 'hybrid', label: 'Hybrid' },
];

export default function RecommendationFeed({ users }) {
  const [userId, setUserId]       = useState('');
  const [mode, setMode]           = useState('neural');
  const [history, setHistory]     = useState([]);
  const [recs, setRecs]           = useState([]);
  const [reasoning, setReasoning] = useState('');
  const [pathway, setPathway]     = useState('');
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState('');
  const [liked, setLiked]         = useState(new Set());

  async function fetchRecs(uid, m) {
    if (!uid) return;
    setLoading(true); setError('');
    try {
      const [hist, rec] = await Promise.all([
        api.userHistory(uid),
        api.recommend({ user_id: parseInt(uid), k: 10, mode: m || mode }),
      ]);
      setHistory(hist.history || []);
      setRecs(rec.items || []);
      setReasoning(rec.reasoning || '');
      setPathway(rec.pathway || '');
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  async function handleLike(itemId) {
    setLiked(s => new Set([...s, itemId]));
    await api.feedback({ user_id: parseInt(userId), item_id: itemId, interaction_type: 'like' }).catch(() => {});
  }

  const maxScore = recs.length ? Math.max(...recs.map(r => r.score)) : 1;

  return (
    <div>
      <style>{`
        .rec-row:hover { background: rgba(200,169,110,0.04) !important; }
        .hist-row:hover { background: rgba(200,169,110,0.03) !important; }
        .like-btn:hover { color: #c8a96e !important; }
        .mode-btn:hover { border-color: #c8a96e !important; color: #c8a96e !important; }
        .user-select option { background: #0d0d0d; }
        @keyframes barGrow { from{width:0} to{width:var(--w)} }
      `}</style>

      {/* Header */}
      <div style={S.header}>
        <div>
          <h2 style={S.pageTitle}>Recommendations</h2>
          <p style={S.pageSubtitle}>Neural retrieval with cross-attention ranking</p>
        </div>
      </div>

      {/* Controls */}
      <div style={S.controls}>
        <div style={S.selectWrap}>
          <select
            className="user-select"
            style={S.select}
            value={userId}
            onChange={e => { setUserId(e.target.value); fetchRecs(e.target.value); }}
          >
            <option value="">Select a user...</option>
            {(users || []).map(u => (
              <option key={u.user_id} value={u.user_id}>
                User {u.user_id} — {u.n_interactions} interactions
              </option>
            ))}
          </select>
        </div>

        <div style={S.modeGroup}>
          {MODES.map(m => (
            <button
              key={m.value}
              className="mode-btn"
              style={{ ...S.modeBtn, ...(mode === m.value ? S.modeBtnActive : {}) }}
              onClick={() => { setMode(m.value); if (userId) fetchRecs(userId, m.value); }}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {error && <div style={S.error}>{error}</div>}

      {/* Panels */}
      <div style={S.panels}>
        {/* History Panel */}
        <div style={S.panel}>
          <div style={S.panelHeader}>
            <span style={S.panelLabel}>Interaction History</span>
            {history.length > 0 && <span style={S.panelCount}>{history.length} items</span>}
          </div>
          <div style={S.panelBody}>
            {history.length === 0 && (
              <div style={S.empty}>Select a user to view history</div>
            )}
            {history.map((item, i) => (
              <div key={i} className="hist-row" style={S.histRow}>
                <span style={S.histIdx}>{String(history.length - i).padStart(2, '0')}</span>
                <span style={S.histTitle}>{item.title}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations Panel */}
        <div style={S.panel}>
          <div style={S.panelHeader}>
            <span style={S.panelLabel}>Recommendations</span>
            {pathway && <span style={S.pathwayTag}>{pathway}</span>}
          </div>
          <div style={S.panelBody}>
            {loading && <div style={S.empty}>Generating recommendations...</div>}
            {!loading && recs.length === 0 && <div style={S.empty}>No recommendations yet</div>}
            {recs.map((item, i) => (
              <div key={item.item_id} className="rec-row" style={S.recRow}>
                <div style={S.recMeta}>
                  <span style={S.recRank}>{String(i + 1).padStart(2, '0')}</span>
                  <span style={S.recTitle}>{item.title}</span>
                  <button
                    className="like-btn"
                    style={{ ...S.likeBtn, ...(liked.has(item.item_id) ? S.likeBtnActive : {}) }}
                    onClick={() => handleLike(item.item_id)}
                  >
                    {liked.has(item.item_id) ? '♥' : '♡'}
                  </button>
                </div>
                <div style={S.barTrack}>
                  <div style={{
                    ...S.barFill,
                    '--w': `${(item.score / maxScore) * 100}%`,
                    width: `${(item.score / maxScore) * 100}%`,
                    animation: `barGrow 0.6s ease ${i * 0.05}s both`,
                  }} />
                </div>
                <span style={S.scoreVal}>{item.score.toFixed(3)}</span>
              </div>
            ))}
            {reasoning && recs.length > 0 && (
              <div style={S.reasoningBox}>
                <div style={S.reasoningLabel}>Reasoning</div>
                <div style={S.reasoningText}>{reasoning}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const S = {
  header: { marginBottom: 32 },
  pageTitle: { fontSize: 28, fontWeight: 800, letterSpacing: '-0.02em', color: '#e8e0d4' },
  pageSubtitle: { fontSize: 12, color: '#3a3530', marginTop: 4, letterSpacing: '0.04em', fontFamily: "'JetBrains Mono', monospace" },

  controls: { display:'flex', gap:12, alignItems:'center', marginBottom:28, flexWrap:'wrap' },
  selectWrap: { position:'relative' },
  select: {
    padding: '10px 16px',
    background: '#0d0d0d',
    border: '1px solid #1a1a1a',
    color: '#e8e0d4',
    fontSize: 13,
    fontFamily: "'Syne', sans-serif",
    minWidth: 240,
    cursor: 'pointer',
    outline: 'none',
    appearance: 'none',
    borderRadius: 0,
  },
  modeGroup: { display:'flex', gap:0 },
  modeBtn: {
    padding: '10px 18px',
    background: 'transparent',
    border: '1px solid #1a1a1a',
    borderLeft: 'none',
    color: '#3a3530',
    cursor: 'pointer',
    fontSize: 12,
    fontFamily: "'Syne', sans-serif",
    fontWeight: 600,
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
    transition: 'all 0.15s',
  },
  modeBtnActive: { background:'#c8a96e', color:'#080808', borderColor:'#c8a96e' },

  panels: { display:'grid', gridTemplateColumns:'1fr 1fr', gap:1, background:'#1a1a1a' },
  panel: { background:'#0d0d0d' },
  panelHeader: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '14px 20px',
    borderBottom: '1px solid #1a1a1a',
  },
  panelLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:10, color:'#3a3530', letterSpacing:'0.12em', textTransform:'uppercase' },
  panelCount: { fontFamily:"'JetBrains Mono',monospace", fontSize:10, color:'#c8a96e' },
  pathwayTag: {
    fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#c8a96e',
    border:'1px solid rgba(200,169,110,0.3)', padding:'2px 8px', letterSpacing:'0.08em',
  },
  panelBody: { padding:'8px 0', maxHeight:520, overflowY:'auto' },

  empty: { padding:'40px 20px', color:'#2a2520', fontSize:12, fontFamily:"'JetBrains Mono',monospace" },

  histRow: { display:'flex', alignItems:'center', gap:12, padding:'8px 20px', transition:'background 0.1s' },
  histIdx: { fontFamily:"'JetBrains Mono',monospace", fontSize:10, color:'#2a2520', minWidth:20, flexShrink:0 },
  histTitle: { fontSize:13, color:'#6a6560', flex:1, lineHeight:1.4 },

  recRow: { padding:'10px 20px', transition:'background 0.1s' },
  recMeta: { display:'flex', alignItems:'center', gap:10, marginBottom:6 },
  recRank: { fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:'#c8a96e', minWidth:22, flexShrink:0, fontWeight:500 },
  recTitle: { fontSize:13, color:'#c8c0b8', flex:1, lineHeight:1.4 },
  likeBtn: { background:'transparent', border:'none', color:'#2a2520', cursor:'pointer', fontSize:14, flexShrink:0, transition:'color 0.15s', padding:'0 2px' },
  likeBtnActive: { color:'#c8a96e' },
  barTrack: { height:2, background:'#111', marginBottom:4 },
  barFill: { height:'100%', background:'linear-gradient(90deg, #c8a96e, #e8d4a8)', transformOrigin:'left' },
  scoreVal: { fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#2a2520' },

  reasoningBox: { margin:'16px 20px 8px', padding:'14px 16px', background:'#080808', borderLeft:'2px solid rgba(200,169,110,0.3)' },
  reasoningLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#c8a96e', letterSpacing:'0.12em', textTransform:'uppercase', marginBottom:8 },
  reasoningText: { fontSize:12, color:'#4a4540', lineHeight:1.7 },

  error: { color:'#ef4444', fontSize:12, marginBottom:16, fontFamily:"'JetBrains Mono',monospace" },
};
