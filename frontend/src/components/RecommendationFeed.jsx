import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Zap, Brain, Layers } from 'lucide-react';

const MODE_OPTIONS = [
  { value: 'neural', label: 'Neural', icon: <Zap size={14}/> },
  { value: 'llm',    label: 'LLM Re-rank', icon: <Brain size={14}/> },
  { value: 'hybrid', label: 'Hybrid', icon: <Layers size={14}/> },
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

  async function fetchRecs(uid) {
    if (!uid) return;
    setLoading(true); setError('');
    try {
      const [hist, rec] = await Promise.all([
        api.userHistory(uid),
        api.recommend({ user_id: parseInt(uid), k: 10, mode }),
      ]);
      setHistory(hist.history || []);
      setRecs(rec.items || []);
      setReasoning(rec.reasoning || '');
      setPathway(rec.pathway || '');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleLike(itemId) {
    setLiked(s => new Set([...s, itemId]));
    await api.feedback({ user_id: parseInt(userId), item_id: itemId, interaction_type: 'like' }).catch(()=>{});
  }

  const maxScore = recs.length ? Math.max(...recs.map(r => r.score)) : 1;

  return (
    <div style={styles.container}>
      {/* Controls */}
      <div style={styles.controls}>
        <select
          style={styles.select}
          value={userId}
          onChange={e => { setUserId(e.target.value); fetchRecs(e.target.value); }}
        >
          <option value="">Select a user…</option>
          {(users || []).map(u => (
            <option key={u.user_id} value={u.user_id}>
              User {u.user_id} ({u.n_interactions} interactions)
            </option>
          ))}
        </select>

        <div style={styles.modeGroup}>
          {MODE_OPTIONS.map(m => (
            <button
              key={m.value}
              style={{ ...styles.modeBtn, ...(mode === m.value ? styles.modeBtnActive : {}) }}
              onClick={() => { setMode(m.value); if (userId) fetchRecs(userId); }}
            >
              {m.icon} {m.label}
            </button>
          ))}
        </div>
      </div>

      {error && <div style={styles.error}>{error}</div>}

      <div style={styles.panels}>
        {/* History */}
        <div style={styles.panel}>
          <h3 style={styles.panelTitle}>Interaction History</h3>
          {history.length === 0 && <p style={styles.empty}>Select a user to view history</p>}
          {history.map((item, i) => (
            <div key={i} style={styles.histItem}>
              <span style={styles.histRank}>#{history.length - i}</span>
              <span style={styles.itemTitle}>{item.title}</span>
            </div>
          ))}
        </div>

        {/* Recommendations */}
        <div style={styles.panel}>
          <h3 style={styles.panelTitle}>
            Recommendations
            {pathway && <span style={styles.pathwayBadge}>{pathway}</span>}
          </h3>
          {loading && <p style={styles.empty}>Generating recommendations…</p>}
          {!loading && recs.length === 0 && <p style={styles.empty}>No recommendations yet</p>}
          {recs.map((item, i) => (
            <div key={item.item_id} style={styles.recItem}>
              <div style={styles.recHeader}>
                <span style={styles.recRank}>#{i + 1}</span>
                <span style={styles.itemTitle}>{item.title}</span>
                <button
                  style={{ ...styles.likeBtn, ...(liked.has(item.item_id) ? styles.likeBtnActive : {}) }}
                  onClick={() => handleLike(item.item_id)}
                >♥</button>
              </div>
              <div style={styles.scoreBar}>
                <div style={{ ...styles.scoreBarFill, width: `${(item.score / maxScore) * 100}%` }} />
              </div>
              <span style={styles.scoreLabel}>{item.score.toFixed(3)}</span>
            </div>
          ))}
          {reasoning && (
            <div style={styles.reasoning}>
              <strong>Reasoning:</strong> {reasoning}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: { padding: 0 },
  controls:  { display:'flex', gap:12, flexWrap:'wrap', marginBottom:20, alignItems:'center' },
  select:    { padding:'8px 12px', borderRadius:8, border:'1px solid #334155', background:'#1e293b', color:'#e2e8f0', fontSize:14, minWidth:220 },
  modeGroup: { display:'flex', gap:6 },
  modeBtn:   { display:'flex', alignItems:'center', gap:4, padding:'6px 14px', borderRadius:8, border:'1px solid #334155', background:'#1e293b', color:'#94a3b8', cursor:'pointer', fontSize:13 },
  modeBtnActive: { background:'#6366f1', border:'1px solid #6366f1', color:'#fff' },
  panels:    { display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 },
  panel:     { background:'#1e293b', borderRadius:12, padding:16 },
  panelTitle:{ margin:'0 0 12px', fontSize:15, color:'#e2e8f0', display:'flex', alignItems:'center', gap:8 },
  pathwayBadge: { fontSize:11, padding:'2px 8px', borderRadius:12, background:'#312e81', color:'#a5b4fc', marginLeft:8 },
  empty:     { color:'#475569', fontSize:14 },
  histItem:  { display:'flex', gap:8, padding:'6px 0', borderBottom:'1px solid #0f172a', alignItems:'center' },
  histRank:  { color:'#475569', fontSize:12, minWidth:28 },
  itemTitle: { color:'#cbd5e1', fontSize:14, flex:1 },
  recItem:   { padding:'8px 0', borderBottom:'1px solid #0f172a' },
  recHeader: { display:'flex', alignItems:'center', gap:8, marginBottom:4 },
  recRank:   { color:'#6366f1', fontSize:13, fontWeight:700, minWidth:28 },
  scoreBar:  { height:4, background:'#0f172a', borderRadius:2, marginBottom:2 },
  scoreBarFill: { height:'100%', background:'linear-gradient(90deg,#6366f1,#818cf8)', borderRadius:2, transition:'width 0.5s' },
  scoreLabel:{ fontSize:11, color:'#475569' },
  likeBtn:   { background:'transparent', border:'none', color:'#475569', cursor:'pointer', fontSize:16 },
  likeBtnActive: { color:'#f43f5e' },
  reasoning: { marginTop:12, padding:10, background:'#0f172a', borderRadius:8, fontSize:13, color:'#94a3b8', lineHeight:1.5 },
  error:     { color:'#f87171', marginBottom:12, fontSize:14 },
};
