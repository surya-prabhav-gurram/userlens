import React, { useState } from 'react';
import { api } from '../api';

export default function ColdStartDemo({ items }) {
  const [sequence, setSequence]   = useState([]);
  const [input, setInput]         = useState('');
  const [recs, setRecs]           = useState([]);
  const [pathway, setPathway]     = useState('');
  const [loading, setLoading]     = useState(false);

  async function addItem() {
    const id = parseInt(input);
    if (isNaN(id)) return;
    const newSeq = [...sequence, id];
    setSequence(newSeq);
    setInput('');
    await getRecs(newSeq);
  }

  async function getRecs(seq) {
    setLoading(true);
    try {
      const res = await api.recommend({ sequence: seq, k: 10, mode: 'neural' });
      setRecs(res.items || []);
      setPathway(res.pathway || '');
    } catch(e) {}
    setLoading(false);
  }

  function reset() { setSequence([]); setRecs([]); setPathway(''); }

  const pathwayColor = pathway === 'cold_content' ? '#f59e0b' : pathway === 'warm_neural' ? '#22c55e' : '#6366f1';

  return (
    <div>
      <h3 style={styles.title}>Cold-Start Demo</h3>
      <p style={styles.desc}>
        Start with an empty history. Add items one-by-one and watch the system switch
        from content-based cold-start retrieval to neural recommendations as history grows.
      </p>

      <div style={styles.controls}>
        <input
          style={styles.input}
          type="number"
          placeholder="Item ID (3–1000)"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && addItem()}
        />
        <button style={styles.btn} onClick={addItem}>Add Item</button>
        <button style={styles.resetBtn} onClick={reset}>Reset</button>
      </div>

      {sequence.length > 0 && (
        <div style={styles.seqBox}>
          <span style={styles.seqLabel}>Current history ({sequence.length} items):</span>
          <div style={styles.seqItems}>
            {sequence.map((id, i) => (
              <span key={i} style={styles.seqChip}>Item {id}</span>
            ))}
          </div>
        </div>
      )}

      {pathway && (
        <div style={{ ...styles.pathwayBanner, borderColor: pathwayColor }}>
          <span style={{ color: pathwayColor, fontWeight:700 }}>
            {pathway === 'cold_content' ? '❄️ Cold Start' : pathway === 'warm_neural' ? '🔥 Warm Neural' : '🔀 Blend'}
          </span>
          <span style={styles.pathwayDesc}>
            {pathway === 'cold_content'
              ? ` — Content-based fallback (< 3 interactions)`
              : ` — Neural two-tower retrieval (≥ 3 interactions)`}
          </span>
        </div>
      )}

      {loading && <p style={styles.dim}>Retrieving recommendations…</p>}

      {recs.length > 0 && (
        <div style={styles.recGrid}>
          {recs.map((item, i) => (
            <div key={item.item_id} style={styles.recCard}>
              <span style={styles.recRank}>#{i+1}</span>
              <span style={styles.recTitle}>{item.title}</span>
              <span style={styles.recScore}>{item.score.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}

      {!loading && recs.length === 0 && (
        <p style={styles.dim}>Add at least one item to get recommendations.</p>
      )}
    </div>
  );
}

const styles = {
  title:        { margin:'0 0 8px', fontSize:16, color:'#e2e8f0' },
  desc:         { color:'#64748b', fontSize:14, marginBottom:20 },
  controls:     { display:'flex', gap:8, marginBottom:16 },
  input:        { flex:1, padding:'8px 12px', borderRadius:8, border:'1px solid #334155', background:'#1e293b', color:'#e2e8f0', fontSize:14 },
  btn:          { padding:'8px 16px', borderRadius:8, border:'none', background:'#6366f1', color:'#fff', cursor:'pointer' },
  resetBtn:     { padding:'8px 16px', borderRadius:8, border:'1px solid #334155', background:'transparent', color:'#94a3b8', cursor:'pointer' },
  seqBox:       { background:'#0f172a', borderRadius:10, padding:12, marginBottom:16 },
  seqLabel:     { color:'#64748b', fontSize:13, display:'block', marginBottom:8 },
  seqItems:     { display:'flex', flexWrap:'wrap', gap:6 },
  seqChip:      { padding:'4px 10px', background:'#1e293b', borderRadius:20, fontSize:12, color:'#94a3b8' },
  pathwayBanner:{ border:'1px solid', borderRadius:10, padding:'10px 14px', marginBottom:16 },
  pathwayDesc:  { color:'#64748b', fontSize:13 },
  recGrid:      { display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 },
  recCard:      { background:'#1e293b', borderRadius:8, padding:'10px 12px', display:'flex', alignItems:'center', gap:8 },
  recRank:      { color:'#6366f1', fontWeight:700, fontSize:13, minWidth:24 },
  recTitle:     { flex:1, color:'#cbd5e1', fontSize:13 },
  recScore:     { color:'#475569', fontSize:12 },
  dim:          { color:'#475569', fontSize:14 },
};
