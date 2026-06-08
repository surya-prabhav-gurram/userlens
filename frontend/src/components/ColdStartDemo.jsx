import React, { useState } from 'react';
import { api } from '../api';

export default function ColdStartDemo() {
  const [inputId, setInputId]   = useState('');
  const [history, setHistory]   = useState([]);
  const [recs, setRecs]         = useState([]);
  const [pathway, setPathway]   = useState('');
  const [loading, setLoading]   = useState(false);

  async function fetchRecs(hist) {
    if (!hist.length) { setRecs([]); setPathway(''); return; }
    setLoading(true);
    try {
      const r = await api.recommend({ sequence: hist.map(Number), k: 10, mode: 'neural' });
      setRecs(r.items || []);
      setPathway(r.pathway || '');
    } catch(e) {}
    setLoading(false);
  }

  function addItem() {
    const id = parseInt(inputId);
    if (!id || isNaN(id)) return;
    const next = [...history, id];
    setHistory(next);
    setInputId('');
    fetchRecs(next);
  }

  function reset() {
    setHistory([]);
    setRecs([]);
    setPathway('');
    setInputId('');
  }

  const isCold = pathway === 'cold_content';

  return (
    <div>
      <style>{`
        .add-btn:hover { background: #b89858 !important; }
        .reset-btn:hover { border-color: #3a3530 !important; color: #6a6560 !important; }
        .cold-input:focus { outline: none; border-color: #c8a96e !important; }
      `}</style>

      <div style={S.header}>
        <div>
          <h2 style={S.pageTitle}>Cold Start Demo</h2>
          <p style={S.pageSubtitle}>Add items one by one — watch the pathway switch at 3 interactions</p>
        </div>
      </div>

      {/* Input */}
      <div style={S.inputRow}>
        <input
          className="cold-input"
          style={S.input}
          type="number"
          min={3} max={32722}
          placeholder="Item ID (3–32722)"
          value={inputId}
          onChange={e => setInputId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && addItem()}
        />
        <button className="add-btn" style={S.addBtn} onClick={addItem}>Add Item</button>
        <button className="reset-btn" style={S.resetBtn} onClick={reset}>Reset</button>
      </div>

      {/* History chips */}
      {history.length > 0 && (
        <div style={S.histChips}>
          <span style={S.histChipsLabel}>Current history ({history.length} items):</span>
          <div style={S.chips}>
            {history.map((id, i) => (
              <span key={i} style={S.chip}>Item {id}</span>
            ))}
          </div>
        </div>
      )}

      {/* Pathway indicator */}
      {pathway && (
        <div style={{ ...S.pathwayBanner, ...(isCold ? S.pathwayCold : S.pathwayWarm) }}>
          <div style={S.pathwayLine} />
          <div style={S.pathwayContent}>
            <span style={S.pathwayName}>{isCold ? 'Cold Start' : 'Warm Neural'}</span>
            <span style={S.pathwayDesc}>
              {isCold
                ? 'Content-based fallback (< 3 interactions)'
                : 'Neural two-tower retrieval (>= 3 interactions)'}
            </span>
          </div>
        </div>
      )}

      {/* Results */}
      {loading && <div style={S.dim}>Fetching recommendations...</div>}
      {!loading && recs.length > 0 && (
        <div style={S.recGrid}>
          {recs.map((item, i) => (
            <div key={item.item_id} style={S.recCard}>
              <div style={S.cardRank}>{String(i + 1).padStart(2, '0')}</div>
              <div style={S.cardTitle}>{item.title}</div>
              <div style={S.cardScore}>{item.score.toFixed(3)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const S = {
  header: { marginBottom:32 },
  pageTitle: { fontSize:28, fontWeight:800, letterSpacing:'-0.02em', color:'#e8e0d4' },
  pageSubtitle: { fontSize:11, color:'#3a3530', marginTop:4, fontFamily:"'JetBrains Mono',monospace" },

  inputRow: { display:'flex', gap:0, marginBottom:20 },
  input: {
    flex:1, maxWidth:320,
    padding:'10px 16px',
    background:'#0d0d0d',
    border:'1px solid #1a1a1a',
    borderRight:'none',
    color:'#e8e0d4',
    fontSize:13,
    fontFamily:"'JetBrains Mono',monospace",
    borderRadius:0,
    transition:'border-color 0.15s',
  },
  addBtn: {
    padding:'10px 24px', background:'#c8a96e', border:'none',
    color:'#080808', fontSize:11, fontFamily:"'Syne',sans-serif", fontWeight:700,
    letterSpacing:'0.06em', textTransform:'uppercase', cursor:'pointer', borderRadius:0,
  },
  resetBtn: {
    padding:'10px 20px', background:'transparent', border:'1px solid #1a1a1a', borderLeft:'none',
    color:'#3a3530', fontSize:11, fontFamily:"'Syne',sans-serif", fontWeight:600,
    letterSpacing:'0.06em', textTransform:'uppercase', cursor:'pointer', borderRadius:0,
    transition:'all 0.15s',
  },

  histChips: { marginBottom:24 },
  histChipsLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#3a3530', letterSpacing:'0.1em', textTransform:'uppercase', display:'block', marginBottom:10 },
  chips: { display:'flex', flexWrap:'wrap', gap:6 },
  chip: { padding:'4px 10px', background:'#111', border:'1px solid #1a1a1a', fontSize:11, color:'#4a4540', fontFamily:"'JetBrains Mono',monospace" },

  pathwayBanner: { display:'flex', alignItems:'stretch', gap:0, marginBottom:28, overflow:'hidden' },
  pathwayCold: { background:'rgba(200,169,110,0.04)', border:'1px solid rgba(200,169,110,0.15)' },
  pathwayWarm: { background:'rgba(74,222,128,0.04)', border:'1px solid rgba(74,222,128,0.15)' },
  pathwayLine: { width:3, background:'#c8a96e', flexShrink:0 },
  pathwayContent: { padding:'16px 20px' },
  pathwayName: { display:'block', fontSize:14, fontWeight:700, color:'#e8e0d4', letterSpacing:'0.02em', marginBottom:4 },
  pathwayDesc: { fontFamily:"'JetBrains Mono',monospace", fontSize:10, color:'#3a3530' },

  dim: { color:'#2a2520', fontSize:12, fontFamily:"'JetBrains Mono',monospace", marginTop:16 },
  recGrid: { display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:1, background:'#1a1a1a' },
  recCard: { background:'#0d0d0d', padding:'14px 16px', display:'flex', alignItems:'center', gap:12 },
  cardRank: { fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:'#c8a96e', minWidth:22, flexShrink:0 },
  cardTitle: { fontSize:12, color:'#6a6560', flex:1, lineHeight:1.4 },
  cardScore: { fontFamily:"'JetBrains Mono',monospace", fontSize:10, color:'#2a2520', flexShrink:0 },
};
