import React, { useState, useEffect } from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';
import { api } from '../api';

export default function BiasAuditChart() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  useEffect(() => { loadAudit(); }, []);

  async function loadAudit() {
    setLoading(true);
    try { setData(await api.biasAudit()); } catch(e) {}
    setLoading(false);
  }

  async function runAudit() {
    setRunning(true);
    try {
      await api.biasRun(100);
      await new Promise(r => setTimeout(r, 2000));
      await loadAudit();
    } catch(e) {}
    setRunning(false);
  }

  // Normalise metrics to 0-100 for radar
  const radarData = data && !data.message ? [
    { dim:'Pop Bias (Top 10%)',  value: (1 - (data.pop_conc_top10pct_at10 || 0)) * 100 },
    { dim:'Pop Bias (Top 20%)',  value: (1 - (data.pop_conc_top20pct_at10 || 0)) * 100 },
    { dim:'Genre Diversity',     value: Math.min(100, (data.genre_entropy || 0) * 20) },
    { dim:'Avg Popularity',      value: Math.max(0, 100 - Math.min(100, (data.avg_pop_at10 || 0) / 10)) },
    { dim:'Genre Count',         value: Math.min(100, (data.n_genres || 0) * 5) },
  ] : [];

  return (
    <div>
      <div style={styles.header}>
        <div>
          <h3 style={styles.title}>Bias Audit</h3>
          {data?.timestamp && <p style={styles.ts}>Last run: {new Date(data.timestamp).toLocaleString()}</p>}
        </div>
        <button style={styles.btn} onClick={runAudit} disabled={running}>
          {running ? 'Running…' : 'Run Audit (100 users)'}
        </button>
      </div>

      {loading && <p style={styles.dim}>Loading…</p>}

      {radarData.length > 0 && (
        <div style={styles.radarBox}>
          <p style={styles.hint}>Higher = better (less biased / more diverse)</p>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#1e293b" />
              <PolarAngleAxis dataKey="dim" tick={{ fill:'#64748b', fontSize:12 }} />
              <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.35} />
              <Tooltip
                contentStyle={{ background:'#1e293b', border:'1px solid #334155', borderRadius:8 }}
                formatter={v => `${v.toFixed(1)} / 100`}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {data && !data.message && (
        <div style={styles.metricGrid}>
          <Metric label="Pop conc. (top 10%)" value={((data.pop_conc_top10pct_at10 || 0)*100).toFixed(1) + '%'} />
          <Metric label="Pop conc. (top 20%)" value={((data.pop_conc_top20pct_at10 || 0)*100).toFixed(1) + '%'} />
          <Metric label="Genre count"         value={data.n_genres || '—'} />
          <Metric label="Genre entropy"       value={(data.genre_entropy || 0).toFixed(2)} />
        </div>
      )}

      {data?.genre_dist && (
        <div style={styles.genreBox}>
          <h4 style={styles.subTitle}>Genre Distribution in Top-10 Recs</h4>
          {Object.entries(data.genre_dist).map(([g, v]) => (
            <div key={g} style={styles.genreRow}>
              <span style={styles.genreLabel}>{g}</span>
              <div style={styles.genreBar}>
                <div style={{ ...styles.genreBarFill, width: `${v * 100}%` }} />
              </div>
              <span style={styles.genrePct}>{(v * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      {data?.message && <p style={styles.dim}>{data.message}</p>}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div style={{ background:'#0f172a', borderRadius:10, padding:'12px 10px', textAlign:'center' }}>
      <div style={{ fontSize:20, fontWeight:700, color:'#818cf8' }}>{value}</div>
      <div style={{ fontSize:12, color:'#64748b', marginTop:2 }}>{label}</div>
    </div>
  );
}

const styles = {
  header:       { display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:20 },
  title:        { margin:0, fontSize:16, color:'#e2e8f0' },
  ts:           { margin:'4px 0 0', fontSize:12, color:'#475569' },
  btn:          { padding:'8px 16px', borderRadius:8, border:'none', background:'#6366f1', color:'#fff', cursor:'pointer', fontSize:13 },
  radarBox:     { background:'#0f172a', borderRadius:12, padding:16, marginBottom:20 },
  hint:         { fontSize:12, color:'#475569', margin:'0 0 8px' },
  metricGrid:   { display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(140px, 1fr))', gap:12, marginBottom:20 },
  genreBox:     { background:'#0f172a', borderRadius:12, padding:16 },
  subTitle:     { margin:'0 0 12px', fontSize:14, color:'#94a3b8' },
  genreRow:     { display:'flex', alignItems:'center', gap:8, marginBottom:6 },
  genreLabel:   { color:'#64748b', fontSize:12, minWidth:120 },
  genreBar:     { flex:1, height:6, background:'#1e293b', borderRadius:3 },
  genreBarFill: { height:'100%', background:'#6366f1', borderRadius:3, transition:'width 0.5s' },
  genrePct:     { color:'#475569', fontSize:12, minWidth:40, textAlign:'right' },
  dim:          { color:'#475569', fontSize:14 },
};
