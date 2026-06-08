import React, { useState, useEffect } from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';
import { api } from '../api';

export default function BiasAuditChart() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  useEffect(() => { load(); }, []);

  async function load() {
    setLoading(true);
    try { setData(await api.biasLatest()); } catch(e) {}
    setLoading(false);
  }

  async function run() {
    setRunning(true);
    try {
      await api.biasRun(100);
      await new Promise(r => setTimeout(r, 3000));
      await load();
    } catch(e) {}
    setRunning(false);
  }

  const radarData = data ? [
    { axis: 'Pop Bias (Top 10%)', value: 1 - (data['pop_conc_top10pct@10'] || 0) },
    { axis: 'Pop Bias (Top 20%)', value: 1 - (data['pop_conc_top20pct@10'] || 0) },
    { axis: 'Genre Diversity',    value: Math.min((data['genre_entropy@10'] || 0) / 3, 1) },
    { axis: 'Avg Popularity',     value: 1 - Math.min((data['avg_pop@10'] || 0) / 20, 1) },
    { axis: 'Genre Count',        value: Math.min((data['avg_genre_count@10'] || 0) / 5, 1) },
  ] : [];

  const stats = data ? [
    { label:'Pop conc. (top 10%)', val: (data['pop_conc_top10pct@10'] || 0).toFixed(3) },
    { label:'Pop conc. (top 20%)', val: (data['pop_conc_top20pct@10'] || 0).toFixed(3) },
    { label:'Genre count',         val: data['avg_genre_count@10'] != null ? data['avg_genre_count@10'].toFixed(2) : '—' },
    { label:'Genre entropy',       val: data['genre_entropy@10'] != null ? data['genre_entropy@10'].toFixed(3) : '0.00' },
  ] : [];

  return (
    <div>
      <style>{`.audit-btn:hover { background: #b89858 !important; }`}</style>

      <div style={S.header}>
        <div>
          <h2 style={S.pageTitle}>Bias Audit</h2>
          {data?.timestamp && (
            <p style={S.pageSubtitle}>Last run: {new Date(data.timestamp).toLocaleString()}</p>
          )}
        </div>
        <button className="audit-btn" style={S.runBtn} onClick={run} disabled={running}>
          {running ? 'Running...' : 'Run Audit (100 users)'}
        </button>
      </div>

      {loading && <div style={S.dim}>Loading audit data...</div>}

      {radarData.length > 0 && (
        <div style={S.body}>
          <div style={S.radarWrap}>
            <div style={S.chartLabel}>Higher = less biased / more diverse</div>
            <ResponsiveContainer width="100%" height={340}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#1a1a1a" />
                <PolarAngleAxis
                  dataKey="axis"
                  tick={{ fill:'#3a3530', fontSize:10, fontFamily:"'JetBrains Mono',monospace" }}
                />
                <Radar dataKey="value" stroke="#c8a96e" fill="#c8a96e" fillOpacity={0.15} strokeWidth={1.5} />
                <Tooltip
                  contentStyle={{ background:'#0d0d0d', border:'1px solid #1a1a1a', fontFamily:"'JetBrains Mono',monospace", fontSize:11 }}
                  formatter={v => [v.toFixed(3), '']}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div style={S.statsCol}>
            <div style={S.statsLabel}>Audit Results</div>
            {stats.map(s => (
              <div key={s.label} style={S.statRow}>
                <span style={S.statKey}>{s.label}</span>
                <span style={S.statVal}>{s.val}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const S = {
  header: { display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:40 },
  pageTitle: { fontSize:28, fontWeight:800, letterSpacing:'-0.02em', color:'#e8e0d4' },
  pageSubtitle: { fontSize:11, color:'#3a3530', marginTop:4, fontFamily:"'JetBrains Mono',monospace" },
  runBtn: {
    padding:'10px 24px', background:'#c8a96e', border:'none',
    color:'#080808', fontSize:11, fontFamily:"'Syne',sans-serif", fontWeight:700,
    letterSpacing:'0.06em', textTransform:'uppercase', cursor:'pointer', borderRadius:0,
  },
  dim: { color:'#2a2520', fontSize:12, fontFamily:"'JetBrains Mono',monospace" },
  body: { display:'grid', gridTemplateColumns:'1fr 280px', gap:1, background:'#1a1a1a' },
  radarWrap: { background:'#0d0d0d', padding:'24px 20px' },
  chartLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#3a3530', letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:8 },
  statsCol: { background:'#0d0d0d', padding:'24px 20px' },
  statsLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#3a3530', letterSpacing:'0.12em', textTransform:'uppercase', marginBottom:20 },
  statRow: { display:'flex', justifyContent:'space-between', alignItems:'center', padding:'12px 0', borderBottom:'1px solid #111' },
  statKey: { fontSize:11, color:'#4a4540' },
  statVal: { fontFamily:"'JetBrains Mono',monospace", fontSize:14, color:'#c8a96e', fontWeight:500 },
};
