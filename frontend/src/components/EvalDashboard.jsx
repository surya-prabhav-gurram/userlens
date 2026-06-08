import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../api';

export default function EvalDashboard() {
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading]   = useState(false);
  const [running, setRunning]   = useState(false);

  useEffect(() => { loadLatest(); }, []);

  async function loadLatest() {
    setLoading(true);
    try { setEvalData(await api.evalLatest()); } catch(e) {}
    setLoading(false);
  }

  async function runEval() {
    setRunning(true);
    try {
      await api.evalRun(50);
      await new Promise(r => setTimeout(r, 3000));
      await loadLatest();
    } catch(e) {}
    setRunning(false);
  }

  const metrics = evalData && !evalData.message ? [
    { key: 'NDCG@10',    val: evalData['ndcg@10']    || 0 },
    { key: 'NDCG@20',    val: evalData['ndcg@20']    || 0 },
    { key: 'HitRate@10', val: evalData['hitrate@10'] || 0 },
    { key: 'HitRate@20', val: evalData['hitrate@20'] || 0 },
    { key: 'MRR@10',     val: evalData['mrr@10']     || 0 },
    { key: 'Coverage',   val: evalData['coverage']   || 0 },
  ] : [];

  return (
    <div>
      <style>{`
        .run-btn:hover { background: #b89858 !important; }
      `}</style>

      <div style={S.header}>
        <div>
          <h2 style={S.pageTitle}>Evaluation Metrics</h2>
          {evalData?.timestamp && (
            <p style={S.pageSubtitle}>Last run: {new Date(evalData.timestamp).toLocaleString()}</p>
          )}
        </div>
        <button className="run-btn" style={S.runBtn} onClick={runEval} disabled={running}>
          {running ? 'Running...' : 'Run Eval (50 users)'}
        </button>
      </div>

      {loading && <div style={S.dim}>Loading metrics...</div>}

      {metrics.length > 0 && (
        <>
          <div style={S.metricGrid}>
            {metrics.map(m => (
              <div key={m.key} style={S.metricCard}>
                <div style={S.metricVal}>{m.val.toFixed(4)}</div>
                <div style={S.metricKey}>{m.key}</div>
                <div style={S.metricBar}>
                  <div style={{ ...S.metricBarFill, width:`${Math.min(m.val * 100 * 4, 100)}%` }} />
                </div>
              </div>
            ))}
          </div>

          <div style={S.chartWrap}>
            <div style={S.chartLabel}>Metric Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={metrics} margin={{ top:10, right:0, bottom:20, left:0 }}>
                <XAxis dataKey="key" tick={{ fill:'#3a3530', fontSize:10, fontFamily:"'JetBrains Mono',monospace" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill:'#3a3530', fontSize:10, fontFamily:"'JetBrains Mono',monospace" }} domain={[0,1]} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background:'#0d0d0d', border:'1px solid #1a1a1a', borderRadius:0, fontFamily:"'JetBrains Mono',monospace", fontSize:11 }}
                  labelStyle={{ color:'#c8a96e' }}
                  formatter={v => [v.toFixed(4), '']}
                  cursor={{ fill:'rgba(200,169,110,0.04)' }}
                />
                <Bar dataKey="val" fill="#c8a96e" radius={0} maxBarSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {evalData.n_users_evaluated && (
            <div style={S.footer}>Evaluated on {evalData.n_users_evaluated} test users</div>
          )}
        </>
      )}
    </div>
  );
}

const S = {
  header: { display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:40 },
  pageTitle: { fontSize:28, fontWeight:800, letterSpacing:'-0.02em', color:'#e8e0d4' },
  pageSubtitle: { fontSize:11, color:'#3a3530', marginTop:4, fontFamily:"'JetBrains Mono',monospace" },
  runBtn: {
    padding:'10px 24px',
    background:'#c8a96e',
    border:'none',
    color:'#080808',
    fontSize:11,
    fontFamily:"'Syne',sans-serif",
    fontWeight:700,
    letterSpacing:'0.06em',
    textTransform:'uppercase',
    cursor:'pointer',
    transition:'background 0.15s',
    borderRadius:0,
  },
  dim: { color:'#2a2520', fontSize:12, fontFamily:"'JetBrains Mono',monospace" },
  metricGrid: { display:'grid', gridTemplateColumns:'repeat(6,1fr)', gap:1, background:'#1a1a1a', marginBottom:1 },
  metricCard: { background:'#0d0d0d', padding:'20px 16px' },
  metricVal: { fontFamily:"'JetBrains Mono',monospace", fontSize:20, color:'#c8a96e', fontWeight:500, marginBottom:4 },
  metricKey: { fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#3a3530', letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:10 },
  metricBar: { height:2, background:'#111' },
  metricBarFill: { height:'100%', background:'#c8a96e', transition:'width 0.8s ease' },
  chartWrap: { background:'#0d0d0d', border:'1px solid #1a1a1a', padding:'24px 20px', marginTop:1 },
  chartLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#3a3530', letterSpacing:'0.12em', textTransform:'uppercase', marginBottom:16 },
  footer: { fontSize:11, color:'#2a2520', fontFamily:"'JetBrains Mono',monospace", marginTop:16 },
};
