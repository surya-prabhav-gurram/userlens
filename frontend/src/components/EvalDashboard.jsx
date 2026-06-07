import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts';
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
      await new Promise(r => setTimeout(r, 2000));
      await loadLatest();
    } catch(e) {}
    setRunning(false);
  }

  const barData = evalData ? [
    { name: 'NDCG@10',    value: evalData['ndcg@10']    || 0 },
    { name: 'NDCG@20',    value: evalData['ndcg@20']    || 0 },
    { name: 'HitRate@10', value: evalData['hitrate@10'] || 0 },
    { name: 'HitRate@20', value: evalData['hitrate@20'] || 0 },
    { name: 'MRR@10',     value: evalData['mrr@10']     || 0 },
    { name: 'Coverage',   value: evalData['coverage']   || 0 },
  ] : [];

  return (
    <div>
      <div style={styles.header}>
        <div>
          <h3 style={styles.title}>Evaluation Metrics</h3>
          {evalData?.timestamp && (
            <p style={styles.ts}>Last run: {new Date(evalData.timestamp).toLocaleString()}</p>
          )}
        </div>
        <button style={styles.btn} onClick={runEval} disabled={running}>
          {running ? 'Running…' : 'Run Eval (50 users)'}
        </button>
      </div>

      {loading && <p style={styles.dim}>Loading…</p>}

      {evalData && !evalData.message && (
        <>
          <div style={styles.metricGrid}>
            {barData.map(m => (
              <div key={m.name} style={styles.metricCard}>
                <div style={styles.metricValue}>{m.value.toFixed(4)}</div>
                <div style={styles.metricName}>{m.name}</div>
              </div>
            ))}
          </div>

          <div style={styles.chartBox}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={barData} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                <XAxis dataKey="name" tick={{ fill:'#64748b', fontSize:12 }} />
                <YAxis tick={{ fill:'#64748b', fontSize:12 }} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ background:'#1e293b', border:'1px solid #334155', borderRadius:8 }}
                  labelStyle={{ color:'#e2e8f0' }}
                  formatter={v => v.toFixed(4)}
                />
                <Bar dataKey="value" fill="#6366f1" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {evalData.n_users_evaluated && (
            <p style={styles.dim}>Evaluated on {evalData.n_users_evaluated} test users</p>
          )}
        </>
      )}

      {evalData?.message && <p style={styles.dim}>{evalData.message}</p>}
    </div>
  );
}

const styles = {
  header:      { display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:20 },
  title:       { margin:0, fontSize:16, color:'#e2e8f0' },
  ts:          { margin:'4px 0 0', fontSize:12, color:'#475569' },
  btn:         { padding:'8px 16px', borderRadius:8, border:'none', background:'#6366f1', color:'#fff', cursor:'pointer', fontSize:13 },
  metricGrid:  { display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(120px, 1fr))', gap:12, marginBottom:24 },
  metricCard:  { background:'#0f172a', borderRadius:10, padding:'14px 10px', textAlign:'center' },
  metricValue: { fontSize:22, fontWeight:700, color:'#818cf8', marginBottom:4 },
  metricName:  { fontSize:12, color:'#64748b' },
  chartBox:    { background:'#0f172a', borderRadius:12, padding:'16px 8px' },
  dim:         { color:'#475569', fontSize:14 },
};
