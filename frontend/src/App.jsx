import React, { useState, useEffect } from 'react';
import RecommendationFeed from './components/RecommendationFeed';
import EvalDashboard from './components/EvalDashboard';
import BiasAuditChart from './components/BiasAuditChart';
import ColdStartDemo from './components/ColdStartDemo';
import { api } from './api';

const TABS = [
  { id: 'feed',   label: '🎬 Recommendations' },
  { id: 'eval',   label: '📊 Evaluation' },
  { id: 'bias',   label: '⚖️ Bias Audit' },
  { id: 'cold',   label: '❄️ Cold Start' },
];

export default function App() {
  const [tab, setTab]         = useState('feed');
  const [health, setHealth]   = useState(null);
  const [users, setUsers]     = useState([]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
    api.listUsers(50).then(d => setUsers(d.users || [])).catch(() => {});
  }, []);

  return (
    <div style={styles.root}>
      {/* Sidebar */}
      <aside style={styles.sidebar}>
        <div style={styles.logo}>
          <div style={styles.logoIcon}>UL</div>
          <div>
            <div style={styles.logoName}>UserLens</div>
            <div style={styles.logoSub}>Adaptive RecSys</div>
          </div>
        </div>

        <nav style={styles.nav}>
          {TABS.map(t => (
            <button
              key={t.id}
              style={{ ...styles.navBtn, ...(tab === t.id ? styles.navBtnActive : {}) }}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {health && (
          <div style={styles.healthBox}>
            <div style={styles.healthRow}>
              <span style={styles.healthDot} />
              <span style={styles.healthLabel}>API Online</span>
            </div>
            <div style={styles.healthStat}>{health.n_items?.toLocaleString()} items</div>
            <div style={styles.healthStat}>{health.n_users?.toLocaleString()} users</div>
            <div style={styles.healthStat}>Model: {health.model_version}</div>
            <div style={styles.healthStat}>DB: {health.db_status}</div>
          </div>
        )}

        <div style={styles.stack}>
          <div style={styles.stackTitle}>Stack</div>
          {['BERT4Rec', 'Two-Tower', 'Cross-Attn Ranker', 'IPS Debiasing', 'Claude LLM'].map(s => (
            <span key={s} style={styles.stackChip}>{s}</span>
          ))}
        </div>
      </aside>

      {/* Main */}
      <main style={styles.main}>
        <div style={styles.content}>
          {tab === 'feed' && <RecommendationFeed users={users} />}
          {tab === 'eval' && <EvalDashboard />}
          {tab === 'bias' && <BiasAuditChart />}
          {tab === 'cold' && <ColdStartDemo />}
        </div>
      </main>
    </div>
  );
}

const styles = {
  root:       { display:'flex', minHeight:'100vh', background:'#0f172a', fontFamily:'"Inter",system-ui,sans-serif', color:'#e2e8f0' },
  sidebar:    { width:220, background:'#1e293b', padding:24, display:'flex', flexDirection:'column', gap:24, flexShrink:0 },
  logo:       { display:'flex', alignItems:'center', gap:10 },
  logoIcon:   { width:40, height:40, borderRadius:10, background:'linear-gradient(135deg,#6366f1,#818cf8)', display:'flex', alignItems:'center', justifyContent:'center', fontWeight:800, fontSize:14, color:'#fff' },
  logoName:   { fontWeight:700, fontSize:16, color:'#e2e8f0' },
  logoSub:    { fontSize:11, color:'#475569' },
  nav:        { display:'flex', flexDirection:'column', gap:4 },
  navBtn:     { padding:'9px 12px', borderRadius:8, border:'none', background:'transparent', color:'#64748b', cursor:'pointer', textAlign:'left', fontSize:14, transition:'all 0.15s' },
  navBtnActive: { background:'#312e81', color:'#a5b4fc' },
  healthBox:  { background:'#0f172a', borderRadius:10, padding:12 },
  healthRow:  { display:'flex', alignItems:'center', gap:6, marginBottom:6 },
  healthDot:  { width:8, height:8, borderRadius:'50%', background:'#22c55e' },
  healthLabel:{ fontSize:13, color:'#94a3b8' },
  healthStat: { fontSize:12, color:'#475569', marginTop:2 },
  stack:      { marginTop:'auto' },
  stackTitle: { fontSize:11, color:'#475569', textTransform:'uppercase', letterSpacing:'0.05em', marginBottom:8 },
  stackChip:  { display:'block', fontSize:11, color:'#64748b', padding:'3px 0', borderBottom:'1px solid #1e293b' },
  main:       { flex:1, overflow:'auto', padding:32 },
  content:    { maxWidth:1100, margin:'0 auto' },
};
