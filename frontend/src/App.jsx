import React, { useState, useEffect } from 'react';
import RecommendationFeed from './components/RecommendationFeed';
import EvalDashboard from './components/EvalDashboard';
import BiasAuditChart from './components/BiasAuditChart';
import ColdStartDemo from './components/ColdStartDemo';
import { api } from './api';

const TABS = [
  { id: 'feed',  label: 'Recommendations' },
  { id: 'eval',  label: 'Evaluation' },
  { id: 'bias',  label: 'Bias Audit' },
  { id: 'cold',  label: 'Cold Start' },
];

const STACK = ['BERT4Rec', 'Two-Tower', 'Cross-Attn Ranker', 'IPS Debiasing', 'Claude LLM'];

export default function App() {
  const [tab, setTab]       = useState('feed');
  const [health, setHealth] = useState(null);
  const [users, setUsers]   = useState([]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
    api.listUsers(50).then(d => setUsers(d.users || [])).catch(() => {});
  }, []);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #080808; color: #e8e0d4; font-family: 'Syne', sans-serif; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #c8a96e; border-radius: 2px; }
        .nav-btn { transition: all 0.2s; }
        .nav-btn:hover { color: #c8a96e !important; }
        .tab-active { color: #c8a96e !important; border-left: 2px solid #c8a96e !important; padding-left: 14px !important; }
        @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        .fade-in { animation: fadeIn 0.4s ease forwards; }
        .grid-bg {
          background-image: linear-gradient(rgba(200,169,110,0.03) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(200,169,110,0.03) 1px, transparent 1px);
          background-size: 40px 40px;
        }
      `}</style>

      <div style={S.root}>
        {/* Grid background */}
        <div className="grid-bg" style={S.gridBg} />

        {/* Sidebar */}
        <aside style={S.sidebar}>
          {/* Logo */}
          <div style={S.logoArea}>
            <div style={S.logoMark}>
              <span style={S.logoMarkText}>UL</span>
            </div>
            <div>
              <div style={S.logoName}>USERLENS</div>
              <div style={S.logoCredit}>by Surya Prabhav Gurram</div>
            </div>
          </div>

          {/* Divider */}
          <div style={S.divider} />

          {/* Nav */}
          <nav style={S.nav}>
            <div style={S.navSection}>Navigation</div>
            {TABS.map(t => (
              <button
                key={t.id}
                className={`nav-btn ${tab === t.id ? 'tab-active' : ''}`}
                style={{ ...S.navBtn, ...(tab === t.id ? S.navBtnActive : {}) }}
                onClick={() => setTab(t.id)}
              >
                <span style={S.navIndex}>{TABS.indexOf(t) + 1 < 10 ? `0${TABS.indexOf(t)+1}` : TABS.indexOf(t)+1}</span>
                {t.label}
              </button>
            ))}
          </nav>

          <div style={S.divider} />

          {/* Status */}
          {health && (
            <div style={S.statusBox}>
              <div style={S.statusRow}>
                <span style={S.statusDot} />
                <span style={S.statusLabel}>SYSTEM ONLINE</span>
              </div>
              <div style={S.statusGrid}>
                <div style={S.statusItem}>
                  <div style={S.statusVal}>{(health.n_items || 0).toLocaleString()}</div>
                  <div style={S.statusKey}>items</div>
                </div>
                <div style={S.statusItem}>
                  <div style={S.statusVal}>{(health.n_users || 0).toLocaleString()}</div>
                  <div style={S.statusKey}>users</div>
                </div>
              </div>
              <div style={S.modelTag}>
                <span style={S.modelDot} />
                {health.model_version} / {health.db_status}
              </div>
            </div>
          )}

          <div style={{ marginTop: 'auto' }}>
            <div style={S.divider} />
            <div style={S.stackSection}>
              <div style={S.navSection}>Tech Stack</div>
              {STACK.map(s => (
                <div key={s} style={S.stackItem}>
                  <span style={S.stackBullet} />
                  {s}
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Main */}
        <main style={S.main}>
          {/* Top bar */}
          <div style={S.topBar}>
            <div style={S.topBarLeft}>
              <span style={S.breadcrumb}>UserLens</span>
              <span style={S.breadcrumbSep}>/</span>
              <span style={S.breadcrumbActive}>{TABS.find(t => t.id === tab)?.label}</span>
            </div>
            <div style={S.topBarRight}>
              <span style={S.topBarTag}>Adaptive RecSys</span>
            </div>
          </div>

          {/* Content */}
          <div key={tab} className="fade-in" style={S.content}>
            {tab === 'feed' && <RecommendationFeed users={users} />}
            {tab === 'eval' && <EvalDashboard />}
            {tab === 'bias' && <BiasAuditChart />}
            {tab === 'cold' && <ColdStartDemo />}
          </div>
        </main>
      </div>
    </>
  );
}

const S = {
  root: { display:'flex', minHeight:'100vh', position:'relative', overflow:'hidden' },
  gridBg: { position:'fixed', inset:0, pointerEvents:'none', zIndex:0 },

  sidebar: {
    width: 240,
    background: '#0d0d0d',
    borderRight: '1px solid #1a1a1a',
    padding: '32px 24px',
    display: 'flex',
    flexDirection: 'column',
    gap: 24,
    flexShrink: 0,
    position: 'relative',
    zIndex: 1,
  },

  logoArea: { display:'flex', alignItems:'center', gap:12 },
  logoMark: {
    width: 42, height: 42,
    background: '#c8a96e',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)',
    flexShrink: 0,
  },
  logoMarkText: { fontFamily:"'JetBrains Mono',monospace", fontWeight:500, fontSize:12, color:'#080808', letterSpacing:'0.05em' },
  logoName: { fontWeight:800, fontSize:15, letterSpacing:'0.12em', color:'#e8e0d4' },
  logoCredit: { fontSize:10, color:'#4a4540', letterSpacing:'0.04em', marginTop:2 },

  divider: { height:1, background:'linear-gradient(90deg, #1a1a1a, #2a2520, #1a1a1a)', flexShrink:0 },

  nav: { display:'flex', flexDirection:'column', gap:2 },
  navSection: { fontSize:9, color:'#3a3530', letterSpacing:'0.15em', textTransform:'uppercase', marginBottom:8, fontFamily:"'JetBrains Mono',monospace" },
  navBtn: {
    display: 'flex', alignItems: 'center', gap:10,
    padding: '9px 0 9px 16px',
    borderRadius: 0,
    border: 'none',
    borderLeft: '2px solid transparent',
    background: 'transparent',
    color: '#4a4540',
    cursor: 'pointer',
    textAlign: 'left',
    fontSize: 13,
    fontFamily: "'Syne', sans-serif",
    fontWeight: 600,
    letterSpacing: '0.02em',
    width: '100%',
  },
  navBtnActive: { color:'#c8a96e', borderLeft:'2px solid #c8a96e', paddingLeft:14 },
  navIndex: { fontFamily:"'JetBrains Mono',monospace", fontSize:10, color:'#2a2520', minWidth:20 },

  statusBox: { display:'flex', flexDirection:'column', gap:10 },
  statusRow: { display:'flex', alignItems:'center', gap:8 },
  statusDot: {
    width:6, height:6, borderRadius:'50%', background:'#4ade80', flexShrink:0,
    animation: 'pulse-dot 2s ease infinite',
  },
  statusLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#4ade80', letterSpacing:'0.12em' },
  statusGrid: { display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 },
  statusItem: { background:'#111', border:'1px solid #1a1a1a', borderRadius:4, padding:'8px 10px' },
  statusVal: { fontFamily:"'JetBrains Mono',monospace", fontSize:14, color:'#c8a96e', fontWeight:500 },
  statusKey: { fontSize:9, color:'#3a3530', letterSpacing:'0.1em', textTransform:'uppercase', marginTop:2 },
  modelTag: {
    display:'flex', alignItems:'center', gap:6,
    fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#3a3530', letterSpacing:'0.08em',
  },
  modelDot: { width:4, height:4, borderRadius:'50%', background:'#c8a96e', flexShrink:0 },

  stackSection: { display:'flex', flexDirection:'column', gap:6 },
  stackItem: { display:'flex', alignItems:'center', gap:8, fontSize:11, color:'#3a3530', letterSpacing:'0.02em' },
  stackBullet: { width:3, height:3, borderRadius:'50%', background:'#c8a96e', opacity:0.5, flexShrink:0 },

  main: { flex:1, display:'flex', flexDirection:'column', position:'relative', zIndex:1, overflow:'auto' },
  topBar: {
    display:'flex', alignItems:'center', justifyContent:'space-between',
    padding:'16px 40px',
    borderBottom:'1px solid #1a1a1a',
    background:'rgba(8,8,8,0.8)',
    backdropFilter:'blur(8px)',
    position:'sticky', top:0, zIndex:10,
    flexShrink:0,
  },
  topBarLeft: { display:'flex', alignItems:'center', gap:8 },
  breadcrumb: { fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:'#3a3530' },
  breadcrumbSep: { color:'#2a2520', fontSize:11 },
  breadcrumbActive: { fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:'#c8a96e' },
  topBarRight: {},
  topBarTag: {
    fontFamily:"'JetBrains Mono',monospace", fontSize:9, color:'#3a3530',
    border:'1px solid #1a1a1a', borderRadius:2, padding:'3px 8px', letterSpacing:'0.1em',
    textTransform:'uppercase',
  },

  content: { padding:'40px', flex:1 },
};
