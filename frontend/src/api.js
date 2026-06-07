const BASE = process.env.REACT_APP_API_URL || '';

async function req(path, opts = {}) {
  const r = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export const api = {
  health:       ()        => req('/health'),
  listUsers:    (limit=20)=> req(`/users?limit=${limit}`),
  userHistory:  (uid)     => req(`/users/${uid}/history`),
  recommend:    (body)    => req('/recommend', { method:'POST', body: JSON.stringify(body) }),
  feedback:     (body)    => req('/feedback',  { method:'POST', body: JSON.stringify(body) }),
  evalLatest:   ()        => req('/eval/latest'),
  evalRun:      (n=100)   => req(`/eval/run?n_users=${n}`, { method:'POST' }),
  biasAudit:    ()        => req('/eval/bias'),
  biasRun:      (n=100)   => req(`/eval/bias/run?n_users=${n}`, { method:'POST' }),
  continual:    ()        => req('/continual/status'),
  continualTrigger: ()   => req('/continual/trigger', { method:'POST' }),
};
