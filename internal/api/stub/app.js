// THE BOOTSTRAP. The per-launch browser token rides `#t=…`. Three things happen in this
// order and the order is the point:
//   1. read it out of location.hash,
//   2. scrub it with history.replaceState so it survives in neither the address bar nor
//      the session history entry,
//   3. keep it in a closure and put it on an Authorization header — never in a cookie,
//      never in localStorage, never in a URL.
// The one-time-secret -> HttpOnly session-cookie exchange is cl-007's, behind its real-UI
// gate. Everything around it is already true here.
const token = (function () {
  const hash = new URLSearchParams((location.hash || '').replace(/^#/, ''));
  const t = hash.get('t') || '';
  if (t) history.replaceState(null, '', location.pathname);
  return t;
})();

const set = (id, text, bad) => {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = bad ? 'bad' : '';
};

set('cred', token ? 'held in memory (' + token.length + ' chars, scrubbed from the URL)'
                  : 'none — open this page with `cozy up --open`', !token);

const api = (path) => fetch(path, { headers: { Authorization: 'Bearer ' + token } });

async function boot() {
  try {
    const health = await (await fetch('/healthz')).json();
    set('svc', health.service);
  } catch (e) { set('svc', 'unreachable', true); return; }

  if (!token) { set('caps', 'a credential is required', true); return; }

  try {
    const caps = await (await api('/v1/capabilities')).json();
    set('caps', caps.core + ' · ' + caps.tokens.length + ' tokens');
  } catch (e) { set('caps', String(e), true); }

  try {
    const eps = await (await api('/v1/local/endpoints')).json();
    set('eps', eps.count ? eps.endpoints.map(e => e.endpoint + ' [' +
        e.functions.join(', ') + ']' + (e.resident ? ' · resident' : '')).join(' · ')
      : 'none installed');
  } catch (e) { set('eps', String(e), true); }

  try {
    const reqs = await (await api('/v1/requests?limit=5')).json();
    set('reqs', reqs.count ? reqs.requests.map(r => r.request_id.slice(0, 12) + ' ' +
        r.status).join(' · ') : 'none yet');
  } catch (e) { set('reqs', String(e), true); }

  // ONE multiplexed stream for every request — the browser caps concurrent connections
  // per origin, so a page that opened one per request would silently stop receiving.
  // EventSource cannot set a header, so this reads the stream with fetch instead: the
  // credential stays on the Authorization header and never becomes a query parameter.
  let seen = 0;
  const res = await api('/v1/events?from=now');
  set('ev', 'streaming…');
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const line = block.split('\n').find(l => l.startsWith('data: '));
      if (!line) continue;
      const event = JSON.parse(line.slice(6));
      seen += 1;
      set('ev', seen + ' event(s) · last ' + event.type + ' ' +
          event.request_id.slice(0, 12));
    }
  }
}
boot();
