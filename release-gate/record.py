"""A run's memory record read the same for 0.30 and 0.32 (the published baselines are 0.30.x). Each function takes one
`measurements.memory` and tolerates absent or other-shaped members.
  0.32: `stats.varena[]`, one row per GPU: {gpu, ceiling, stats: varena's Stats, torch_peak}; Stats has `moved` (kept,
        leased, store, derived as {bytes, ns}), `peak` (held), `working` (needed at once), `evicted`, `leftover`,
        `outlived`, `host`.
  0.30: `stats.copies`, `rows`, `paged_in`, `evictions`, `leftover`, and `peak_reserved_bytes` (torch's pool, weights
        included)."""


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _stats(memory):
    s = (memory or {}).get('stats')
    return s if isinstance(s, dict) else {}


def rows(memory):
    """0.32's per-GPU varena Stats (none on 0.30)."""
    v = _stats(memory).get('varena')
    return [r['stats'] for r in v if isinstance(r, dict) and isinstance(r.get('stats'), dict)] if isinstance(v, list) else []


def held(memory):
    """The device bytes the run held at its fullest: varena's peak (pages and torch), else 0.30's reserved peak."""
    return max((_num(s.get('peak')) for s in rows(memory)), default=0) or _num((memory or {}).get('peak_reserved_bytes'))


def working(memory):
    """The most it needed at once: varena's `working`; 0.30's reserved peak (it paged per step, so it is that); 0 for a
    0.32 record from before `working` (its reserved figure is torch's alone, or the held peak: neither compares)."""
    if rows(memory):
        return max((_num(s.get('working')) for s in rows(memory)), default=0)
    return _num((memory or {}).get('peak_reserved_bytes'))


#: The ceiling a copy kind is held against.
CEILING = {'store': 'disk_gbps', 'derived': 'disk_gbps', 'load': 'disk_gbps', 'refill': 'h2d_gbps'}


def copies(memory):
    """Bytes and seconds moved by kind (CEILING): 0.32's `store` and `derived` fills, 0.30's `load` (disk) and
    `refill` (host to device). Kept and leased pages are mapped, not copied."""
    out = {}
    for s in rows(memory):
        for kind in ('store', 'derived'):
            f = (s.get('moved') or {}).get(kind) or {}
            b, t = out.get(kind, (0, 0.0))
            out[kind] = (b + _num(f.get('bytes')), t + _num(f.get('ns')) / 1e9)
    for c in _stats(memory).get('copies') or []:
        if isinstance(c, dict) and c.get('kind') in ('disk', 'h2d'):
            kind = 'load' if c['kind'] == 'disk' else 'refill'
            b, t = out.get(kind, (0, 0.0))
            out[kind] = (b + _num(c.get('bytes')), t + _num(c.get('seconds')))
    return {k: v for k, v in out.items() if v[0]}


def paged(memory):
    """Bytes paged in or spilled: 0.32's store and derived fills and make_room's evictions; 0.30's paged_in, streamed
    and evicted bytes."""
    st = _stats(memory)
    new = sum(_num(((s.get('moved') or {}).get(k) or {}).get('bytes')) for s in rows(memory) for k in ('store', 'derived'))
    new += sum(_num(e) for s in rows(memory) for e in s.get('evicted') or [])
    old = _num(st.get('paged_in')) + sum(_num(r.get('streamed')) for r in st.get('rows') or [] if isinstance(r, dict))
    old += sum(_num(e.get('nbytes')) for e in st.get('evictions') or [] if isinstance(e, dict))
    return new + old


def step_held(memory):
    """The most one step held: varena's peak, else 0.30's resident weights plus torch's allocation per row."""
    old = max((_num(r.get('resident')) + _num(r.get('allocated')) for r in _stats(memory).get('rows') or [] if isinstance(r, dict)), default=0)
    return max((_num(s.get('peak')) for s in rows(memory)), default=0) or old


def leftover(memory):
    """Bytes still held at the request's end: varena's pins and outlived torch blocks, else 0.30's leftover."""
    new = sum(_num(s.get('leftover')) + sum(_num(o.get('bytes')) for o in s.get('outlived') or [] if isinstance(o, dict)) for s in rows(memory))
    return new or _num(_stats(memory).get('leftover'))


def host(memory):
    """The process's anonymous host high-water (0.32), None where not recorded."""
    return max((_num(s.get('host')) for s in rows(memory)), default=0) or None
