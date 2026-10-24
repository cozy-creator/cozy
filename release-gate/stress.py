"""Verdict on the memory stress cells (cells/stress.sh), one of analyze.py's CHECKS; analyze.py already holds every
cell, these included, to completion, kernels, times and peak VRAM against the baseline. A cell
`s<N>-<model>[-<i>]` ran under an N GiB cap, and must:
  - run on a machine whose ledger took the cap (its memory log after the cap write);
  - never hold more than the cap: at every step its resident weights plus torch's allocation, and torch's reserved
    peak, fit under it;
  - serve the kernels its uncapped product cell of the same shape served (another kernel under pressure is a slow
    fallback); H3's capped cell is 5 s against the product's 15 s, so analyze.py's baseline holds its kernels;
  - leave no bytes behind; the alternating SDXL/Anima runs must not slow down as they evict each other;
  - for H3, produce the whole 5 s video.
On a CPU rental (cells/stress-host.sh) `host-klein` and `host-qwen` ran under the container's cgroup, which `host-cap`
measured; the cap must sit below the host's RAM, and each run must record the bytes it paged or spilled under it.
    stress.py <gate dir>     the same verdict alone"""
import json, os, re, subprocess, sys

GIB = 1 << 30
# The third alternating run may cost this much more than the second (both refill after an eviction), plus 5 s.
CHURN = 0.25
CELL = re.compile(r's(\d+)-([a-z0-9]+)(?:-(\d+))?')
SAME_SHAPE = ('sdxl', 'anima', 'klein')


def kernels(rec):
    obs = [o for m in [rec.get('measurements') or {}] + [c.get('measurements') or {} for c in rec.get('calls') or []]
           for o in m.get('observations') or []]
    return (sorted({(o['fields'].get('format', ''), o['value']) for o in obs if o.get('name') == 'gemm.selected'}),
            sorted({o['value'] for o in obs if o.get('name') == 'attention.selected'}))


def memories(rec):
    """The run's and its calls' memory records (H3 keeps its rows in its calls)."""
    return [m for m in [((r.get('measurements') or {}).get('memory') or {}) for r in [rec] + (rec.get('calls') or [])] if m]


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)
    else:
        yield o


def host(recs):
    fails, notes = [], []
    cells = [n for n in ('host-klein', 'host-qwen') if n in recs]
    if not cells and 'host-cap' not in recs:
        return fails, notes
    cap = next((d for v in walk((recs.get('host-cap') or {}).get('result')) if isinstance(v, str) and '"cap"' in v
                for d in [json.loads(v)]), None)
    if not cap or not cap.get('cap') or not cap.get('mem_total') or cap['cap'] >= cap['mem_total']:
        return [f'host-cap: no host cap below the host RAM ({cap}): the host cells prove nothing here'], notes
    notes.append(f'host cap {cap["cap"] / GIB:.1f} GiB of {cap["mem_total"] / GIB:.1f} GiB RAM, swap {(cap.get("swap_total") or 0) / GIB:.1f} GiB')
    for name in cells:
        mems = memories(recs[name])
        stats = [m.get('stats') or {} for m in mems]
        moved = sum((st.get('paged_in') or 0) + sum(r.get('streamed', 0) for r in st.get('rows') or [])
                    + sum(e.get('nbytes', 0) for e in st.get('evictions') or []) for st in stats)
        anon = max((st['host'] for st in stats if st.get('host')), default=None)
        line = f'{name}: paged/spilled {moved / GIB:.1f} GiB' + (f', anonymous host high-water {anon / GIB:.1f} GiB' if anon else '')
        if not moved:
            fails.append(f'{name}: no page or spill bytes recorded under the {cap["cap"] / GIB:.1f} GiB host cap')
        elif anon and anon > cap['cap']:
            fails.append(line + f' over the {cap["cap"] / GIB:.1f} GiB cap')
        else:
            notes.append(line)
    return fails, notes


def check(gate):
    out, recs = gate['out'], gate['records']
    fails, notes = host(recs)
    capped = [(n, int(m.group(1)), m.group(2), m.group(3)) for n in recs for m in [CELL.fullmatch(n)] if m]
    for gib in sorted({c[1] for c in capped}, reverse=True):
        log = os.path.join(out, f'cap-{gib}GiB.memory.log')
        caps = [int(x) for x in re.findall(r'"capacity":(\d+)[^\n]*"event":"ledger"', open(log).read())] if os.path.exists(log) else []
        if not caps:
            fails.append(f'cap {gib} GiB: no ledger line in the memory log after the cap write')
        elif caps[-1] > gib * GIB:
            fails.append(f'cap {gib} GiB: the ledger took {caps[-1] / GIB:.2f} GiB')
        else:
            notes.append(f'cap {gib} GiB: ledger capacity {caps[-1] / GIB:.2f} GiB')
    execs = {}
    for name, gib, model, i in capped:
        rec = recs[name]
        mems = memories(rec)
        rows = [r for m in mems for r in (m.get('stats') or {}).get('rows') or []]
        held = max((r.get('resident', 0) + r.get('allocated', 0) for r in rows), default=0)
        reserved = max((m.get('peak_reserved_bytes') or 0 for m in mems), default=0)
        resident = {}
        for r in rows:
            resident[r['model']] = max(resident.get(r['model'], 0), r.get('resident', 0))
        moved = sum(r.get('streamed', 0) + r.get('paged_in', 0) for r in rows)
        line = (f'{name}: held {held / GIB:.2f} GiB at its fullest step, torch reserved {reserved / GIB:.2f} GiB, '
                f'every weight resident + reserved {(sum(resident.values()) + reserved) / GIB:.2f} GiB, '
                f'streamed + paged {moved / GIB:.1f} GiB, cap {gib} GiB')
        (fails if held > gib * GIB or reserved > gib * GIB else notes).append(line)
        left = sum((m.get('stats') or {}).get('leftover') or 0 for m in mems)
        if left:
            fails.append(f'{name}: left {left / GIB:.2f} GiB behind')
        ref = recs.get(model) if model in SAME_SHAPE else None
        if ref and kernels(ref) != kernels(rec):
            fails.append(f'{name}: kernels {kernels(rec)} under {gib} GiB; uncapped {model} served {kernels(ref)}')
        execs[name] = rec.get('execution_ms')
        if model == 'h3':
            vids = [o.get('path') for o in rec.get('output') or [] if (o.get('path') or '').endswith('.mp4')]
            if not vids:
                fails.append(f'{name}: no video output'); continue
            p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                                'stream=height,nb_read_packets', '-of', 'csv=p=0', vids[-1]], capture_output=True, text=True)
            h, fr = (int(x) for x in p.stdout.strip().split(',')[:2])
            (fails if h < 700 or fr < 120 else notes).append(f'{name}: video {h}p, {fr} frames')
    for model in sorted({m for _, _, m, i in capped if i}):
        runs = [execs[n] for n, _, m, i in sorted(capped, key=lambda c: int(c[3] or 0)) if m == model and i and execs.get(n)]
        if len(runs) >= 3:
            line = f'{model} alternation: ' + ' / '.join(f'{x / 1000:.1f}' for x in runs) + ' s'
            (fails if runs[2] > runs[1] * (1 + CHURN) + 5000 else notes).append(line)
    return fails, notes


if __name__ == '__main__':
    out = sys.argv[1].rstrip('/')
    recs = {}
    for line in open(os.path.join(out, 'cells.tsv')):
        name = line.split('\t')[0]
        p = os.path.join(out, name + '.json')
        rec = json.load(open(p)) if os.path.exists(p) else None
        if rec and rec.get('status') == 'completed':
            recs[name] = rec
    fails, notes = check({'out': out, 'records': recs})
    for n in notes: print('note', n)
    for f in fails: print('FAIL', f)
    print('STRESS', 'FAIL' if fails else 'PASS')
    sys.exit(1 if fails else 0)
