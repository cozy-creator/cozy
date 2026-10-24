"""Verdict for one release-gate directory. Each cell ran once on a fresh pod; its execution time, denoise step 1,
steady s/step and peak reserved VRAM are held against the SKU's last known-good numbers (the baseline file,
gate-<SKU>.json, which gate.sh fetches from and publishes to the `release-gate-baselines` release). A PASS is recorded
in the baseline file; `--accept` records a FAIL the lead accepted. Exit 1 on any failure: the publish is blocked.
    analyze.py <gate dir> <SKU> <baseline file> [--accept]
More checks (quality, lm, swap, a suite's check.py) are modules imported here and listed in CHECKS: check(gate) -> (fails, notes)."""
import fnmatch, glob, json, os, re, subprocess, sys, time
import numpy as np
from PIL import Image

out, sku, BASE = sys.argv[1].rstrip('/'), sys.argv[2], sys.argv[3]
accept = '--accept' in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'suites'))
import lm, quality, swap  # noqa: E402
from varena import check as varena  # noqa: E402
# A suite's check runs when gate.sh ran the suite (--suite: the names in <gate dir>/suites).
SUITES = open(os.path.join(out, 'suites')).read().split() if os.path.exists(os.path.join(out, 'suites')) else []
SUITE_CHECKS = {'varena': varena.check}  # runtime and packages: quality, lm and swap, which every gate runs
CHECKS = [quality.check, lm.check, swap.check] + [SUITE_CHECKS[s] for s in SUITES if s in SUITE_CHECKS]
# Allowed rise over the worst known-good number, unless the good numbers' own spread is wider.
SLACK = {'execution_ms': 0.25, 'step1_ms': 0.5, 'steady_ms': 0.10, 'peak_vram_mb': 0.10}
# A step may cost at most 2x its reference plus 5 s: step 1 against step 2 (known-good fresh H3 executors 0.9-1.8x; the
# 10-22 split regression 8-17x), every later step against the median of steps 2..n (a length change, such as a video
# reference's rows leaving the sparse window, must not stall a step). The seconds absorb sub-second image steps.
STEP_RATIO, STEP_EXTRA_MS = 2, 5_000
# Request-path blocking work the Runtime still does (tracker cr-217), as fnmatch patterns. The list may only shrink.
KNOWN_BLOCKING = os.path.join(HERE, 'known-blocking.json')
BLOCKING_SLACK = 0.5
# Cells that fail on a known open bug and are not run until it is fixed (lead-ruled), listed in every verdict.
KNOWN_FAILING = os.path.join(HERE, 'known-failing.json')
fails, notes = [], []


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def measurements(rec):
    return [rec.get('measurements') or {}] + [c.get('measurements') or {} for c in rec.get('calls') or []]


def observations(rec):
    return [o for m in measurements(rec) for o in m.get('observations') or []]


def peak_vram_mb(rec):
    """The largest peak reserved device memory of the run and its calls, in MiB."""
    peaks = [(m.get('memory') or {}).get('peak_reserved_bytes') or 0 for m in measurements(rec)]
    return round(max(peaks) / 2**20) if any(peaks) else None


def canonical(target):
    """The package function a target names, whether published (org/pkg/fn@v) or a checkout (./pkg/fn)."""
    return re.sub(r'^(\./|[^/]+/(?=[^/]+/[^/]+$))', '', re.sub(r'@.*$', '', target))


def steps(rec):
    """The dominant step series (largest total) of the run and its calls, in ms."""
    best = None
    for m in measurements(rec):
        for v in ((m.get('attribution') or {}).get('steps') or {}).values():
            if v.get('count') and (best is None or v.get('total_ms', 0) > best.get('total_ms', 0)):
                best = v
    return [x[1] for x in (best or {}).get('series') or []]


cells = [l.rstrip('\n').split('\t') for l in open(os.path.join(out, 'cells.tsv')) if l.strip()]
summary, recs, host_legs = {}, {}, {}
for name, target, number in cells:
    p = os.path.join(out, name + '.json')
    rec = json.load(open(p)) if os.path.exists(p) else None
    if rec is None or rec.get('status') != 'completed':
        why = (rec or {}).get('reason')
        why = f"{why.get('code')}: {why.get('message')}" if isinstance(why, dict) else why
        fails.append(f'{name}: run {number} {rec and rec.get("status")} {why}')
        continue
    recs[name] = rec
    obs = observations(rec)
    gemm = [list(x) for x in sorted({(o['fields'].get('format', ''), o['value']) for o in obs if o.get('name') == 'gemm.selected'})]
    attn = sorted({o['value'] for o in obs if o.get('name') == 'attention.selected'})
    for o in obs:
        text = f"{o.get('name')} {o.get('value')}"
        if any(s in text for s in ('gemm.fallback', 'decoded_float')):
            fails.append(f'{name}: {text} {o.get("fields")}')
    for fmt, val in gemm:
        if fmt.startswith('int8') and val != 'cozy-int8':
            fails.append(f'{name}: int8 GEMM served by {val}, not cozy-int8')
        if any(s in val for s in ('_int_mm', 'dense', 'decoded_float', 'fallback')):
            fails.append(f'{name}: GEMM {fmt} served by {val}')
    blocking = {}
    for o in obs:
        if o.get('name') == 'request_path.blocking':
            blocking[o['value']] = blocking.get(o['value'], 0) + round(float(o['fields'].get('ms', 0)))
    s = steps(rec)
    legs = {}
    for m in measurements(rec):
        ex = m.get('execution') or {}
        for part in ('executor', 'construction'):
            for leg in ('cuda_init', 'measure_device'):
                v = ((ex.get(part) or {}).get('legs_ms') or {}).get(leg)
                if v: legs[leg] = max(legs.get(leg, 0), round(v))
    host_legs[name] = legs
    c = summary[name] = {'blocking': blocking, 'run': rec['number'], 'target': canonical(target), 'gemm': gemm, 'attention': attn,
                         'execution_ms': rec.get('execution_ms'), 'step1_ms': round(s[0]) if s else None,
                         'peak_vram_mb': peak_vram_mb(rec),
                         'steady_ms': round(sum(s[1:]) / len(s[1:])) if len(s) > 1 else None,
                         'files': sorted(glob.glob(os.path.expanduser(f"~/.cozy/outputs/*/{rec['number']}-*")))}
    if len(s) > 2:
        median = sorted(s[1:])[(len(s) - 1) // 2]
        for i, (ms, ref) in enumerate([(s[0], s[1])] + [(x, median) for x in s[1:]]):
            if ms > STEP_RATIO * ref + STEP_EXTRA_MS:
                fails.append(f'{name}: step {i + 1} {ms / 1000:.1f} s vs {"step 2" if i == 0 else "median"} '
                             f'{ref / 1000:.1f} s (at most {STEP_RATIO}x + {STEP_EXTRA_MS // 1000} s)')


def image(name):
    c = summary.get(name)
    if not c: return None
    p = [f for f in c['files'] if f.endswith(('.png', '.webp', '.jpg'))]
    if not p:
        fails.append(f'{name}: no image output'); return None
    im = Image.open(p[0]); a = np.asarray(im.convert('RGB')).astype(float)
    if a.std() < 10: fails.append(f'{name}: flat image (std {a.std():.1f})')
    return im


for n in ('sdxl', 'klein', 'anima', 'sdxl-p2'):
    image(n)
# Candidate cells: any other image output must not be flat.
for n, c in summary.items():
    if n not in ('sdxl', 'klein', 'anima', 'sdxl-p2', 'qwen', 'birefnet', 'describer', 'h3', 'seedvr2') and any(f.endswith(('.png', '.webp', '.jpg')) for f in c['files']):
        image(n)
im = image('qwen')
if im is not None:
    if im.mode != 'RGBA':
        fails.append(f'qwen: {im.mode}, not RGBA')
    else:
        z = (np.asarray(im)[..., 3] == 0).mean(); notes.append(f'qwen: alpha 0 {z:.0%}')
        if z < 0.05: fails.append(f'qwen: no real transparency (alpha 0 {z:.1%})')
im = image('birefnet')  # composites the cutout onto white
if im is not None:
    white = (np.asarray(im.convert('L')) > 245).mean(); notes.append(f'birefnet: white background {white:.0%}')
    if white < 0.05: fails.append(f'birefnet: no white background ({white:.1%})')
if 'describer' in recs:
    texts = [s for d in walk(recs['describer'].get('result')) for v in d.values() for s in (v if isinstance(v, list) else [v])
             if isinstance(s, str) and len(s) > 20]
    if not texts: fails.append('describer: no description text')
for n, min_h, min_frames in (('h3', 700, 300), ('seedvr2', 1000, 20)):
    c = summary.get(n)
    vids = [f for f in (c or {}).get('files', []) if f.endswith('.mp4')]
    if c and not vids:
        fails.append(f'{n}: no video output'); continue
    if vids:
        p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                            'stream=height,nb_read_packets', '-of', 'csv=p=0', vids[-1]], capture_output=True, text=True)
        h, fr = (int(x) for x in p.stdout.strip().split(',')[:2])
        notes.append(f'{n}: {h}p, {fr} frames')
        if h < min_h or fr < min_frames: fails.append(f'{n}: video {h}p, {fr} frames')
a, b = summary.get('sdxl'), summary.get('sdxl-p2')
if a and b:
    if not b['gemm'] and not b['attention']:
        fails.append('sdxl-p2: no kernel observations (memoized or not executed)')
    elif (a['gemm'], a['attention']) != (b['gemm'], b['attention']):
        fails.append(f'sdxl-p2: second executor serves {b["gemm"]} / {b["attention"]}, first {a["gemm"]} / {a["attention"]}')

# Against the last known-good numbers.
base = json.load(open(BASE)) if os.path.exists(BASE) else {'sku': sku, 'cells': {}}
known = json.load(open(KNOWN_BLOCKING)) if os.path.exists(KNOWN_BLOCKING) else []
for name, c in summary.items():
    good = [g for g in base['cells'].get(name, {}).get('good', []) if canonical(g['target']) == c['target']]
    before = good[-1].get('blocking', {}) if good else {}
    for work, ms in sorted(c['blocking'].items()):
        site = next((k for k in known if fnmatch.fnmatch(work, k['pattern'])), None)
        if site is None:
            fails.append(f'{name}: NEW request-path blocking: {work} {ms / 1000:.1f} s')
        elif work in before and ms > before[work] * (1 + BLOCKING_SLACK) + 1000:
            fails.append(f'{name}: known blocking {work} grew {before[work] / 1000:.1f} -> {ms / 1000:.1f} s ({site["issue"]})')
        else:
            notes.append(f'{name}: known blocking {work} {ms / 1000:.1f} s ({site["issue"]}, {site["site"]})')
for name, c in summary.items():
    good = [g for g in base['cells'].get(name, {}).get('good', []) if canonical(g['target']) == c['target']]
    if not good:
        notes.append(f'{name}: no known-good numbers for {c["target"]} on {sku}: this run starts them'); continue
    last = good[-1]
    if (last['gemm'], last['attention']) != (c['gemm'], c['attention']):
        fails.append(f'{name}: kernels {last["gemm"]} / {last["attention"]} -> {c["gemm"]} / {c["attention"]}')
    for k, slack in SLACK.items():
        vals = [g[k] for g in good if g.get(k)]
        if not vals or not c.get(k):
            continue
        limit = max(vals) + max(slack * max(vals), max(vals) - min(vals))
        unit, div = ('GiB', 1024) if k == 'peak_vram_mb' else ('s', 1000)
        line = (f'{name} {k.rsplit("_", 1)[0]}: {c[k] / div:.1f} {unit} vs known-good {min(vals) / div:.1f}-{max(vals) / div:.1f} {unit}'
                f' (limit {limit / div:.1f})')
        (fails if c[k] > limit else notes).append(line)

for k in (json.load(open(KNOWN_FAILING)) if os.path.exists(KNOWN_FAILING) else []):
    if k['sku'] == sku:
        notes.append(f"known failing, not run: {k['cell']} ({k['issue']}: {k['what']})")
# The host: a slow driver makes every first pass slow and must read as the host, not the code (lead 10-23).
for leg in ('cuda_init', 'measure_device'):
    vals = [l[leg] for l in host_legs.values() if leg in l]
    if vals:
        notes.append(f'host {leg}: median {sorted(vals)[len(vals) // 2] / 1000:.1f} s, max {max(vals) / 1000:.1f} s over {len(vals)} executors')
gate = {'out': out, 'sku': sku, 'summary': summary, 'records': recs, 'baseline': base}
# CLI-behaviour checks a cells/*.sh recorded with gate.sh's `expect` (expect.tsv: check, PASS|FAIL, what happened).
EXPECT = os.path.join(out, 'expect.tsv')
for row in (l.rstrip('\n').split('\t') + [''] for l in open(EXPECT)) if os.path.exists(EXPECT) else []:
    (notes if row[1] == 'PASS' else fails).append(f'{row[0]}: {row[1]} {row[2]}'.rstrip())
for check in CHECKS:
    f, n = check(gate)
    fails += f; notes += n
for n in notes: print('note', n)
for f in fails: print('FAIL', f)
print('SUITES', ','.join(SUITES) or 'none')
print('VERDICT', 'FAIL' if fails else 'PASS', f'({len(cells)} cells)', '' if not fails or not accept else '(accepted by the lead)')
if not fails or accept:
    label = open(os.path.join(out, 'label')).read().strip()
    for name, c in summary.items():
        entry = {k: c[k] for k in ('run', 'target', 'gemm', 'attention', 'execution_ms', 'step1_ms', 'steady_ms', 'peak_vram_mb', 'blocking')}
        entry.update(label=label, at=time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()))
        base['cells'].setdefault(name, {'good': []})['good'].append(entry)
    json.dump(base, open(BASE, 'w'), indent=1)
    print('recorded', BASE)
sys.exit(1 if fails else 0)
