"""Suite `tensorfs`'s verdict, from what cells.sh kept under <gate>/suite-tensorfs (analyze.py runs check(gate) when
the suite is selected). Times are held against the SKU's last passed suite (baseline['suites']['tensorfs']), like the
product cells: the worst known-good plus 25%, unless the known-good spread is wider. Rates are reported beside the
probe's disk ceiling; an executor never dials, so there is no network ceiling to read them against. A PASS appends
this run to the baseline.
    check.py --past <ndjson> <stage> <fraction>   bytes moved once <stage> passed the fraction (cells.sh's tfs_cut)"""
import json
import os
import sys
import time

SLACK = 0.25


def events(path):
    out = []
    for line in open(path) if os.path.exists(path) else []:
        line = line.strip()
        if line.startswith('{'):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def stage_progress(evs, stage):
    """(bytes done, bytes total) of the last progress in a stage naming <stage>."""
    last = (0, 0)
    for e in evs:
        p = e.get('progress') or {}
        if e.get('type') == 'run.progress' and stage in (p.get('stage') or ''):
            last = (p.get('bytes_done') or 0, p.get('bytes_total') or 0)
    return last


def stage_seconds(evs, stage):
    """From the log line that opened <stage> to the run's end."""
    start = next((e['at_ms'] for e in evs if e.get('type') == 'run.log' and stage in (e.get('text') or '')), None)
    end = next((e['run'].get('settled_at_ms') for e in reversed(evs) if isinstance(e.get('run'), dict)
                and e['run'].get('settled_at_ms')), None)
    return (end - start) / 1000 if start and end else None


def hub_bytes(rec):
    """What the run fetched from the Hub: every `bytes: {hub, volume}` its result carries."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            b = o.get('bytes')
            if isinstance(b, dict) and 'hub' in b:
                found.append(b)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk((rec or {}).get('result'))
    return (sum(b.get('hub') or 0 for b in found), sum(b.get('volume') or 0 for b in found)) if found else (None, None)


def tree(rec):
    return next((m.get('tree') for m in ((rec or {}).get('result') or {}).get('models', [])), None)


def check(gate):
    t = os.path.join(gate['out'], 'suite-tensorfs')
    if not os.path.isdir(t):
        return [], []
    fails, notes, rows = [], [], {}
    for line in open(os.path.join(t, 'steps.tsv')) if os.path.exists(os.path.join(t, 'steps.tsv')) else []:
        name, number, wall = (line.rstrip('\n').split('\t') + ['', ''])[:3]
        p = os.path.join(t, name + '.json')
        rows[name] = {'run': number, 'wall_s': int(wall) / 1000 if wall else None,
                      'rec': json.load(open(p)) if os.path.exists(p) else None, 'evs': events(os.path.join(t, name + '.ndjson'))}

    def done(name, cancelled=False):
        r = rows.get(name)
        status = ((r or {}).get('rec') or {}).get('status')
        if cancelled:
            return r is not None
        if status != 'completed':
            why = ((r or {}).get('rec') or {}).get('reason')
            fails.append(f'tensorfs {name}: run {(r or {}).get("run")} {status} {json.dumps(why)[:300] if why else ""}')
            return False
        return True

    base = gate['baseline'].setdefault('suites', {}).setdefault('tensorfs', {})
    now = {}

    def against(case, seconds, nbytes, what):
        if seconds is None:
            return
        now[case] = {'seconds': round(seconds, 2), 'bytes': nbytes}
        good = [g for g in base.get(case, []) if not nbytes or not g.get('bytes') or abs(g['bytes'] - nbytes) <= 0.05 * nbytes]
        if not good:
            notes.append(f'tensorfs {case}: {what} {seconds:.1f} s, no known-good yet: this run starts them')
            return
        vals = [g['seconds'] for g in good]
        limit = max(vals) + max(SLACK * max(vals), max(vals) - min(vals))
        (fails if seconds > limit else notes).append(
            f'tensorfs {case}: {what} {seconds:.1f} s vs known-good {min(vals):.1f}-{max(vals):.1f} s (limit {limit:.1f})')

    if done('probe'):
        d = rows['probe']['rec'].get('result') or {}
        notes.append(f'tensorfs probe: the machine\'s disk writes {(d.get("disk_write_bytes_per_s") or 0) / 1e6:.0f} MB/s, '
                     f'reads cold {(d.get("disk_read_bytes_per_s") or 0) / 1e6:.0f} MB/s')
    st = {}
    for line in open(os.path.join(t, 'straggler.tsv')) if os.path.exists(os.path.join(t, 'straggler.tsv')) else []:
        name, code, ms = line.rstrip('\n').split('\t')
        proxy = events(os.path.join(t, name + '.proxy'))
        objs = [r for r in proxy if r.get('kind') == 'object']
        st[name] = {'ok': code == '0', 'seconds': int(ms) / 1000, 'slowed': sum(1 for r in objs if r['slow']),
                    'reasked': sum(1 for r in objs if r['ask'] > 1), 'bytes': sum(r['bytes'] for r in objs),
                    'error': open(os.path.join(t, name + '.err')).read()[-300:] if code != '0' else ''}
    c, sl = st.get('straggler-clean'), st.get('straggler-slow')
    if not c or not sl:
        fails.append('tensorfs straggler: the case did not run')
    elif not c['ok'] or not sl['ok']:
        fails.append(f'tensorfs straggler: tfs download failed (clean: {c["error"]}; with stragglers: {sl["error"]})')
    elif not sl['slowed']:
        fails.append('tensorfs straggler: no object was slowed: the case tested nothing')
    else:
        limit = 1.5 * c['seconds'] + 20
        (fails if sl['seconds'] > limit else notes).append(
            f'tensorfs straggler: {sl["slowed"]} slow objects, {sl["reasked"]} re-asked; {sl["seconds"]:.1f} s vs clean '
            f'{c["seconds"]:.1f} s (limit {limit:.1f})')
        against('straggler', sl['seconds'], c['bytes'], 'download with stragglers')

    def rate(name, nbytes, seconds):
        if nbytes and seconds:
            notes.append(f'tensorfs {name}: {nbytes / seconds / 1e6:.0f} MB/s')

    if done('download-cold'):
        hub, vol = hub_bytes(rows['download-cold']['rec'])
        total = (hub or 0) + (vol or 0)
        # A datacenter's model cache serves a "cold" download as volume bytes: its times are its own baseline.
        src = 'hub' if hub and hub >= total / 2 else 'cache'
        if not total:
            fails.append('tensorfs download-cold: nothing was downloaded (the store held it already?)')
        rate('download-cold', total, rows['download-cold']['wall_s'])
        notes.append(f'tensorfs download-cold: {(hub or 0) / 2**30:.1f} GiB from the Hub, {(vol or 0) / 2**30:.1f} GiB from the volume')
        against(f'download-cold-{src}', rows['download-cold']['wall_s'], total, f'{total / 2**30:.1f} GiB ({src})')
    if done('download-again'):
        hub, _ = hub_bytes(rows['download-again']['rec'])
        if hub:
            fails.append(f'tensorfs download-again: {hub / 2**20:.0f} MiB came from the Hub again: the store did not hold it')
    if done('download-cut', cancelled=True) and done('download-resume'):
        cut = os.path.join(t, 'download-cut.cut')
        moved = int(open(cut).read()) if os.path.exists(cut) else 0
        hub, vol = hub_bytes(rows['download-resume']['rec'])
        total = (hub or 0) + (vol or 0)
        if not moved:
            fails.append('tensorfs download-resume: the first download was never cut part-way: nothing was resumed')
        elif hub is None or hub > max(0, total - 0.8 * min(moved, total)) + 2**20:
            fails.append(f'tensorfs download-resume: {(hub or 0) / 2**20:.0f} MiB fetched from the Hub of {total / 2**20:.0f} MiB '
                         f'after {moved / 2**20:.0f} MiB had landed: it started over')
        else:
            notes.append(f'tensorfs download-resume: {hub / 2**20:.0f} MiB from the Hub, {(vol or 0) / 2**20:.0f} MiB from the volume, '
                         f'after {moved / 2**20:.0f} MiB had landed' + ('' if hub else ' (the cache served it: the Hub path was not exercised)'))
    up = None
    if done('upload'):
        evs = rows['upload']['evs']
        sec, (_, total) = stage_seconds(evs, 'uploading to'), stage_progress(evs, 'uploading to')
        up = (total / sec) if sec and total else None
        rate('upload', total, sec)
        against('upload', sec, total, f'{total / 2**20:.0f} MiB sent')
    if done('upload-again'):
        wall, first = rows['upload-again']['wall_s'], (rows.get('upload') or {}).get('wall_s') or 0
        if tree(rows['upload-again']['rec']) != tree((rows.get('upload') or {}).get('rec')):
            fails.append('tensorfs upload-again: the same folder published another tree')
        if wall is None or wall > max(15, 0.1 * first):
            fails.append(f'tensorfs upload-again: {wall} s to send what the Hub holds (the first took {first:.0f} s): no dedup')
        else:
            notes.append(f'tensorfs upload-again: {wall:.1f} s for content the Hub held (the first took {first:.0f} s)')
    if done('upload-cut', cancelled=True) and done('upload-resume'):
        cut = os.path.join(t, 'upload-cut.cut')
        moved = int(open(cut).read()) if os.path.exists(cut) else 0
        evs = rows['upload-resume']['evs']
        sec, (_, total) = stage_seconds(evs, 'uploading to'), stage_progress(evs, 'uploading to')
        if not moved or not sec or not total or not up:
            fails.append(f'tensorfs upload-resume: cut at {moved} bytes, resumed in {sec} s: nothing to judge')
        else:
            limit = 1.5 * (total - 0.8 * moved) / up + 15
            (fails if sec > limit else notes).append(
                f'tensorfs upload-resume: {sec:.1f} s for the rest after {moved / 2**20:.0f} of {total / 2**20:.0f} MiB '
                f'(limit {limit:.1f} s at the upload\'s own rate)')
    if not fails:
        label = open(os.path.join(gate['out'], 'label')).read().strip()
        at = time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())
        for case, v in now.items():
            base.setdefault(case, []).append({**v, 'label': label, 'at': at})
    return fails, notes


if __name__ == '__main__' and sys.argv[1:2] == ['--past']:
    done_, total = stage_progress(events(sys.argv[2]), sys.argv[3])
    if total and done_ >= float(sys.argv[4]) * total:
        print(done_)
