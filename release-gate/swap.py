"""Two checkpoints of one release that share a component name, loaded in turn on one executor with no restart between
them: Qwen-Image's text and edit transformers, then its two Lightning LoRAs (TFS 0.15.0 named varena's weight sets
`{release}/{component}`, so the second load was refused: `set … is loaded with another layout or release`, run 15744).
Each cell must complete (a failed one already fails the gate), and every cell of a group must have run on the executor
its first cell started, or the check proves nothing.
    check(gate) -> (fails, notes)"""
GROUPS = (('qwen-image', 'qwen-image-edit', 'qwen-image-t2i'), ('qwen-image-turbo', 'qwen-image-turbo-edit'))


def started(rec):
    return (((rec.get('measurements') or {}).get('execution') or {}).get('executor') or {}).get('started_unix_ms')


def check(gate):
    fails, notes = [], []
    for group in GROUPS:
        recs = {n: gate['records'][n] for n in group if n in gate['records']}
        if len(recs) < len(group):
            continue  # not selected, or a cell failed, which fails on its own
        runs = ', '.join(f'{n} {r["number"]}' for n, r in recs.items())
        executors = {started(r) for r in recs.values()}
        if len(executors) == 1 and None not in executors:
            notes.append(f'swap: {runs} loaded in turn on one executor')
        else:
            fails.append(f'swap: {runs} ran on {len(executors)} executors, not one: the swap was not tested')
    return fails, notes
