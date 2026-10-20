"""Existing public Tree/physical binding rules, copied unchanged from consumer validator1a221."""
import hashlib
import json
import math
from pathlib import Path
import re


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uuid(value):
    value = str(value).lower().removeprefix('gpu-').replace('-', '')
    require(re.fullmatch('[0-9a-f]{32}', value) is not None, 'missing/invalid physical GPU UUID')
    return value


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def tree_document(root):
    """TensorD trees::document/device_executor::tree_binding, not an invented wire format.

    entries sorted by path; serde_json's sorted keys, UTF-8, compact separators, no newline.
    Include files.json too: the PUBLIC run digest must bind every returned receipt byte.
    """
    entries = [{'blob': {'length': p.stat().st_size, 'sha256': sha(p)}, 'kind': 'file',
                'path': p.relative_to(root).as_posix()}
               for p in sorted(Path(root).rglob('*')) if p.is_file()]
    return json.dumps({'entries': entries}, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode()


def public_tree(root, run, identity):
    require(isinstance(run.get('id'), str) and run['id'].startswith('run-') and
            type(run.get('number')) is int and run['number'] > 0, 'public run identity absent')
    # Context.request_id is TensorD's local Execution.id, not Calcifer's public run ID/number.
    require(isinstance(identity.get('request_id'), str) and identity['request_id'].isdecimal() and
            int(identity['request_id']) > 0, 'missing worker Context request identity')
    trees = [o for o in run.get('output', []) if o.get('media_type') == 'application/vnd.cozy.tree-manifest']
    require(len(trees) == 1, 'public run must contain this one receipt Tree')
    item = trees[0]
    require(item.get('index') == 0 and item.get('status') == 'completed' and not item.get('export_error') and
            isinstance(item.get('name'), str) and bool(item['name']) and
            item.get('id') == f"{run['number']}/{item['name']}", 'public Tree output/run association differs')
    document = tree_document(root)
    require(type(item.get('length')) is int and item['length'] == len(document) and
            item.get('sha256') == 'sha256:' + hashlib.sha256(document).hexdigest(),
            'downloaded receipts differ from this public run Tree digest/length')
    return {'run_id': run['id'], 'number': run['number'], 'output_id': item['id'],
            'sha256': item['sha256'], 'length': len(document), 'context_request_id': identity['request_id']}


def physical(run, machine, identities, launch, payload_sha):
    machine = machine.get('machine', machine)
    require(run.get('status') == 'completed', 'public run did not complete')
    require(run.get('machine') in {machine.get('name'), machine.get('alias')}, 'public run/machine identity differs')
    require(launch.get('payload_sha256') == payload_sha, 'public launch payload differs')
    argv = launch.get('argv', [])
    require(isinstance(argv, list) and all(isinstance(a, str) for a in argv) and len(argv) > 2 and argv[1] == 'run', 'missing public cozy launch command')
    def flag(name):
        require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), f'missing/duplicate launch {name}')
        return argv[argv.index(name) + 1]
    require(flag('--gpus') == '1', 'public launch did not request exactly one GPU')
    require(flag('--on') in {machine.get('name'), machine.get('alias')}, 'public launch allocation/machine differs')
    executions = []
    def visit(record):
        execution = record.get('measurements', {}).get('execution', {})
        if execution:
            executions.append(execution)
        for call in record.get('calls', []):
            visit(call)
    visit(run)
    rows = [g for e in executions for g in e.get('gpus', []) if type(g.get('gpu')) is int and g['gpu'] >= 0]
    pairs = {(r['gpu'], uuid(r.get('uuid'))) for r in rows}
    require(len(pairs) == 1, 'public run lacks exactly one positive GPU execution identity')
    observed = {(i['physical']['gpu'], uuid(i['gpu_uuid'])) for i in identities}
    require(len(observed) == 1, 'children executed different physical GPU allocations')
    index, gpu_uuid = next(iter(observed))
    require(type(index) is int and index >= 0, 'child has no physical GPU index')
    require(pairs == observed and all(type(e.get('degree')) is int and e['degree'] == 1 for e in executions if e.get('gpus') and
            any(type(g.get('gpu')) is int and g['gpu'] >= 0 for g in e['gpus'])), 'public physical GPU measurement differs')
    require(all(type(r.get('pid')) is int and r['pid'] > 0 and finite(r.get('start_us'), True) and
                finite(r.get('end_us'), True) and r['end_us'] > r['start_us'] and
                re.fullmatch('sm_[0-9]+', r.get('arch', '')) for r in rows), 'GPU measurement has no real process/interval/architecture')
    inventory = machine.get('status', {}).get('gpus', [])
    matches = [g for g in inventory if g.get('index') == index]
    require(len(matches) == 1, 'executed physical GPU absent from public machine snapshot')
    for identity in identities:
        p = identity['physical']
        require(p['gpu'] == index and uuid(p['uuid']) == gpu_uuid == uuid(identity['gpu_uuid']), 'sealed/Torch/NVML/public physical allocation differs')
        require(p['name'] == matches[0].get('name') and p['driver'] == matches[0].get('driver'), 'physical machine name/driver differs')
    return {'gpu': index, 'uuid': gpu_uuid, 'public_measured_gpu_rows': len(rows),
            'evidence': 'positive public GPU execution UUID/index/PID/time + sealed child Torch UUID/NVML + public one-GPU launch/machine snapshot'}
