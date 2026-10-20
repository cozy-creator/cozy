#!/usr/bin/env python3
"""Refuse incomplete downloaded GPU proof Trees against the coordinator's frozen manifest."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

from make_payload import (COMFY, CONSUMERS, PRE_GPU, KERNEL_SOURCE, KERNEL_SCRIPT,
                          KERNEL_CORPUS, KERNEL_SCHEMA, KERNEL_FILE, kernel_cases)
from owned_child import CONSUMER_SECONDS, KERNEL_SECONDS, SETUP_SECONDS


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def read(path):
    def invalid(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uuid(value):
    value = str(value).lower().removeprefix('gpu-').replace('-', '')
    require(re.fullmatch('[0-9a-f]{32}', value) is not None, 'missing/invalid physical GPU UUID')
    return value


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def exact(value, expected):
    """JSON counts/ordinals/flags keep their types: 209.0 and False are not 209 and 0."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(exact(value[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(exact(v, e) for v, e in zip(value, expected))
    return value == expected


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


def completed_child(part, limit):
    return (part.get('ok') is True and type(part.get('rc')) is int and part['rc'] == 0 and
            part.get('timed_out') is False and part.get('reaped') is True and
            part.get('lingering_descendants') is False and part.get('signals') == [] and
            part.get('timeout_s') == limit and finite(part.get('elapsed_s'), True) and
            type(part.get('pid')) is int and part['pid'] > 0 and part.get('pgid') == part['pid'])


def consumer(name, value, spec, device):
    expected = spec['consumers'][name]
    require(value.get('proof') == f'public-tensorfs-{name}-unet-cuda', f'{name}: no real CUDA consumer report')
    require(value.get('device') == 'cuda:0' and value.get('gpu_origin_writer') is True, f'{name}: CPU/fallback producer')
    require(value.get('device_info', {}).get('type') == 'cuda', f'{name}: CUDA device absent')
    require(uuid(value['device_info'].get('uuid')) == uuid(device['gpu_uuid']), f'{name}: GPU UUID differs')
    require(type(value.get('tensors')) is int and value['tensors'] == expected['tensors'], f'{name}: tensor count differs')
    require(type(value.get('bytes')) is int and value['bytes'] == expected['bytes'], f'{name}: byte count differs')
    require(value.get('output_shape') == [1, 4, 8, 8] and all(type(n) is int for n in value['output_shape']), f'{name}: output shape differs')
    require(value.get('native_source') == spec['native']['source'] and value.get('wheel_sha256') == spec['native']['sha256'], f'{name}: native source/wheel differs')
    require(value.get('tensorfs') == spec['native']['version'], f'{name}: installed native version differs')
    require(value.get('script_sha256') == expected['sha256'], f'{name}: consumer source differs')
    require(isinstance(value.get('torch_cuda'), str) and bool(value['torch_cuda']) and '+cpu' not in value.get('torch', ''), f'{name}: CPU Torch')
    for key in ('assigned_without_copy', 'survives_checkpoint_close', 'reference_computed_before_write', 'producer_released_before_forward'):
        require(value.get(key) is True, f'{name}: missing {key} assertion')
    require(not value.get('skipped') and not value.get('skip'), f'{name}: consumer skipped')
    require(finite(value.get('max_abs_error')) and value['max_abs_error'] <= 1e-5, f'{name}: nonfinite/excessive numerical error')
    phases = {'producer_init', 'reference_forward', 'writer', 'open', 'read_all', 'dlpack_assign_verify', 'restored_forward'}
    require(set(value.get('timings_ms', {})) == phases and all(finite(v, True) for v in value['timings_ms'].values()), f'{name}: missing/nonfinite timing')
    require(re.fullmatch('sha256:[0-9a-f]{64}', value.get('header', '')) is not None, f'{name}: no checkpoint identity')
    if name == 'comfy':
        require(value.get('comfy_source') == COMFY and value.get('model_class') == 'comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel', 'wrong upstream Comfy model/source')
        require(value.get('comfy_kitchen') == '0.2.39' and value.get('comfy_aimdo') == '0.5.6', 'Comfy dependency pins differ')
    else:
        require(value.get('diffusers') == '0.40.0', 'Diffusers dependency pin differs')


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


def validate(root, expected, run, machine, launch, require_kernel=False):
    root = Path(root)
    spec = dict(expected)
    payload_sha = spec.pop('payload_sha256')
    require(type(spec.get('schema')) is int and spec['schema'] == 1 and type(spec.get('gpu_count')) is int and
            spec['gpu_count'] == 1 and spec.get('device') == 'cuda:0', 'wrong proof schema/GPU request')
    for name in ('native', 'runtime'):
        require(re.fullmatch('[0-9a-f]{40}', spec[name].get('source', '')) is not None and
                re.fullmatch('[0-9a-f]{64}', spec[name].get('sha256', '')) is not None, f'{name}: malformed artifact/source hash')
    require(spec['native']['source'] not in PRE_GPU, 'known pre-GPU/fork-fix source')
    requested = spec.get('requested')
    require(requested in (['diffusers', 'comfy'], ['diffusers', 'comfy', 'kernel']), 'wrong requested consumer/case set')
    require(spec.get('unrequested') == ([] if 'kernel' in requested else ['kernel']), 'unrequested cases are ambiguous')
    require(not require_kernel or 'kernel' in requested, 'kernel proof was not requested')
    require(read(root / 'manifest.json') == spec, 'returned manifest differs from coordinator freeze')
    require(sha(root / 'source/payload.py') == payload_sha, 'executed payload source differs')
    for name, (ref, _, source_sha, count, size) in CONSUMERS.items():
        c = spec['consumers'][name]
        require((c['source'], c['sha256'], c['tensors'], c['bytes']) == (ref, source_sha, count, size), f'{name}: altered frozen consumer contract')
    inventory = read(root / 'files.json')
    require(not any(p.is_symlink() for p in root.rglob('*')), 'receipt Tree contains symlinks')
    actual = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file() and p != root / 'files.json'}
    require(actual == inventory, 'receipt file inventory/hash differs')
    expected_files = {'manifest.json', 'identity.json', 'verdict.json', 'source/payload.py',
                      'source/comfy-upstream.tar'}
    setup_names = ['comfy-init', 'comfy-remote', 'comfy-fetch', 'comfy-checkout', 'comfy-archive']
    setup_names += [name + '-purelib' for name in requested]
    for name in setup_names:
        expected_files.update({f'logs/{name}.stdout', f'logs/{name}.stderr'})
    expected_files.update('source/' + name for name in spec['sources'])
    for name in requested:
        expected_files.update({f'logs/{name}.stdout', f'logs/{name}.stderr', f'reports/{name}.json',
                               f'reports/{name}-identity.json'})
    require(set(actual) == expected_files, 'unexpected/missing receipt files or model/checkpoint bytes')
    for name, expected_sha in spec['sources'].items():
        require(sha(root / 'source' / name) == expected_sha, 'retained source hash differs')
    identity, verdict = read(root / 'identity.json'), read(root / 'verdict.json')
    require(identity.get('ok') is True and verdict.get('ok') is True, 'worker failed')
    operations = identity.get('operations', [])
    require([p.get('name') for p in operations] == setup_names and
            all(completed_child(p, SETUP_SECONDS) for p in operations), 'setup child timed out/failed/not reaped')
    require(isinstance(identity.get('context_device'), str) and bool(identity['context_device']) and
            identity.get('python_optimize') == 0, 'missing public Context/optimized Python')
    require(identity.get('runtime_source') == spec['runtime']['source'] and identity.get('runtime') == spec['runtime']['version'], 'live Runtime differs')
    require(identity.get('native_sha256') == spec['native']['sha256'], 'native input differs')
    require(identity.get('outer_torch_present') is True, 'outer Runtime lacks Torch identity path')
    require(identity.get('triton_interpret') in ('', '0'), 'interpreter fallback')
    require(sha(root / 'source/comfy-upstream.tar') == identity.get('comfy_archive_sha256'), 'Comfy archive hash differs')
    with tarfile.open(root / 'source/comfy-upstream.tar') as archive:
        require(archive.pax_headers.get('comment') == COMFY, 'Comfy archive lacks its exact Git commit')
        require(all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and
                    not m.name.endswith(('.safetensors', '.cozytensors', '.ckpt', '.whl')) for m in archive.getmembers()), 'unsafe/model bytes in source archive')
    parts = verdict.get('parts', [])
    require(verdict.get('requested') == requested and verdict.get('unrequested') == spec['unrequested'] and
            len(parts) == len(requested) and {p.get('name') for p in parts} == set(requested), 'missing/duplicate/requested consumer case')
    devices = []
    for part in parts:
        name = part['name']
        require(completed_child(part, KERNEL_SECONDS if name == 'kernel' else CONSUMER_SECONDS),
                f'{name}: child failed/timed out/not reaped')
        child = read(root / f'reports/{name}-identity.json')
        require(child.get('ok') is True and child.get('python_optimize') == 0 and child.get('triton_interpret') in ('', '0'), f'{name}: skipped/interpreter/optimized child')
        require(child.get('script_completed') is True and child.get('native_extension_sha256') == spec['native']['extension_sha256'], f'{name}: native binary differs or child incomplete')
        require(child.get('cublas_workspace_config') == ':4096:8', f'{name}: missing pre-init deterministic matmul environment')
        value = read(root / f'reports/{name}.json')
        stdout = (root / f'logs/{name}.stdout').read_text()
        rows = []
        for line in stdout.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and ((name != 'kernel' and row.get('proof') == value.get('proof')) or
                                         (name == 'kernel' and row.get('schema') == KERNEL_SCHEMA)):
                rows.append(row)
        require(rows == [value], f'{name}: report differs from actual subprocess stdout')
        if name == 'kernel':
            require(child.get('torch_present') is False and child.get('triton_present') is False and
                    child.get('torch_loaded') is False and child.get('triton_loaded') is False, 'kernel child is not clean TensorFS-only')
            contract = spec['kernel']
            require((contract.get('source'), contract.get('sha256'), contract.get('file'), contract.get('schema'),
                     contract.get('corpus_sha256'), contract.get('argv')) ==
                    (KERNEL_SOURCE, KERNEL_SCRIPT, KERNEL_FILE, KERNEL_SCHEMA, KERNEL_CORPUS, ['--device', '0']),
                    'kernel frozen source/contract differs')
            require(spec['sources'].get(KERNEL_FILE) == KERNEL_SCRIPT, 'kernel script hash differs')
            files = {name: (root / 'source' / name).read_text() for name in spec['sources']}
            expected_cases = kernel_cases(files)
            schemes = sorted({c['scheme'] for c in expected_cases})
            require(contract.get('cases') == [c['case'] for c in expected_cases] and contract.get('schemes') == schemes,
                    'kernel requested case set differs')
            require(value.get('schema') == KERNEL_SCHEMA and exact(value.get('device'), {'type': 'cuda', 'ordinal': 0}), 'kernel CPU/wrong CUDA ordinal')
            require(exact(value.get('counts'), {'full': 209, 'windows': 209, 'total': 418, 'cases': 209, 'schemes': 39}) and
                    value.get('schemes') == schemes and value.get('corpus_sha256') == KERNEL_CORPUS and
                    exact(value.get('cases'), expected_cases), 'kernel missing/duplicate/skipped/non-bit-exact case')
            require(value.get('routing') == {'prepared': 'Kernel', 'host_only_exclusions': []}, 'kernel host fallback')
            require(exact(value.get('runtime'), {'torch_installed': False, 'triton_installed': False,
                                                'torch_imported': False, 'triton_imported': False}), 'kernel runtime is not clean')
            require(child.get('cuda_visible_devices') == identity.get('cuda_visible_devices') and
                    isinstance(child.get('cuda_visible_devices'), str) and bool(child['cuda_visible_devices']) and
                    ',' not in child['cuda_visible_devices'], 'kernel inherited CUDA scope differs')
            continue
        require(child.get('cuda_available') is True and type(child.get('device_count')) is int and child['device_count'] == 1 and
                child.get('cuda_initialized_before_probe') is False, f'{name}: no exact real CUDA/pre-init observation')
        visible = child.get('cuda_visible_devices')
        require(isinstance(visible, str) and visible and ',' not in visible and visible == identity.get('cuda_visible_devices'), f'{name}: sealed GPU visibility differs')
        consumer(name, value, spec, child)
        devices.append(child)
    observed = physical(run, machine, devices, launch, payload_sha)
    bound = public_tree(root, run, identity)
    return {'ok': True, 'requested': requested, 'unrequested': spec['unrequested'],
            'context_device': identity['context_device'], 'physical': observed, 'public_tree': bound,
            'scope': 'tiny model/library forwards; no ComfyUI graph/visual or unrequested kernel qualification'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tree', type=Path)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--run-show', type=Path, required=True)
    p.add_argument('--machine-show', type=Path, required=True)
    p.add_argument('--launch', type=Path, required=True, help='coordinator-recorded actual public cozy argv and payload_sha256')
    p.add_argument('--require-kernel', action='store_true')
    args = p.parse_args()
    try:
        print(json.dumps(validate(args.tree, read(args.manifest), read(args.run_show), read(args.machine_show), read(args.launch), args.require_kernel)))
    except (OSError, ValueError, TypeError, KeyError, IndexError, tarfile.TarError) as error:
        raise SystemExit(f'GPU consumer receipt refused: {error}') from error
