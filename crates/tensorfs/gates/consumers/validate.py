#!/usr/bin/env python3
"""Refuse incomplete downloaded GPU proof Trees against the coordinator's frozen manifest."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

from make_payload import COMFY, CONSUMERS, PRE_GPU


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
    require(len(pairs) <= 1, 'public run used a different GPU count/identity')
    # A model-less parent may report CPU/-1. Those rows contribute no physical evidence.
    # Physical evidence comes from each sealed child, matched to public machine inventory.
    observed = {(i['physical']['gpu'], uuid(i['gpu_uuid'])) for i in identities}
    require(len(observed) == 1, 'children executed different physical GPU allocations')
    index, gpu_uuid = next(iter(observed))
    require(type(index) is int and index >= 0, 'child has no physical GPU index')
    if pairs:
        require(pairs == observed and all(e.get('degree') == 1 for e in executions if e.get('gpus') and
                any(type(g.get('gpu')) is int and g['gpu'] >= 0 for g in e['gpus'])), 'public physical GPU measurement differs')
        require(any(type(r.get('pid')) is int and r['pid'] > 0 and finite(r.get('start_us'), True) and
                    finite(r.get('end_us'), True) and r['end_us'] > r['start_us'] for r in rows), 'GPU measurement has no real process/interval')
    inventory = machine.get('status', {}).get('gpus', [])
    matches = [g for g in inventory if g.get('index') == index]
    require(len(matches) == 1, 'executed physical GPU absent from public machine snapshot')
    for identity in identities:
        p = identity['physical']
        require(p['gpu'] == index and uuid(p['uuid']) == gpu_uuid == uuid(identity['gpu_uuid']), 'sealed/Torch/NVML/public physical allocation differs')
        require(p['name'] == matches[0].get('name') and p['driver'] == matches[0].get('driver'), 'physical machine name/driver differs')
    return {'gpu': index, 'uuid': gpu_uuid, 'public_measured_gpu_rows': len(rows),
            'evidence': 'sealed child Torch UUID + physical NVML + public one-GPU launch/machine snapshot'}


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
                      'source/comfy-upstream.tar', 'logs/comfy-setup.log'}
    expected_files.update('source/' + name for name in spec['sources'])
    for name in requested:
        expected_files.update({f'logs/{name}.stdout', f'logs/{name}.stderr', f'reports/{name}.json',
                               f'reports/{name}-identity.json'})
    require(set(actual) == expected_files, 'unexpected/missing receipt files or model/checkpoint bytes')
    for name, expected_sha in spec['sources'].items():
        require(sha(root / 'source' / name) == expected_sha, 'retained source hash differs')
    identity, verdict = read(root / 'identity.json'), read(root / 'verdict.json')
    require(identity.get('ok') is True and verdict.get('ok') is True, 'worker failed')
    require(isinstance(identity.get('context_device'), str) and bool(identity['context_device']) and
            identity.get('python_optimize') == 0, 'missing public Context/optimized Python')
    require(identity.get('runtime_source') == spec['runtime']['source'] and identity.get('runtime') == spec['runtime']['version'], 'live Runtime differs')
    require(identity.get('native_sha256') == spec['native']['sha256'], 'native input differs')
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
        require(type(part.get('rc')) is int and part['rc'] == 0 and part.get('ok') is True, f'{name}: child failed')
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
            if isinstance(row, dict) and row.get('proof') == value.get('proof'):
                rows.append(row)
        require(rows == [value], f'{name}: report differs from actual subprocess stdout')
        if name == 'kernel':
            require(child.get('torch_present') is False and child.get('triton_present') is False and
                    child.get('torch_loaded') is False and child.get('triton_loaded') is False, 'kernel child is not clean TensorFS-only')
            contract = spec['kernel']['contract']
            require(value.get('proof') == contract['proof'], 'wrong kernel proof marker')
            require(value.get('native_source') == spec['native']['source'] and value.get('wheel_sha256') == spec['native']['sha256'] and
                    value.get('script_sha256') == spec['kernel']['sha256'], 'kernel artifact/source differs')
            require(type(value.get('device_count')) is int and value['device_count'] == 1 and
                    uuid(value.get('gpu_uuid')) == uuid(devices[0]['gpu_uuid']), 'kernel physical CUDA device differs')
            cases = value.get('cases', [])
            require(len(cases) == len(contract['cases']) and {c.get('name') for c in cases} == set(contract['cases']), 'kernel requested/missing case set differs')
            require(all(c.get('executed') is True and c.get('skipped') is False and c.get('kernel') is True and c.get('device') == 'cuda:0' and finite(c.get('max_abs_error')) for c in cases), 'kernel CPU/skip/interpreter case')
            continue
        require(child.get('cuda_available') is True and type(child.get('device_count')) is int and child['device_count'] == 1 and
                child.get('cuda_initialized_before_probe') is False, f'{name}: no exact real CUDA/pre-init observation')
        visible = child.get('cuda_visible_devices')
        require(isinstance(visible, str) and visible and ',' not in visible and visible == identity.get('cuda_visible_devices'), f'{name}: sealed GPU visibility differs')
        consumer(name, value, spec, child)
        devices.append(child)
    observed = physical(run, machine, devices, launch, payload_sha)
    return {'ok': True, 'requested': requested, 'unrequested': spec['unrequested'],
            'context_device': identity['context_device'], 'physical': observed,
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
