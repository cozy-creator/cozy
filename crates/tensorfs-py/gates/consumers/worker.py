"""Existing author-main/Outputs Tree transport with fresh isolated native-wheel children."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import site
import sys
import tempfile
import traceback
from typing import Annotated
import venv
import zipfile

from cozy_runtime.author import AssetBound, Context, FileAsset, Outputs, Tree


def digest(data):
    return hashlib.sha256(data).hexdigest()


def report(stdout, marker):
    rows = []
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and (row.get('proof') == marker or row.get('schema') == marker):
            rows.append(row)
    if len(rows) != 1:
        raise ValueError(f'expected one executed {marker} JSON report, found {len(rows)}')
    return rows[0]


def main(ctx: Context, *, native_wheel: Annotated[FileAsset, AssetBound(max_bytes=NATIVE_ASSET_BYTES)],
         comfy_bundle: Annotated[FileAsset, AssetBound(max_bytes=COMFY_ASSET_BYTES)], out: Outputs) -> Tree:
    root = Path(tempfile.mkdtemp(prefix='cozy-consumer-proof-'))
    receipts, work = root / 'receipts', root / 'work'
    for name in ('source', 'logs', 'reports'):
        (receipts / name).mkdir(parents=True)
    work.mkdir()
    (receipts / 'manifest.json').write_text(json.dumps(SPEC, indent=2) + '\n')
    (receipts / 'source/payload.py').write_bytes(Path(__file__).read_bytes())
    for name, source in FILES.items():
        (receipts / 'source' / name).parent.mkdir(parents=True, exist_ok=True)
        (receipts / 'source' / name).write_text(source)
    identity = {'request_id': ctx.request_id, 'context_device': str(ctx.device),
                'python_optimize': sys.flags.optimize, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'triton_interpret': os.environ.get('TRITON_INTERPRET', '')}
    parts, operations = [], []
    helper = {}
    exec(compile(FILES['owned_child.py'], 'owned_child.py', 'exec'), helper)
    run_owned = helper['run_owned']
    native = {}
    exec(compile(FILES['native_io.py'], 'native_io.py', 'exec'), native)

    def setup(name, command, env, cwd=None, timeout=60):
        result = run_owned(command, receipts / f'logs/{name}.stdout', receipts / f'logs/{name}.stderr',
                           timeout=timeout, env=env, cwd=cwd)
        operations.append({'name': name, **result})
        if not result['ok']:
            raise RuntimeError(f'{name} failed: {result}')
        return (receipts / f'logs/{name}.stdout').read_bytes()
    try:
        assert sys.flags.optimize == 0 and identity['triton_interpret'] in ('', '0')
        identity['outer_torch_present'] = importlib.util.find_spec('torch') is not None
        assert identity['outer_torch_present'], 'outer Runtime needs Torch for existing public GPU execution identity'
        existing = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        assert existing in (None, ':4096:8', ':16:8'), 'conflicting deterministic CUDA matmul environment'
        identity['parent_cublas_workspace_config'] = existing
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONNOUSERSITE='1')
        env['COZY_PROOF_NATIVE_EXTENSION_SHA256'] = SPEC['native']['extension_sha256']
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        from cozy_runtime._build_provenance import COMMIT
        identity.update(runtime_source=COMMIT, runtime=importlib.metadata.version('cozy-runtime'))
        assert COMMIT == SPEC['runtime']['source'] and identity['runtime'] == SPEC['runtime']['version'], 'wrong live candidate Runtime'
        runtime_parent = Path(importlib.util.find_spec('cozy_runtime').origin).parent.parent
        identity['runtime_kernel_sha256'] = {name: digest((runtime_parent / name).read_bytes())
                                             for name in SPEC['runtime']['kernel_sha256']}
        assert identity['runtime_kernel_sha256'] == SPEC['runtime']['kernel_sha256'], 'wrong live Runtime native kernels'
        assert SPEC['proof_source'] == SPEC['native']['source'], 'proof/native source mismatch'
        identity['dependencies'] = {name: importlib.metadata.version(name) for name in SPEC['dependencies']['versions']}
        for name, expected in SPEC['dependencies']['versions'].items():
            actual = identity['dependencies'][name]
            assert actual == expected, 'installed dependency differs: ' + name
        raw = native_wheel.read_bytes()
        assert digest(raw) == SPEC['native']['sha256'] and len(raw) == SPEC['native']['bytes'], 'wrong native FileAsset'
        wheel = work / SPEC['native']['file']
        wheel.write_bytes(raw)
        identity['native_sha256'] = digest(raw)
        bundled = comfy_bundle.read_bytes()
        assert len(bundled) == SPEC['comfy_bundle']['bytes'] and digest(bundled) == SPEC['comfy_bundle']['sha256'], 'wrong Comfy bundle FileAsset'
        bundle = work / 'comfy.bundle'
        bundle.write_bytes(bundled)
        identity['comfy_bundle_sha256'] = digest(bundled)
        comfy = work / 'comfy'
        commands = [['git', 'init', '--quiet', str(comfy)],
                    ['git', '-C', str(comfy), 'bundle', 'verify', str(bundle)],
                    ['git', '-C', str(comfy), 'fetch', '--quiet', str(bundle), SPEC['comfy_source']],
                    ['git', '-C', str(comfy), 'checkout', '--quiet', '--detach', 'FETCH_HEAD']]
        for name, command in zip(('comfy-init', 'comfy-bundle-verify', 'comfy-fetch', 'comfy-checkout'), commands):
            setup(name, command, env)
        archive = setup('comfy-archive', ['git', '-C', str(comfy), 'archive', '--format=tar', 'HEAD', 'comfy', 'LICENSE'], env)
        (receipts / 'source/comfy-upstream.tar').write_bytes(archive)
        identity['comfy_archive_sha256'] = digest(archive)
        for name in SPEC['requested']:
            native_case = name in native['IDS']
            clean = name in ('kernel', 'fork-cleanup')
            directory = work / name
            directory.mkdir()
            venv.EnvBuilder(with_pip=False).create(directory / 'venv')
            py = directory / 'venv/bin/python'
            purelib = Path(setup(f'{name}-purelib', [str(py), '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'], env).decode().strip())
            if not clean:
                (purelib / 'zz-parent.pth').write_text(''.join(str(p) + '\n' for p in site.getsitepackages()))
            with zipfile.ZipFile(wheel) as source:
                assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in source.namelist())
                source.extractall(purelib)
            if native_case:
                row = next(c for c in SPEC['native_io']['commands'] if c['id'] == name)
                script = receipts / 'source' / row['file']
                # The metadata-only check runs in a different process. The direct proof
                # starts with no imported framework or initialized CUDA, including fork.
                code = ('import importlib.metadata as m,json,os,sys; '
                        'versions={d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}; '
                        'print(json.dumps({"executable":sys.executable,"python_optimize":sys.flags.optimize,'
                        '"no_user_site":sys.flags.no_user_site,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),'
                        '"cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),'
                        '"torch":versions.get("torch"),"triton":versions.get("triton"),"numpy":versions.get("numpy")}))')
                observed = json.loads(setup(name + '-environment', [str(py), '-c', code], env))
                observed['profile'] = row['profile']
                observed['installed_native_extension_sha256'] = digest((purelib / SPEC['native']['extension_file']).read_bytes())
                assert observed['installed_native_extension_sha256'] == SPEC['native']['extension_sha256']
                assert (observed['torch'] is None and observed['triton'] is None) if clean else bool(observed['torch'] and observed['numpy'])
                (receipts / f'reports/{name}-environment.json').write_text(json.dumps(observed, indent=2) + '\n')
                command = [str(py), '-u', str(script), *row['argv']]
                limit = row['timeout_s']
            elif clean:
                script = receipts / 'source' / SPEC['kernel']['file']
                extra = SPEC['kernel']['argv']
                marker = SPEC['kernel']['schema']
            elif name == native['RESIDENT']:
                row = SPEC['resident_decode']
                script = receipts / 'source' / row['file']
                extra = ['--gpu', '--vectors', str(receipts / 'source' / row['vectors_file'])]
            else:
                script = receipts / 'source' / SPEC['consumers'][name]['file']
                extra = ['--device', 'cuda:0', '--native-source', SPEC['native']['source'], '--wheel', str(wheel),
                         '--wheel-sha256', SPEC['native']['sha256']]
                if name == 'comfy':
                    extra += ['--comfy-root', str(comfy)]
                marker = f'public-tensorfs-{name}-unet-cuda'
            if not native_case:
                command = [str(py), str(receipts / 'source/child.py'), str(receipts / f'reports/{name}-identity.json'),
                           name, str(script), *extra]
                limit = helper['KERNEL_SECONDS'] if clean else helper['CONSUMER_SECONDS']
            done = run_owned(command, receipts / f'logs/{name}.stdout', receipts / f'logs/{name}.stderr',
                             timeout=limit, env=env, cwd=directory)
            part = {'name': name, **done, 'ok': False}
            parts.append(part)
            try:
                stdout = (receipts / f'logs/{name}.stdout').read_text()
                if native_case:
                    gpu = json.loads((receipts / 'reports/diffusers-identity.json').read_text())['gpu_uuid']
                    result = native['parse'](name, stdout, SPEC['native'], gpu)
                elif name == native['RESIDENT']:
                    gpu = json.loads((receipts / f'reports/{name}-identity.json').read_text())['gpu_uuid']
                    result = native['parse_resident'](stdout, SPEC['resident_decode'], SPEC['native'],
                                                      SPEC['dependencies']['versions']['torch'], gpu)
                else:
                    result = report(stdout, marker)
                (receipts / f'reports/{name}.json').write_text(json.dumps(result, indent=2) + '\n')
                part['ok'] = done['ok']
            except ValueError as error:
                part['failure'] = str(error)
            if not done['ok']:
                break
        identity['ok'] = True
    except Exception:
        (receipts / 'logs/worker-failure.log').write_text(traceback.format_exc())
        identity['ok'] = False
    identity['operations'] = operations
    (receipts / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    verdict = {'ok': identity['ok'] and len(parts) == len(SPEC['requested']) and all(p['ok'] for p in parts),
               'requested': SPEC['requested'], 'unrequested': SPEC['unrequested'], 'parts': parts}
    (receipts / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
    inventory = {str(p.relative_to(receipts)): digest(p.read_bytes()) for p in receipts.rglob('*') if p.is_file()}
    (receipts / 'files.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return out.save_tree(receipts)
