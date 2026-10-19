"""Existing author-main/Outputs Tree transport with fresh isolated native-wheel children."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import traceback
import venv
import zipfile

from cozy_runtime.author import Context, FileAsset, Outputs, Tree


def digest(data):
    return hashlib.sha256(data).hexdigest()


def report(stdout, marker):
    rows = []
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('proof') == marker:
            rows.append(row)
    if len(rows) != 1:
        raise ValueError(f'expected one executed {marker} JSON report, found {len(rows)}')
    return rows[0]


def main(ctx: Context, *, native_wheel: FileAsset, out: Outputs) -> Tree:
    root = Path(tempfile.mkdtemp(prefix='cozy-consumer-proof-'))
    receipts, work = root / 'receipts', root / 'work'
    for name in ('source', 'logs', 'reports'):
        (receipts / name).mkdir(parents=True)
    work.mkdir()
    (receipts / 'manifest.json').write_text(json.dumps(SPEC, indent=2) + '\n')
    (receipts / 'source/payload.py').write_bytes(Path(__file__).read_bytes())
    for name, source in FILES.items():
        (receipts / 'source' / name).write_text(source)
    identity = {'request_id': ctx.request_id, 'context_device': str(ctx.device),
                'python_optimize': sys.flags.optimize, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'triton_interpret': os.environ.get('TRITON_INTERPRET', '')}
    parts = []
    try:
        assert sys.flags.optimize == 0 and identity['triton_interpret'] in ('', '0')
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
        raw = native_wheel.read_bytes()
        assert digest(raw) == SPEC['native']['sha256'] and len(raw) == SPEC['native']['bytes'], 'wrong native FileAsset'
        wheel = work / SPEC['native']['file']
        wheel.write_bytes(raw)
        identity['native_sha256'] = digest(raw)
        comfy = work / 'comfy'
        commands = [['git', 'init', '--quiet', str(comfy)],
                    ['git', '-C', str(comfy), 'remote', 'add', 'origin', 'https://github.com/Comfy-Org/ComfyUI.git'],
                    ['git', '-C', str(comfy), 'fetch', '--quiet', '--depth=1', 'origin', SPEC['comfy_source']],
                    ['git', '-C', str(comfy), 'checkout', '--quiet', '--detach', 'FETCH_HEAD']]
        with (receipts / 'logs/comfy-setup.log').open('w') as log:
            for command in commands:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        archive = subprocess.check_output(['git', '-C', str(comfy), 'archive', '--format=tar', 'HEAD', 'comfy', 'LICENSE'])
        (receipts / 'source/comfy-upstream.tar').write_bytes(archive)
        identity['comfy_archive_sha256'] = digest(archive)
        for name in SPEC['requested']:
            clean = name == 'kernel'
            directory = work / name
            directory.mkdir()
            venv.EnvBuilder(with_pip=False).create(directory / 'venv')
            py = directory / 'venv/bin/python'
            purelib = Path(subprocess.check_output([str(py), '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True, env=env).strip())
            if not clean:
                (purelib / 'zz-parent.pth').write_text(''.join(str(p) + '\n' for p in site.getsitepackages()))
            with zipfile.ZipFile(wheel) as source:
                assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in source.namelist())
                source.extractall(purelib)
            if clean:
                script = receipts / 'source/kernel.py'
                contract = SPEC['kernel']['contract']
                extra = [arg.replace('{native_source}', SPEC['native']['source']).replace('{wheel}', str(wheel))
                         for arg in contract.get('argv', [])]
                marker = contract['proof']
            else:
                script = receipts / 'source' / SPEC['consumers'][name]['file']
                extra = ['--device', 'cuda:0', '--native-source', SPEC['native']['source'], '--wheel', str(wheel)]
                if name == 'comfy':
                    extra += ['--comfy-root', str(comfy)]
                marker = f'public-tensorfs-{name}-unet-cuda'
            command = [str(py), str(receipts / 'source/child.py'), str(receipts / f'reports/{name}-identity.json'),
                       name, str(script), *extra]
            done = subprocess.run(command, capture_output=True, text=True, env=env, cwd=directory)
            (receipts / f'logs/{name}.stdout').write_text(done.stdout)
            (receipts / f'logs/{name}.stderr').write_text(done.stderr)
            part = {'name': name, 'rc': done.returncode, 'command': command, 'ok': False}
            parts.append(part)
            try:
                result = report(done.stdout, marker)
                (receipts / f'reports/{name}.json').write_text(json.dumps(result, indent=2) + '\n')
                part['ok'] = done.returncode == 0
            except ValueError as error:
                part['failure'] = str(error)
        identity['ok'] = True
    except Exception:
        (receipts / 'logs/worker-failure.log').write_text(traceback.format_exc())
        identity['ok'] = False
    (receipts / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    verdict = {'ok': identity['ok'] and len(parts) == len(SPEC['requested']) and all(p['ok'] for p in parts),
               'requested': SPEC['requested'], 'unrequested': SPEC['unrequested'], 'parts': parts}
    (receipts / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
    inventory = {str(p.relative_to(receipts)): digest(p.read_bytes()) for p in receipts.rglob('*') if p.is_file()}
    (receipts / 'files.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return out.save_tree(receipts)
