#!/usr/bin/env python3
"""Freeze ordinary public-cozy consumer source and exact candidate wheel metadata."""

import argparse
import ast
import base64
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
COMFY = "35f6d4f8bdefe6eea17bfc4631bba8c7f5a66360"
CONSUMERS = {
    "diffusers": ("05dc128f291878eccea757aa9c1c5db230071bd9", "prove-diffusers-unet-cpu.py",
                  "100aa089ea0394837faf048c5979ace39b5357319a60f01484f7674b700eceb8", 208, 802576),
    "comfy": ("868c4e9635c2e522ba80077307be3b8054791584", "prove-comfy-unet.py",
              "7b0c10a859d553674cd55a3259999a92bd8476982bb979e494060b70068b6dfb", 130, 2893456),
}
PRE_GPU = {"88a49ac8e634035301a3314088a856d214e55db8", "f0601a0d5323cd4ed3da829072ab9d6ca61d7748"}
KERNEL_SOURCE = 'f8245484c34e45ece257715569468ced526c94fd'
KERNEL_SCRIPT = '7660d1cd97c95ca460e8f4bc884c01ee1f68b8adf8db8d755adda8fe8e053475'
KERNEL_CORPUS = '92a0f4f01f4490c2f7f8c391e5eb600f5e5b33d647eed48d91ff7c21e7af57d4'
KERNEL_SCHEMA = 'tensorfs.cuda-kernel-conformance/1'
KERNEL_FILE = 'kernel/scripts/cuda_kernel_conform.py'
KERNEL_CASES = 'kernel/conformance/vectors/cases/'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], timeout=15)


def kernel_cases(files):
    """Derive the exact receipt rows from the owner's frozen, hash-checked corpus."""
    corpus, rows = hashlib.sha256(), []
    for name, contents in sorted(files.items()):
        if not name.startswith(KERNEL_CASES):
            continue
        filename = Path(name).name
        corpus.update(filename.encode() + b'\0' + contents.encode())
        for case in json.loads(contents)['cases']:
            if case.get('encoding') is None or case.get('expect') is None:
                continue
            length = len(base64.b64decode(case['expect'], validate=True))
            rows.append({'case': f"{Path(filename).stem}/{case['component']}/{case['key']}",
                         'scheme': case['encoding']['scheme'], 'dtype': case['dtype'], 'shape': case['shape'],
                         'full_bytes': length, 'full_passed': True,
                         'window': {'offset': 1, 'length': min(13, max(0, length - 1)), 'passed': True}})
    if corpus.hexdigest() != KERNEL_CORPUS or len(rows) != 209 or len({r['case'] for r in rows}) != 209 or len({r['scheme'] for r in rows}) != 39:
        raise ValueError('frozen kernel corpus/hash/case set differs')
    return rows


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(value, width):
    if re.fullmatch(f"[0-9a-f]{{{width}}}", value) is None:
        raise ValueError(f"expected a full {width}-character source/hash pin")
    return value


def wheel(path, name, expected, source):
    data = path.read_bytes()
    if digest(data) != pin(expected, 64):
        raise ValueError(f"{name} wheel hash differs from the explicit freeze")
    with zipfile.ZipFile(path) as archive:
        extension = None
        metadata = [p for p in archive.namelist() if p.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError("wheel must contain exactly one METADATA")
        fields = Parser().parsestr(archive.read(metadata[0]).decode())
        if fields['Name'].lower().replace('_', '-') != name:
            raise ValueError(f"expected {name} wheel")
        if not fields['Version'] or not fields['Requires-Python']:
            raise ValueError("wheel lacks version or Requires-Python")
        if name == 'cozy-runtime':
            tree = ast.parse(archive.read('cozy_runtime/_build_provenance.py').decode())
            commits = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'COMMIT' for t in n.targets)]
            if commits != [source]:
                raise ValueError("Runtime wheel COMMIT differs from the explicit source pin")
        else:
            extensions = [p for p in archive.namelist() if p.startswith('tensorfs/_ext') and p.endswith(('.so', '.pyd'))]
            if len(extensions) != 1:
                raise ValueError("native wheel needs exactly one TensorFS extension")
            extension = digest(archive.read(extensions[0]))
    result = {'file': path.name, 'sha256': expected, 'bytes': len(data), 'source': pin(source, 40),
              'version': fields['Version'], 'requires_python': fields['Requires-Python']}
    if extension:
        result['extension_sha256'] = extension
    return result


def generate(args):
    if args.native_source in PRE_GPU:
        raise ValueError("this known pre-GPU/fork-fix native source cannot qualify the final GPU proof")
    native = wheel(args.native_wheel, 'tensorfs', args.native_sha256, args.native_source)
    runtime = wheel(args.runtime_wheel, 'cozy-runtime', args.runtime_sha256, args.runtime_source)
    if native['requires_python'] != runtime['requires_python']:
        raise ValueError("review the candidate wheels' different Requires-Python constraints")
    files, consumers = {}, {}
    for name, (ref, filename, expected, count, size) in CONSUMERS.items():
        data = git(args.creator_repo, 'show', f'{ref}:crates/tensorfs/scripts/{filename}')
        if digest(data) != expected:
            raise ValueError(f"frozen {name} source differs")
        files[filename] = data.decode()
        consumers[name] = {'source': ref, 'file': filename, 'sha256': expected, 'tensors': count, 'bytes': size}
    files['child.py'] = (HERE / 'child.py').read_text()
    files['owned_child.py'] = (HERE / 'owned_child.py').read_text()
    spec = {'schema': 1, 'requested': ['diffusers', 'comfy'], 'unrequested': ['kernel'],
            'native': native, 'runtime': runtime, 'consumers': consumers, 'comfy_source': COMFY,
            'gpu_count': 1, 'device': 'cuda:0', 'cublas_workspace_config': ':4096:8'}
    if args.include_kernel:
        data = git(args.creator_repo, 'show', f'{KERNEL_SOURCE}:crates/tensorfs/scripts/cuda_kernel_conform.py')
        if digest(data) != KERNEL_SCRIPT:
            raise ValueError('frozen kernel script hash differs')
        files[KERNEL_FILE] = data.decode()
        prefix = 'crates/tensorfs/conformance/vectors/cases/'
        paths = git(args.creator_repo, 'ls-tree', '-r', '--name-only', KERNEL_SOURCE, prefix).decode().splitlines()
        for path in paths:
            if not path.endswith('.json'):
                raise ValueError('unexpected file in frozen kernel corpus')
            files[KERNEL_CASES + Path(path).name] = git(args.creator_repo, 'show', f'{KERNEL_SOURCE}:{path}').decode()
        rows = kernel_cases(files)
        spec['kernel'] = {'source': KERNEL_SOURCE, 'sha256': KERNEL_SCRIPT, 'file': KERNEL_FILE,
                          'schema': KERNEL_SCHEMA, 'corpus_sha256': KERNEL_CORPUS,
                          'cases': [r['case'] for r in rows], 'schemes': sorted({r['scheme'] for r in rows}),
                          'argv': ['--device', '0']}
        spec['requested'].append('kernel')
        spec['unrequested'] = []
    spec['sources'] = {name: digest(text.encode()) for name, text in files.items()}
    dependencies = [f"cozy-runtime=={runtime['version']}", f"tensorfs=={native['version']}",
                    'torch==2.14.0', 'diffusers==0.40.0', 'numpy', 'einops', 'safetensors', 'Pillow',
                    'scipy', 'tqdm', 'psutil', 'pyyaml', 'packaging', 'comfy-kitchen==0.2.39', 'comfy-aimdo==0.5.6']
    output = args.output.resolve()
    sidecar = output.with_suffix('.manifest.json')
    if output.exists() or sidecar.exists():
        raise ValueError("choose new payload/manifest paths; frozen artifacts are never overwritten")
    paths = {name: os.path.relpath(path.resolve(), output.parent)
             for name, path in [('cozy-runtime', args.runtime_wheel), ('tensorfs', args.native_wheel)]}
    header = '# /// script\n# requires-python = ' + json.dumps(runtime['requires_python']) + '\n'
    header += '# dependencies = ' + json.dumps(dependencies) + '\n# [tool.uv.sources]\n'
    header += ''.join(f'# {name} = {{ path = {json.dumps(path)} }}\n' for name, path in paths.items())
    header += '# ///\n'
    payload = header + '\nSPEC = ' + repr(spec) + '\nFILES = ' + repr(files) + '\n\n' + (HERE / 'worker.py').read_text()
    ast.parse(payload)
    if len(payload.encode()) > 32 << 20:
        raise ValueError("payload exceeds the existing public script source bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    sidecar.write_text(json.dumps({**spec, 'payload_sha256': digest(payload.encode())}, indent=2) + '\n')
    return sidecar


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--creator-repo', type=Path, required=True)
    p.add_argument('--native-wheel', type=Path, required=True)
    p.add_argument('--native-source', required=True)
    p.add_argument('--native-sha256', required=True)
    p.add_argument('--runtime-wheel', type=Path, required=True)
    p.add_argument('--runtime-source', required=True)
    p.add_argument('--runtime-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-kernel', action='store_true', help='include the frozen f824 clean-child 209-case kernel proof')
    print(generate(p.parse_args()))
