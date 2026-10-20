#!/usr/bin/env python3
"""Freeze ordinary public-cozy consumer source and exact candidate wheel metadata."""

import argparse
import ast
import base64
from email.parser import Parser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import zipfile
import native_io

HERE = Path(__file__).resolve().parent
COMFY = "35f6d4f8bdefe6eea17bfc4631bba8c7f5a66360"
CONSUMERS = {
    "diffusers": ("prove-diffusers-unet-cpu.py", 208, 802576),
    "comfy": ("prove-comfy-unet.py", 130, 2893456),
}
PRE_GPU = {"88a49ac8e634035301a3314088a856d214e55db8", "f0601a0d5323cd4ed3da829072ab9d6ca61d7748"}
SCRIPT_ROOT = 'crates/tensorfs-py/scripts/'
KERNEL_CORPUS = 'fa2f92b926ba1f0e1195027eaf4066ae0d72c20ca4119584ca8547d5e91167b1'
KERNEL_SCHEMA = 'tensorfs.cuda-kernel-conformance/1'
KERNEL_FILE = 'crates/tensorfs-py/scripts/cuda_kernel_conform.py'
KERNEL_CASES = 'crates/tensorfs/conformance/vectors/cases/'
KERNEL_PARTITIONS = {'nonzero-element': 18, 'group': 15, 'rotation': 7,
                     'row': 18, 'matrix': 1, 'near-end-partial': 18}
KERNEL_COUNTS = {'full': 227, 'odd_windows': 227, 'partition_windows': 77,
                 'windows': 304, 'total': 531, 'cases': 227, 'schemes': 42}
PARTITION_SCHEMES = {'comfy.asym_w4a8_int8', 'comfy.w6a8_int8', 'comfy.convrot_w4a4'}
RUNTIME_TAG = 'cp312-abi3-manylinux_2_28_x86_64'
RUNTIME_KERNELS = {f'cozy_runtime/_kernels/{name}.abi3.so' for name in ('_gemm', '_norm', '_fa3')}
PYTORCH_INDEX = 'https://download.pytorch.org/whl/cu130'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], timeout=15)


def expected_partitions(case, length):
    """Byte windows from frozen geometry, independent of the GPU subject script."""
    if case['encoding']['scheme'] not in PARTITION_SCHEMES:
        return []
    width = {'F32': 4, 'BF16': 2, 'F16': 2}[case['dtype']]
    shape, params = case['shape'], case['encoding']['params']
    row_bytes = shape[-1] * width
    offsets = [('nonzero-element', 3 * width + 1)]
    for label, elements in [('group', params.get('group_size')), ('rotation', params['rotation_size'])]:
        if elements is not None and elements < shape[-1]:
            offsets.append((label, elements * width - width + 1))
    if math.prod(shape[:-1]) > 1:
        offsets.append(('row', row_bytes - width + 1))
    if len(shape) > 2 and math.prod(shape[:-2]) > 1:
        offsets.append(('matrix', shape[-2] * row_bytes - width + 1))
    offsets.append(('near-end-partial', length - 13))
    if any(offset < width or offset + 13 > length or not (offset % width or 13 % width)
           for _, offset in offsets):
        raise ValueError('frozen kernel boundary window is outside its logical bytes')
    return [{'kind': kind, 'offset': offset, 'length': 13, 'passed': True} for kind, offset in offsets]


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
            row = {'case': f"{Path(filename).stem}/{case['component']}/{case['key']}",
                   'scheme': case['encoding']['scheme'], 'dtype': case['dtype'], 'shape': case['shape'],
                   'full_bytes': length, 'full_passed': True}
            window = min(13, max(0, length - 1))
            if window:
                row['window'] = {'offset': 1, 'length': window, 'passed': True}
            partitions = expected_partitions(case, length)
            if partitions:
                row['partition_windows'] = partitions
            rows.append(row)
    if (corpus.hexdigest() != KERNEL_CORPUS or len(rows) != 227 or len({r['case'] for r in rows}) != 227
            or len({r['scheme'] for r in rows}) != 42 or sum('window' in r for r in rows) != 227):
        raise ValueError('frozen kernel corpus/hash/case set differs')
    counts = {kind: 0 for kind in KERNEL_PARTITIONS}
    for row in rows:
        for window in row.get('partition_windows', []):
            counts[window['kind']] += 1
    if counts != KERNEL_PARTITIONS or sum(counts.values()) != KERNEL_COUNTS['partition_windows']:
        raise ValueError('frozen kernel boundary inventory differs')
    return rows


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(value, width):
    if re.fullmatch(f"[0-9a-f]{{{width}}}", value) is None:
        raise ValueError(f"expected a full {width}-character source/hash pin")
    return value


def source_file(repo, source, name):
    path = SCRIPT_ROOT + name
    raw = git(repo, 'show', f'{source}:{path}')
    native_io.check_source(name, raw.decode())
    return path, raw


def requirements(path):
    raw = path.read_bytes()
    values, versions = [], {}
    for line in raw.decode().splitlines():
        value = line.strip()
        if not value or value.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9!+_.-]+)', value)
        if match is None:
            raise ValueError('requirements must contain exact name==version pins only')
        name = match[1].lower().replace('_', '-')
        if name in versions or name in {'tensorfs', 'cozy-runtime'}:
            raise ValueError('duplicate or artifact-owned dependency in requirements')
        values.append(value)
        versions[name] = match[2]
    needed = {'torch', 'diffusers', 'numpy', 'einops', 'safetensors', 'pillow', 'scipy',
              'tqdm', 'psutil', 'pyyaml', 'packaging', 'comfy-kitchen', 'comfy-aimdo'}
    if not needed <= versions.keys():
        raise ValueError('requirements missing consumer dependency pins')
    if not versions['torch'].endswith('+cu130'):
        raise ValueError('Torch requires its exact CUDA build tag +cu130 in the dependency freeze')
    return {'sha256': digest(raw), 'requirements': values, 'versions': versions}, raw


def comfy_bundle(path, expected):
    raw = path.read_bytes()
    if digest(raw) != pin(expected, 64):
        raise ValueError('Comfy bundle hash differs from the explicit freeze')
    header = raw.split(b'\n\n', 1)[0].decode()
    if not header.startswith(('# v2 git bundle\n', '# v3 git bundle\n')) or not any(
            line.startswith(COMFY + ' ') for line in header.splitlines()):
        raise ValueError('Comfy bundle must name the exact upstream commit')
    if any(line.startswith('-') for line in header.splitlines()):
        raise ValueError('Comfy bundle requires unavailable prerequisite commits')
    return {'file': path.name, 'bytes': len(raw), 'sha256': expected, 'source': COMFY}


def wheel(path, name, expected, source):
    data = path.read_bytes()
    if digest(data) != pin(expected, 64):
        raise ValueError(f"{name} wheel hash differs from the explicit freeze")
    with zipfile.ZipFile(path) as archive:
        extension = None
        runtime_kernels = None
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
            # Runtime9b1 scripts/verify-wheels.py: x86_64 Linux native wheel,
            # three exact kernel modules. Presence is artifact preflight, not GPU proof.
            wheel_info = Parser().parsestr(archive.read(metadata[0].removesuffix('METADATA') + 'WHEEL').decode())
            if (wheel_info['Root-Is-Purelib'] != 'false' or wheel_info.get_all('Tag') != [RUNTIME_TAG]
                    or path.name != f"cozy_runtime-{fields['Version']}-{RUNTIME_TAG}.whl"):
                raise ValueError('GPU proof requires the non-pure Linux x86_64 Runtime wheel')
            modules = {name for name in archive.namelist() if name.startswith('cozy_runtime/_kernels/')
                       and name.endswith(('.so', '.pyd'))}
            if modules != RUNTIME_KERNELS:
                raise ValueError('Runtime native kernel module set differs')
            runtime_kernels = {name: digest(archive.read(name)) for name in sorted(modules)}
        else:
            extensions = [p for p in archive.namelist() if p.startswith('tensorfs/_ext') and p.endswith(('.so', '.pyd'))]
            if len(extensions) != 1:
                raise ValueError("native wheel needs exactly one TensorFS extension")
            extension = digest(archive.read(extensions[0]))
    result = {'file': path.name, 'sha256': expected, 'bytes': len(data), 'source': pin(source, 40),
              'version': fields['Version'], 'requires_python': fields['Requires-Python']}
    if extension:
        result['extension_sha256'] = extension
        result['extension_file'] = extensions[0]
    if runtime_kernels is not None:
        result.update(tag=RUNTIME_TAG, kernel_sha256=runtime_kernels)
    return result


def generate(args):
    proof_source = pin(args.proof_source, 40)
    if proof_source != args.native_source:
        raise ValueError('proof source must equal the native wheel producer source')
    if args.native_source in PRE_GPU:
        raise ValueError("this known pre-GPU/fork-fix native source cannot qualify the final GPU proof")
    native = wheel(args.native_wheel, 'tensorfs', args.native_sha256, args.native_source)
    runtime = wheel(args.runtime_wheel, 'cozy-runtime', args.runtime_sha256, args.runtime_source)
    if native['requires_python'] != runtime['requires_python']:
        raise ValueError("review the candidate wheels' different Requires-Python constraints")
    dependencies, dependency_bytes = requirements(args.requirements)
    upstream = comfy_bundle(args.comfy_bundle, args.comfy_sha256)
    files, consumers = {}, {}
    for name, (filename, count, size) in CONSUMERS.items():
        path, data = source_file(args.creator_repo, proof_source, filename)
        files[filename] = data.decode()
        consumers[name] = {'source': proof_source, 'path': path, 'file': filename,
                           'sha256': digest(data), 'tensors': count, 'bytes': size}
    files['child.py'] = (HERE / 'child.py').read_text()
    files['owned_child.py'] = (HERE / 'owned_child.py').read_text()
    files['native_io.py'] = (HERE / 'native_io.py').read_text()
    files['requirements.txt'] = dependency_bytes.decode()
    spec = {'schema': 1, 'requested': ['diffusers', 'comfy'], 'unrequested': ['kernel'],
            'native': native, 'runtime': runtime, 'consumers': consumers, 'comfy_source': COMFY,
            'proof_source': proof_source, 'dependencies': dependencies, 'comfy_bundle': upstream,
            'gpu_count': 1, 'device': 'cuda:0', 'cublas_workspace_config': ':4096:8'}
    if args.include_kernel:
        _, data = source_file(args.creator_repo, proof_source, 'cuda_kernel_conform.py')
        files[KERNEL_FILE] = data.decode()
        prefix = 'crates/tensorfs/conformance/vectors/cases/'
        paths = git(args.creator_repo, 'ls-tree', '-r', '--name-only', proof_source, prefix).decode().splitlines()
        for path in paths:
            if not path.endswith('.json'):
                raise ValueError('unexpected file in frozen kernel corpus')
            files[path] = git(args.creator_repo, 'show', f'{proof_source}:{path}').decode()
        rows = kernel_cases(files)
        spec['kernel'] = {'source': proof_source, 'sha256': digest(data), 'file': KERNEL_FILE,
                          'schema': KERNEL_SCHEMA, 'corpus_sha256': KERNEL_CORPUS,
                          'counts': dict(KERNEL_COUNTS), 'partition_windows': dict(KERNEL_PARTITIONS),
                          'cases': [r['case'] for r in rows], 'schemes': sorted({r['scheme'] for r in rows}),
                          'argv': ['--device', '0']}
        spec['requested'].append('kernel')
    raw_contract = (HERE / 'native-io-contract.json').read_bytes()
    contract = native_io.contract(raw_contract)
    required_modes = getattr(args, 'require_native_mode', [])
    if any(mode not in native_io.IDS for mode in required_modes):
        raise ValueError('required native mode absent from the owner-approved source/command inventory')
    if required_modes and not args.include_native_io:
        raise ValueError('required native modes need --include-native-io')
    if args.include_native_io:
        files['native/contract.json'] = raw_contract.decode()
        commands = []
        for row in contract['commands']:
            copied = dict(row)
            frozen = []
            for filename in [row['file'], *row['helper_files']]:
                path, raw = source_file(args.creator_repo, proof_source, filename)
                if filename == row['file']:
                    native_io.check_source(row['id'], raw.decode())
                files[path] = raw.decode()
                frozen.append({'path': path, 'sha256': digest(raw), 'bytes': len(raw)})
            copied.update(file=frozen[0]['path'], helper_files=frozen[1:], sources=frozen)
            commands.append(copied)
        spec['native_io'] = {'source': proof_source, 'contract_sha256': digest(raw_contract), 'commands': commands}
        spec['requested'].extend(native_io.IDS)
    if getattr(args, 'include_resident_decode', False):
        for path in native_io.RESIDENT_FILES:
            try:
                files[path] = git(args.creator_repo, 'show', f'{proof_source}:{path}').decode()
            except subprocess.CalledProcessError as error:
                raise ValueError('resident source/corpus unavailable: ' + path) from error
        spec['resident_decode'] = {'source': proof_source, **native_io.resident_contract(files)}
        spec['requested'].append(native_io.RESIDENT)
    spec['pending'] = contract['pending']
    spec['unrequested'] = [name for name in native_io.ALL if name not in spec['requested']]
    spec['sources'] = {name: digest(text.encode()) for name, text in files.items()}
    dependency_pins = [f"cozy-runtime=={runtime['version']}", f"tensorfs=={native['version']}", *dependencies['requirements']]
    output = args.output.resolve()
    sidecar = output.with_suffix('.manifest.json')
    if output.exists() or sidecar.exists():
        raise ValueError("choose new payload/manifest paths; frozen artifacts are never overwritten")
    paths = {name: os.path.relpath(path.resolve(), output.parent)
             for name, path in [('cozy-runtime', args.runtime_wheel), ('tensorfs', args.native_wheel)]}
    header = '# /// script\n# requires-python = ' + json.dumps(runtime['requires_python']) + '\n'
    header += '# dependencies = ' + json.dumps(dependency_pins) + '\n# [tool.uv.sources]\n'
    header += ''.join(f'# {name} = {{ path = {json.dumps(path)} }}\n' for name, path in paths.items())
    header += '# torch = { index = "pytorch-cu130" }\n# [[tool.uv.index]]\n'
    header += '# name = "pytorch-cu130"\n# url = ' + json.dumps(PYTORCH_INDEX) + '\n# explicit = true\n'
    header += '# ///\n'
    worker = (HERE / 'worker.py').read_text().replace('NATIVE_ASSET_BYTES', str(native['bytes'])).replace('COMFY_ASSET_BYTES', str(upstream['bytes']))
    payload = header + '\nSPEC = ' + repr(spec) + '\nFILES = ' + repr(files) + '\n\n' + worker
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
    p.add_argument('--proof-source', required=True, help='full exact source SHA, equal to the native wheel producer')
    p.add_argument('--native-wheel', type=Path, required=True)
    p.add_argument('--native-source', required=True)
    p.add_argument('--native-sha256', required=True)
    p.add_argument('--runtime-wheel', type=Path, required=True)
    p.add_argument('--runtime-source', required=True)
    p.add_argument('--runtime-sha256', required=True)
    p.add_argument('--requirements', type=Path, required=True, help='reviewed exact name==version dependency pins')
    p.add_argument('--comfy-bundle', type=Path, required=True, help='self-contained local Git bundle at the fixed upstream commit')
    p.add_argument('--comfy-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-kernel', action='store_true', help='include the source-matched clean-child kernel proof')
    p.add_argument('--include-native-io', action='store_true', help='include source-matched read/provider/fork proofs')
    p.add_argument('--include-resident-decode', action='store_true', help='include the frozen resident GPU decode proof')
    p.add_argument('--require-native-mode', action='append', default=[], help='fail if this owner-approved mode is absent')
    print(generate(p.parse_args()))
