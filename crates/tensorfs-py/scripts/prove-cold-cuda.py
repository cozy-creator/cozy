#!/usr/bin/env python3
"""Locate a cold TensorFS CUDA crash; diagnostic only, never an acceptance proof.

Requires an already installed exact wheel. The coordinator imports only stdlib;
each stage starts a fresh interpreter without Torch/CuPy or CUDA preinitialization.
"""
import argparse
import base64
from email.parser import Parser
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile

STAGES = ('import', 'host-u8', 'cuda-u8', 'cuda-awq')
AWQ_SHA256 = '8d3db11524999d42dee58f69b664ea6c8c1ba07db784be9423b9870f992aa253'
AWQ_KEY = 'autoawq-32x64-g32-f16'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def emit(stage, point, **facts):
    print(json.dumps({'stage': stage, 'point': point, 'monotonic_ns': time.monotonic_ns(), **facts},
                     sort_keys=True), flush=True)


def wheel_files(path, digest):
    raw = path.read_bytes()
    if sha(raw) != digest:
        raise ValueError('wheel SHA256 differs')
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('unsafe wheel member')
        metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('wheel must contain one METADATA')
        fields = Parser().parsestr(wheel.read(metadata[0]).decode())
        if fields['Name'] != 'tensorfs' or not fields['Version']:
            raise ValueError('expected a TensorFS wheel')
        files = {n: sha(wheel.read(n)) for n in names if n.startswith('tensorfs/') and not n.endswith('/')}
        extensions = [n for n in files if n.startswith('tensorfs/_ext') and n.endswith(('.so', '.pyd'))]
        if len(extensions) != 1:
            raise ValueError('wheel must contain one native extension')
        return fields['Version'], files, extensions[0]


def first_awq(path):
    raw = path.read_bytes()
    if sha(raw) != AWQ_SHA256:
        raise ValueError('requires the exact current25 awq.json fixture')
    row = next(r for r in json.loads(raw)['cases'] if r.get('encoding') and r.get('expect') is not None)
    if row['key'] != AWQ_KEY or row['dtype'] != 'F16' or row['shape'] != [32, 64]:
        raise ValueError('first AWQ fixture differs')
    return row


def child(args):
    stage = args.stage
    emit(stage, 'process-start', python=sys.version, executable=sys.executable)
    if sys.flags.optimize or any(n in sys.modules or importlib.util.find_spec(n) is not None
                                for n in ('torch', 'cupy', 'triton')):
        raise RuntimeError('requires nonoptimized framework-free startup')
    version, files, extension = wheel_files(args.wheel, args.wheel_sha256)
    distribution = importlib.metadata.distribution('tensorfs')
    if distribution.version != version:
        raise ValueError('installed distribution version differs')
    for name, digest in files.items():
        if sha(Path(distribution.locate_file(name)).read_bytes()) != digest:
            raise ValueError('installed file differs: ' + name)
    emit(stage, 'provenance-ok', declared_native_source=args.native_source,
         wheel_sha256=args.wheel_sha256, extension_sha256=files[extension], version=version)
    emit(stage, 'import-start')
    import tensorfs
    from tensorfs import _ext
    if sha(Path(_ext.__file__).read_bytes()) != files[extension]:
        raise ValueError('imported native module differs')
    emit(stage, 'import-complete', native_path=_ext.__file__)
    if stage == 'import':
        emit(stage, 'stage-complete')
        return
    writer = component = checkpoint = tensors = arrays = value = None
    with tempfile.TemporaryDirectory(prefix='tensorfs-cold-cuda-') as folder:
        try:
            emit(stage, 'writer-start')
            writer = tensorfs.Writer(cache_dir=folder)
            component = writer.component('model', cls='test.ColdCuda', config={})
            if stage == 'cuda-awq':
                row = first_awq(args.vectors)
                expected = base64.b64decode(row['expect'], validate=True)
                parts = {p['role']: (p['dtype'], p['shape'], base64.b64decode(p['bits'], validate=True))
                         for p in row['parts']}
                component.put_encoded('value', dtype=row['dtype'], shape=row['shape'],
                                      scheme=row['encoding']['scheme'], params=row['encoding']['params'], parts=parts)
                component.put('window', bytes(13), dtype='U8', shape=[13])
            else:
                expected = bytes(range(64))
                component.put('value', expected, dtype='U8', shape=[len(expected)])
            emit(stage, 'writer-put-complete')
            checkpoint = writer.commit()
            emit(stage, 'writer-commit-complete')
            tensors = checkpoint.components['model'].tensors
            device = 'cpu' if stage == 'host-u8' else 'cuda:0'
            emit(stage, 'read-start', device=device)
            if stage == 'cuda-awq':
                arrays = tensors.read(['value', 'window'], device=device, decode=True)
                value = arrays['value']
            else:
                arrays = tensors.read('value', device=device, decode=False)
                value = arrays
            emit(stage, 'read-return')
            actual_device = tuple(value.__dlpack_device__())
            if actual_device != ((1, 0) if device == 'cpu' else (2, 0)):
                raise AssertionError('wrong returned device: ' + repr(actual_device))
            emit(stage, 'copy-to-host-start')
            actual = value.tobytes()
            if actual != expected:
                raise AssertionError('output bytes differ')
            emit(stage, 'bytes-match', bytes=len(actual), sha256=sha(actual), device=actual_device)
            if stage == 'cuda-awq':
                emit(stage, 'window-read-start')
                tensors.read_into('value', arrays['window'], offset=1)
                if arrays['window'].tobytes() != expected[1:14]:
                    raise AssertionError('window bytes differ')
                emit(stage, 'window-bytes-match')
            value = None
        finally:
            emit(stage, 'cleanup-start')
            if checkpoint is not None:
                checkpoint.close()
            value = arrays = tensors = checkpoint = component = writer = None
            gc.collect()
            emit(stage, 'cleanup-complete')
    if any(n in sys.modules for n in ('torch', 'cupy', 'triton')):
        raise RuntimeError('a framework was imported')
    emit(stage, 'stage-complete')


def command(args, stage):
    return [sys.executable, '-u', '-X', 'faulthandler', str(Path(__file__).resolve()),
            '--stage', stage, '--native-source', args.native_source, '--wheel', str(args.wheel.resolve()),
            '--wheel-sha256', args.wheel_sha256, '--vectors', str(args.vectors.resolve())]


def coordinate(args):
    # Validate input bytes without importing TensorFS or touching a CUDA driver.
    wheel_files(args.wheel, args.wheel_sha256)
    first_awq(args.vectors)
    env = dict(os.environ, PYTHONFAULTHANDLER='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    if not env.get('CUDA_VISIBLE_DEVICES') or ',' in env['CUDA_VISIBLE_DEVICES']:
        raise ValueError('seal exactly one CUDA_VISIBLE_DEVICES entry before starting')
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'diagnostic_only': True, 'declared_native_source': args.native_source,
              'wheel_sha256': args.wheel_sha256, 'vectors_sha256': AWQ_SHA256,
              'script_sha256': sha(Path(__file__).read_bytes()), 'stages': []}
    for stage in STAGES:
        argv = command(args, stage)
        start = time.monotonic()
        with (args.out / (stage + '.stdout')).open('wb') as stdout, (args.out / (stage + '.stderr')).open('wb') as stderr:
            try:
                completed = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           timeout=args.timeout, check=False)
                rc, timed_out = completed.returncode, False
            except subprocess.TimeoutExpired:
                rc, timed_out = None, True
        records, unparsed = [], []
        for line in (args.out / (stage + '.stdout')).read_text(errors='replace').splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                unparsed.append(line)
                continue
            if isinstance(value, dict) and value.get('stage') == stage:
                records.append(value)
            else:
                unparsed.append(line)
        done = bool(records and records[-1].get('point') == 'stage-complete' and not unparsed)
        row = {'stage': stage, 'command': argv, 'rc': rc, 'timed_out': timed_out,
               'elapsed_s': time.monotonic() - start, 'last_marker': records[-1] if records else None,
               'unparsed_stdout': unparsed, 'ok': rc == 0 and not timed_out and done}
        result['stages'].append(row)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        if not row['ok']:
            break
    result['ok'] = len(result['stages']) == len(STAGES) and all(r['ok'] for r in result['stages'])
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    return 0 if result['ok'] else 1


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-source', required=True, help='declared build receipt source, not an embedded native claim')
    parser.add_argument('--wheel', type=Path, required=True, help='already installed exact wheel; never installed by this script')
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--vectors', type=Path, required=True, help='exact current25 awq.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--stage', choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.native_source) or not re.fullmatch('[0-9a-f]{64}', args.wheel_sha256):
        parser.error('full lowercase source and wheel hashes are required')
    if not 1 <= args.timeout <= 120:
        parser.error('timeout must be1..120 seconds per child')
    if args.stage is None and args.out is None:
        parser.error('--out is required')
    return args


if __name__ == '__main__':
    options = arguments()
    if options.stage is not None:
        child(options)
    else:
        raise SystemExit(coordinate(options))
