"""Native-I/O receipt oracles, recovered from the prior public proof tooling.

These validate emitted evidence only. They import no framework and execute no proof.
"""
import ast
import base64
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import struct

IDS = ['read-engine', 'provider-cleanup', 'fork-cleanup']
RESIDENT = 'resident-decode'
ALL = ['diffusers', 'comfy', 'kernel', *IDS, RESIDENT]
RESIDENT_FILES = {
    'crates/tensorfs-py/scripts/prove-resident-decode.py': '2e46411b9d17c2fac6ddbf226e7a0edb9750c8ebc2033fc5ddaf5aca0b0c86f7',
    'crates/tensorfs-py/scripts/prove-dlpack-owner.py': '29f1e043be520010ddfd1a6981c5f786563688df241ad2e8fe19e5c4cf4a06a0',
    'crates/tensorfs/conformance/vectors/cases/comfy.json': '56758d0c82b7b0d173d8db25cd5362ad9db348839e5f4e22ce664784c0e56043',
}
RESIDENT_TIMEOUT = 180
CASES = ['last-weightset', 'callback-close', 'last-pin', 'last-consumer', 'pending-consumer-b',
         'pending-ptds-consumer', 'pending-legacy-consumer']
READ_MARKER = 'CUDA arrays: chunk-spanning reads, prebuilt decode, raw parts, DLPack sharing/copy and retained consumers passed'
TIMEOUT = {'read-engine': 120, 'provider-cleanup': 360, 'fork-cleanup': 90}
PROVIDER_DATA_SHA256 = hashlib.sha256(bytes(range(256)) * 256).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def exact(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(exact(value[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(exact(v, e) for v, e in zip(value, expected))
    return value == expected


def resident_contract(files):
    """Independent expected rows from def5's frozen corpus, never the subject code."""
    for path, expected in RESIDENT_FILES.items():
        require(path in files and sha(files[path].encode()) == expected,
                'resident source/corpus differs: ' + path)
    script, helper, vectors = RESIDENT_FILES
    cases = strict_json(files[vectors])['cases']
    positive = [r for r in cases if not r.get('refuse')]
    negative = [r for r in cases if r.get('refuse')]
    require(len(positive) == 18 and len(negative) == 14, 'resident corpus inventory differs')
    # The subject adds this exact F16 carrier variant without changing logical output.
    original = next(r for r in positive if any(p['role'] == 'weight_correction' and
                                               p['dtype'] == 'BF16' for p in r['parts']))
    correction = next(p for p in original['parts'] if p['role'] == 'weight_correction')
    values = [struct.unpack('<f', struct.pack('<I', word[0] << 16))[0]
              for word in struct.iter_unpack('<H', base64.b64decode(correction['bits'], validate=True))]
    require(all(math.isfinite(v) and struct.unpack('<e', struct.pack('<e', v))[0] == v for v in values),
            'resident correction cannot be narrowed exactly')
    derived = {**original, 'key': original['key'] + '-exact-f16-correction'}
    rows, goldens = [], []
    for row in [*positive, derived]:
        raw = base64.b64decode(row['expect'], validate=True)
        require(len(raw) == math.prod(row['shape']) * {'F32': 4, 'BF16': 2, 'F16': 2}[row['dtype']],
                'resident golden length differs')
        rows.append({'key': row['key'], 'result': 'passed', 'kind': 'golden', 'bytes': len(raw)})
        goldens.append({'key': row['key'], 'dtype': row['dtype'], 'shape': row['shape'], 'sha256': sha(raw)})
    rows.extend({'key': row['key'], 'result': 'passed', 'kind': 'expected-refusal', 'family': 'decode',
                 'value_error_output_unspecified': row['key'] == 's-rel-nan'} for row in negative)
    rows.append({'key': 'zero-output-pending-codebook', 'result': 'passed', 'kind': 'completion-control',
                 'producer_complete_at_return': True, 'output_complete_at_return': True,
                 'codebook_sha256': sha(struct.pack('<16f', *range(16)))})
    require(len(rows) == len({r['key'] for r in rows}) == 34, 'resident duplicate/missing case')
    return {'file': script, 'helper_file': helper, 'vectors_file': vectors,
            'argv': ['--gpu', '--vectors', vectors], 'timeout_s': RESIDENT_TIMEOUT,
            'sources': dict(RESIDENT_FILES), 'cases': rows, 'goldens': goldens,
            'counts': {'passed': 34, 'failed': 0, 'skipped': 0, 'original_positive': 18,
                       'derived_f16_positive': 1, 'expected_refusal': 14, 'completion_control': 1, 'raw_calls': 41}}


def parse_resident(stdout, contract, native, torch_version, gpu_uuid):
    lines = [line for line in stdout.splitlines() if line.strip()]
    require(len(lines) == 1, 'resident missing/duplicate/extra stdout record')
    value = strict_json(lines[0])
    require(isinstance(value, dict), 'resident output is not an object')
    expected = {'mode': 'gpu', **contract['counts'], 'cases': contract['cases'],
                'metadata_cast_dtypes': ['BF16', 'F16', 'F32', 'F8_E4M3'],
                'corpus_sha256': RESIDENT_FILES[contract['vectors_file']],
                'tensorfs_version': native['version'], 'torch_version': torch_version,
                'device_count': 1, 'native_sha256': native['extension_sha256'],
                'script_sha256': RESIDENT_FILES[contract['file']],
                'scope': 'resident GPU decode corpus/refusals and checked producer/output ordering; no full model or unknown CUDA-fault injection'}
    require(value.keys() == expected.keys() | {'stream_control', 'storage_controls', 'gpu_uuid'},
            'resident field inventory differs')
    require(all(exact(value[k], v) for k, v in expected.items()), 'resident counts/cases/artifact differ')
    require(uuid(value['gpu_uuid']) == uuid(gpu_uuid), 'resident physical GPU UUID differs')
    stream = value['stream_control']
    flags = {k: True for k in ('producer_pending_before', 'output_pending_before',
                               'producer_complete_at_return', 'output_complete_at_return')}
    require(isinstance(stream, dict) and stream.keys() == flags.keys() | {'source_stream', 'output_stream'}
            and all(stream[k] is v for k, v in flags.items())
            and all(type(stream[k]) is int and stream[k] > 2 for k in ('source_stream', 'output_stream'))
            and stream['source_stream'] != stream['output_stream'], 'resident stream completion differs')
    storage = {'byte_offset': 8, 'managed_deletions': 3, 'cuda_managed_deletions': 1,
               'mixed_cpu_managed_deletions': 2, 'actual_cuda_owner_released': True,
               'used_alias_refused': True, 'overlap_family': 'invalid', 'overlap_output_unchanged': True,
               'mixed_host_cuda_family': 'unsupported', 'mixed_cuda_host_family': 'unsupported',
               'mixed_outputs_unchanged': True}
    require(exact(value['storage_controls'], storage), 'resident ownership/mixed-device evidence differs')
    return value


def contract(raw):
    value = json.loads(raw)
    require(value.get('schema') == 'tensorfs.gpu-proof-command-contract/2', 'native I/O contract schema differs')
    commands = value.get('commands', [])
    require([c.get('id') for c in commands] == IDS, 'native I/O command inventory differs')
    for row in commands:
        require(row.get('timeout_s') == TIMEOUT[row['id']], 'native I/O timeout differs')
        require(row.get('profile') == ('clean' if row['id'] == 'fork-cleanup' else 'torch'), 'native I/O profile differs')
        if row['id'] == 'provider-cleanup':
            require(row.get('cases') == CASES and type(row.get('case_timeout_s')) is int
                    and row['case_timeout_s'] == 45 and row['argv'] == ['--timeout', '45'],
                    'provider child inventory/timeout differs')
        for name in [row['file'], *row['helper_files']]:
            require(PurePosixPath(name).name == name and name.endswith('.py'), 'native I/O source path differs')
    return value


def check_source(name, source):
    """Refuse stale API bodies or a silently changed owner-mode inventory."""
    tree = ast.parse(source)
    require(not any(isinstance(n, ast.Attribute) and n.attr == 'read_all' for n in ast.walk(tree)),
            f'{name}: obsolete read_all source')
    if name == 'provider-cleanup':
        cases = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'CASES' for t in n.targets)]
        require(cases == [tuple(CASES)], 'provider mode inventory differs; review the owner contract')
        flags = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                 and n.args and isinstance(n.args[0], ast.Constant)}
        require({'--timeout', '--child-case'} <= flags, 'provider argument inventory differs')
    elif name == 'read-engine':
        require(READ_MARKER in [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)],
                'read-engine success marker differs')
    elif name == 'fork-cleanup':
        require(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == 'os'
                    and n.func.attr == 'fork' for n in ast.walk(tree)), 'fork proof no longer forks')


def uuid(value):
    text = str(value).lower().removeprefix('gpu-').replace('-', '')
    require(re.fullmatch('[0-9a-f]{32}', text) is not None, 'native I/O missing physical UUID')
    return text


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def consumer_receipt(case, value):
    require(isinstance(value, dict), 'pending consumer evidence missing')
    for name in ('fill_ready_before_consume', 'pending_after_drop', 'source_retained_while_pending',
                 'foreign_python_owner_collected'):
        require(value.get(name) is True, 'pending consumer evidence differs: ' + name)
    require(finite(value.get('drop_seconds')) and value['drop_seconds'] < 0.25,
            'pending consumer release blocked')
    require(type(value.get('heartbeat_ticks')) is int and value['heartbeat_ticks'] >= 3,
            'pending consumer heartbeat missing')
    require(type(value.get('copied_bytes')) is int and value['copied_bytes'] == 65536
            and value.get('copied_sha256') == PROVIDER_DATA_SHA256, 'pending consumer byte proof differs')
    if case == 'pending-consumer-b':
        require(value.get('distinct_nondefault') is True
                and all(type(value.get(k)) is int and value[k] > 2 for k in ('bind_stream', 'consumer_stream'))
                and value['bind_stream'] != value['consumer_stream'], 'consumer B streams are not distinct')
    elif case == 'pending-ptds-consumer':
        require(value.get('scope') == 'final-release-custody', 'PTDS scope differs')
        require(all(type(value.get(k)) is int and value[k] == 2 for k in ('bind_stream', 'consumer_stream')),
                'PTDS stream identity differs')
        require(all(type(value.get(k)) is int and value[k] > 0 for k in ('bind_thread', 'consumer_thread'))
                and value['bind_thread'] != value['consumer_thread'], 'PTDS threads are not distinct')
        for name in ('distinct_threads', 'bind_thread_alive_while_pending', 'bind_thread_source_handles_released', 'ptds'):
            require(value.get(name) is True, 'PTDS evidence differs: ' + name)
        require(value.get('pending_readiness_device_proof') is False, 'PTDS final release is not a readiness proof')
    else:
        require(case == 'pending-legacy-consumer' and value.get('scope') == 'final-release-custody',
                'legacy consumer scope differs')
        require(type(value.get('bind_stream')) is int and value['bind_stream'] > 2,
                'legacy consumer did not follow a named bind stream')
        require('dlpack_stream' in value and value['dlpack_stream'] is None,
                'legacy consumer must explicitly export stream=None')
        require(type(value.get('consumer_stream')) is int and value['consumer_stream'] == 1
                and type(value.get('torch_consumer_stream')) is int and value['torch_consumer_stream'] == 0,
                'None must use native legacy stream 1 and Torch default stream 0')
        require(value.get('default_legacy') is True and value.get('distinct_nondefault') is False,
                'legacy consumer was replaced by a nondefault stream')
        require(value.get('pending_readiness_device_proof') is False,
                'legacy final release is not a readiness proof')


def parse(name, stdout, native, gpu_uuid):
    lines = [line for line in stdout.splitlines() if line.strip()]
    if name == 'read-engine':
        require(lines == [READ_MARKER], 'missing/duplicate/skip read-engine success marker')
        return {'case': name, 'format': 'pinned-source-text', 'marker': READ_MARKER}
    if name == 'fork-cleanup':
        matches = [re.fullmatch(r'CUDA fork cleanup: CPU-only parent import, child allocation/capsule release recovered ([0-9]+) bytes', line) for line in lines]
        require(len(matches) == 1 and matches[0] is not None and int(matches[0][1]) >= 33554449,
                'missing/duplicate/skip/insufficient fork recovery marker')
        return {'case': name, 'format': 'pinned-source-text', 'marker': lines[0], 'recovered_bytes': int(matches[0][1])}
    require(name == 'provider-cleanup', 'unrequested native I/O case')
    def invalid(value):
        raise ValueError('nonfinite native I/O number: ' + value)
    records = [json.loads(line, parse_constant=invalid) for line in lines]
    require(len(records) == 2 * len(CASES) + 1 and all(isinstance(row, dict) for row in records),
            'provider missing/duplicate/extra record')
    final = records[-1]
    require(final == {'result': 'passed', 'cases': CASES, 'timeout_per_child': 45}
            and type(final.get('timeout_per_child')) is int, 'provider final case set differs')
    for index, case in enumerate(CASES):
        identity, passed = records[index * 2:index * 2 + 2]
        require(identity.get('case') == case and 'result' not in identity
                and type(identity.get('device')) is int and identity['device'] == 0, 'provider identity/case/device differs')
        require(uuid(identity.get('uuid')) == uuid(gpu_uuid), 'provider/public GPU UUID differs')
        require(identity.get('extension_sha256') == native['extension_sha256']
                and identity.get('distribution') == native['version'] and identity.get('tensorfs') == native['version'],
                'provider loaded native artifact differs')
        require(passed.get('case') == case and passed.get('result') == 'passed'
                and not passed.get('skipped') and not passed.get('skip'), 'provider partial/skipped/error case')
        clean = passed.get('cleanup', {})
        require(all(type(clean.get(k)) is int and clean[k] == 0 for k in ('sets', 'mapped', 'idle', 'slots')),
                'provider cleanup is incomplete')
        require(type(passed.get('callback', {}).get('chunks')) is int and passed['callback']['chunks'] > 0,
                'provider never fetched real chunks')
        require(any(type(c.get('bytes')) is int and c['bytes'] == 65536 and finite(c.get('ns'))
                    for c in passed.get('h2d', [])), 'provider missing real H2D record')
        if case in ('pending-consumer-b', 'pending-ptds-consumer', 'pending-legacy-consumer'):
            consumer_receipt(case, passed.get('consumer'))
    return {'case': name, 'format': 'json-lines', 'records': records}


def setup_names(requested):
    names = ['comfy-init', 'comfy-bundle-verify', 'comfy-fetch', 'comfy-checkout', 'comfy-archive']
    for name in requested:
        names.append(name + '-purelib')
        if name in IDS:
            names.append(name + '-environment')
    return names
