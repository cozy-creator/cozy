"""Native-I/O receipt oracles, recovered from the prior public proof tooling.

These validate emitted evidence only. They import no framework and execute no proof.
"""
import ast
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

IDS = ['read-engine', 'provider-cleanup', 'fork-cleanup']
ALL = ['diffusers', 'comfy', 'kernel', *IDS]
CASES = ['last-weightset', 'callback-close', 'last-pin', 'last-consumer', 'pending-consumer-b', 'pending-ptds-consumer']
READ_MARKER = 'CUDA arrays: chunk-spanning reads, prebuilt decode, raw parts, DLPack sharing/copy and retained consumers passed'
TIMEOUT = {'read-engine': 120, 'provider-cleanup': 300, 'fork-cleanup': 90}
PROVIDER_DATA_SHA256 = hashlib.sha256(bytes(range(256)) * 256).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


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
    else:
        require(case == 'pending-ptds-consumer' and value.get('scope') == 'final-release-custody', 'PTDS scope differs')
        require(all(type(value.get(k)) is int and value[k] == 2 for k in ('bind_stream', 'consumer_stream')),
                'PTDS stream identity differs')
        require(all(type(value.get(k)) is int and value[k] > 0 for k in ('bind_thread', 'consumer_thread'))
                and value['bind_thread'] != value['consumer_thread'], 'PTDS threads are not distinct')
        for name in ('distinct_threads', 'bind_thread_alive_while_pending', 'bind_thread_source_handles_released', 'ptds'):
            require(value.get(name) is True, 'PTDS evidence differs: ' + name)
        require(value.get('pending_readiness_device_proof') is False, 'PTDS final release is not a readiness proof')


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
        if case in ('pending-consumer-b', 'pending-ptds-consumer'):
            consumer_receipt(case, passed.get('consumer'))
    return {'case': name, 'format': 'json-lines', 'records': records}


def setup_names(requested):
    names = ['comfy-init', 'comfy-bundle-verify', 'comfy-fetch', 'comfy-checkout', 'comfy-archive']
    for name in requested:
        names.append(name + '-purelib')
        if name in IDS:
            names.append(name + '-environment')
    return names
