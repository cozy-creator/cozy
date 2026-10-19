"""Frozen native-I/O proof oracles and portable artifact inspection; no device imports."""
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import re
import tarfile

SOURCE = '3be2dbc2a1f6cfd867199d77a6ca94a4240be44b'
CONTRACT_SHA = '160d4c2604bb221536855a8c76eac169074413906fa442335bb8c9c30a60681f'
IDS = ['read-engine', 'provider-cleanup', 'fork-cleanup']
HOSTPIN = 'direct-host-pin-and-zero'
EXCLUDED = 'native-full-element-windows'
ALL = ['diffusers', 'comfy', 'kernel', *IDS, HOSTPIN, EXCLUDED]
CASES = ['last-weightset', 'callback-close', 'last-pin', 'last-consumer']
READ_MARKER = 'CUDA arrays: chunk-spanning reads, prebuilt decode, raw parts, DLPack sharing/copy and retained consumers passed'
PIN_MARKER = 'HostPin: owned explicit/implicit unregister, borrowed explicit/implicit preservation, engine preservation and context restoration passed'
PIN_TEST = 'cuda_staging_pins_device_copies_and_chunk_spans_keep_exact_bytes'
PIN_ARGV = ['--ignored', '--exact', PIN_TEST, '--nocapture', '--test-threads=1']
PIN_SHA = '91db7556e78bbffdfccbcd91e671b205b20261589bcb37bbfd030c54c636e514'
PIN_BINARY_SHA = 'c6cf3db17c1fd21651a50095a04c42061580447c32964980407e95f3b00c280a'
PIN_SOURCE = 'c6f535feeca522b4f4375fa8840fba2b5feb618a'
CORE = '0800dbcfee93b887891624a303124de2c017a034'
TIMEOUT = {'read-engine': 120, 'provider-cleanup': 210, 'fork-cleanup': 90, HOSTPIN: 120}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def contract(raw):
    require(sha(raw) == CONTRACT_SHA, 'native I/O command contract hash differs')
    return json.loads(raw)


def uuid(value):
    text = str(value).lower().removeprefix('gpu-').replace('-', '')
    require(re.fullmatch('[0-9a-f]{32}', text) is not None, 'native I/O missing physical UUID')
    return text


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def parse(name, stdout, native, gpu_uuid):
    """Only emitted evidence: text assertions are never invented JSON counters."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if name == 'read-engine':
        require(lines == [READ_MARKER], 'missing/duplicate/skip read-engine success marker')
        return {'case': name, 'format': 'pinned-source-text', 'marker': READ_MARKER}
    if name == 'fork-cleanup':
        matches = [re.fullmatch(r'CUDA fork cleanup: CPU-only parent import, child allocation/capsule release recovered ([0-9]+) bytes', line) for line in lines]
        matches = [m for m in matches if m is not None]
        require(len(matches) == 1 and lines == [matches[0].group(0)] and int(matches[0][1]) >= 33554449, 'missing/duplicate/skip/insufficient fork recovery marker')
        return {'case': name, 'format': 'pinned-source-text', 'marker': matches[0][0], 'recovered_bytes': int(matches[0][1])}
    if name == HOSTPIN:
        require(stdout.count(PIN_MARKER) == 1, 'missing/duplicate HostPin marker')
        require(lines.count('running 1 test') == 1 and len(re.findall(r'^test ' + PIN_TEST + r' \.\.\.', stdout, re.M)) == 1, 'HostPin selected test differs')
        summary = re.findall(r'^test result: ok\. ([0-9]+) passed; ([0-9]+) failed; ([0-9]+) ignored; [0-9]+ measured; [0-9]+ filtered out; finished in [0-9.]+s$', stdout, re.M)
        require(summary == [('1', '0', '0')], 'HostPin partial/ignored/failed libtest result')
        return {'case': name, 'format': 'libtest-text', 'marker': PIN_MARKER, 'passed': 1, 'failed': 0, 'ignored': 0, 'selected_test': PIN_TEST}
    require(name == 'provider-cleanup', 'unrequested native I/O case')
    def invalid(value):
        raise ValueError('nonfinite native I/O number: ' + value)
    records = []
    for line in lines:
        try:
            row = json.loads(line, parse_constant=invalid)
        except json.JSONDecodeError as error:
            raise ValueError('provider emitted partial/error/non-JSON output') from error
        require(isinstance(row, dict), 'provider emitted a non-record')
        records.append(row)
    require(len(records) == 9, 'provider missing/duplicate/extra identity or result')
    require(records[-1] == {'result': 'passed', 'cases': CASES, 'timeout_per_child': 45} and
            type(records[-1].get('timeout_per_child')) is int, 'provider final case set differs')
    for index, name in enumerate(CASES):
        identity, passed = records[index * 2:index * 2 + 2]
        require(identity.get('case') == name and 'result' not in identity and type(identity.get('device')) is int and identity['device'] == 0, 'provider identity/case/device differs')
        require(uuid(identity.get('uuid')) == uuid(gpu_uuid), 'provider/public GPU UUID differs')
        require(identity.get('extension_sha256') == native['extension_sha256'] and identity.get('distribution') == native['version'] and identity.get('tensorfs') == native['version'], 'provider loaded native artifact differs')
        require(passed.get('case') == name and passed.get('result') == 'passed' and not passed.get('skipped') and not passed.get('skip'), 'provider partial/skipped/error case')
        clean = passed.get('cleanup', {})
        require(all(type(clean.get(k)) is int and clean[k] == 0 for k in ('sets', 'mapped', 'idle', 'slots')), 'provider cleanup is incomplete')
        require(type(passed.get('callback', {}).get('chunks')) is int and passed['callback']['chunks'] > 0, 'provider never fetched real chunks')
        require(any(type(c.get('bytes')) is int and c['bytes'] == 65536 and finite(c.get('ns'), True) for c in passed.get('h2d', [])), 'provider missing real H2D record')
    return {'case': 'provider-cleanup', 'format': 'json-lines', 'records': records}


def archive_files(raw):
    """Read only regular bounded members; never delegate extraction to tarfile."""
    members = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and '..' not in path.parts and str(path) == member.name.rstrip('/'), 'unsafe HostPin archive path')
            if member.isdir():
                require(str(path) == 'hostpin', 'unexpected HostPin directory')
                continue
            require(member.isfile() and member.name not in members and member.size <= 10 << 20, 'HostPin link/duplicate/oversize member')
            require(path.parent == PurePosixPath('hostpin'), 'unexpected HostPin member')
            members[member.name] = archive.extractfile(member).read()
    expected = {'elf-dynamic.txt', 'elf-header.txt', 'elf-notes.txt', 'elf-versions.txt', 'embedded-temp-paths.txt',
                'hostpin-io', 'io.rs', 'ldd.txt', 'list.stderr', 'list.stdout', 'provenance.json'}
    require(set(members) == {'hostpin/' + n for n in expected} and sum(map(len, members.values())) < 16 << 20, 'HostPin archive member set differs')
    return members


def inspect_hostpin(raw):
    require(len(raw) == 3448805 and sha(raw) == PIN_SHA, 'HostPin FileAsset archive hash/length differs')
    files = archive_files(raw)
    binary = files['hostpin/hostpin-io']
    require(len(binary) == 8714656 and sha(binary) == PIN_BINARY_SHA and binary[:6] == b'\x7fELF\x02\x01' and int.from_bytes(binary[18:20], 'little') == 62, 'HostPin binary hash/ELF target differs')
    receipt = json.loads(files['hostpin/provenance.json'])
    require(receipt.get('native_source') == PIN_SOURCE and receipt.get('production_core') == CORE and receipt.get('focused_base') == SOURCE, 'HostPin source/core provenance differs')
    info = receipt['binary']
    require(info['sha256'] == PIN_BINARY_SHA and info['bytes'] == len(binary) and info['test'] == PIN_TEST and info['argv'] == PIN_ARGV and info['selected_tests'] == 1 and info['gpu_execution'] is False, 'HostPin frozen selection differs')
    versions = sorted({tuple(map(int, v.split(b'.'))) for v in re.findall(rb'GLIBC_([0-9]+\.[0-9]+(?:\.[0-9]+)?)\x00', binary)})
    require(versions and versions[-1] == (2, 28) and info['versions']['GLIBCXX'] == [] and info['versions']['CXXABI'] == [], 'HostPin requires newer glibc/C++ runtime')
    require({'.'.join(map(str, v)) for v in versions} == set(info['versions']['GLIBC']), 'HostPin ELF/provenance glibc versions differ')
    require(sha(files['hostpin/io.rs']) == receipt['source_sha256']['crates/tensorfs-core/tests/io.rs'], 'HostPin test source hash differs')
    require(files['hostpin/list.stdout'].decode().splitlines().count(PIN_TEST + ': test') == 1, 'HostPin selected test missing from export list')
    return {'sha256': PIN_SHA, 'bytes': len(raw), 'source': PIN_SOURCE, 'core_source': CORE,
            'binary_sha256': PIN_BINARY_SHA, 'binary_bytes': len(binary), 'glibc_max': '2.28',
            'argv': PIN_ARGV, 'timeout_s': 120}, files


def setup_names(requested):
    names = ['comfy-init', 'comfy-remote', 'comfy-fetch', 'comfy-checkout', 'comfy-archive']
    for name in requested:
        if name != HOSTPIN:
            names.append(name + '-purelib')
        if name in IDS:
            names.append(name + '-environment')
    return names
