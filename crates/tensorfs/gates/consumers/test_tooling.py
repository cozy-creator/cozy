"""Stdlib proof-contract tests. Synthetic GPU-shaped records are NOT device evidence."""
import argparse
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import tomllib
import unittest
from unittest import mock
import zipfile

import make_payload
import validate
from owned_child import CONSUMER_SECONDS, KERNEL_SECONDS, SETUP_SECONDS, run_owned

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
UUID = 'GPU-12345678-1234-1234-1234-123456789abc'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def completed(limit):
    return {'rc': 0, 'ok': True, 'timed_out': False, 'reaped': True,
            'lingering_descendants': False, 'signals': [], 'timeout_s': limit,
            'elapsed_s': 0.1, 'pid': 123, 'pgid': 123}


class Contract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.native = self.root / 'tensorfs-0.11.0-cp312-abi3-linux_x86_64.whl'
        self.runtime = self.root / 'cozy_runtime-0.28.2-py3-none-any.whl'
        for path, name, version in [(self.native, 'tensorfs', '0.11.0'), (self.runtime, 'cozy-runtime', '0.28.2')]:
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr(f'{name.replace("-", "_")}-{version}.dist-info/METADATA',
                                 f'Name: {name}\nVersion: {version}\nRequires-Python: >=3.12\n')
                if name == 'tensorfs':
                    archive.writestr('tensorfs/_ext.abi3.so', b'unit-fixture-not-a-native-binary')
                else:
                    archive.writestr('cozy_runtime/_build_provenance.py', 'COMMIT = ' + repr('3' * 40))
        self.args = argparse.Namespace(creator_repo=REPO, native_wheel=self.native, native_source='2' * 40,
                                      native_sha256=validate.sha(self.native), runtime_wheel=self.runtime,
                                      runtime_source='3' * 40, runtime_sha256=validate.sha(self.runtime),
                                      output=self.root / 'payload.py', include_kernel=False,
                                      include_native_io=False, hostpin_archive=None)

    def fixture(self, kernel=False, native=False, hostpin=False):
        if kernel:
            self.args.include_kernel = True
        if native or hostpin:
            self.args.include_native_io = True
            self.args.native_source = validate.native_io.CORE
        if hostpin:
            self.args.hostpin_archive = self.root / 'hostpin-linux-x86_64.tar.gz'
            self.args.hostpin_archive.write_bytes(b'unit shape fixture; not an executable artifact')
            pin = validate.native_io
            source = make_payload.git(REPO, 'show', pin.PIN_SOURCE + ':crates/tensorfs-core/tests/io.rs')
            provenance = {'native_source': pin.PIN_SOURCE, 'production_core': pin.CORE,
                          'binary': {'sha256': pin.PIN_BINARY_SHA},
                          'source_sha256': {'crates/tensorfs-core/tests/io.rs': hashlib.sha256(source).hexdigest()}}
            info = {'sha256': pin.PIN_SHA, 'bytes': 3448805, 'source': pin.PIN_SOURCE, 'core_source': pin.CORE,
                    'binary_sha256': pin.PIN_BINARY_SHA, 'binary_bytes': 8714656, 'glibc_max': '2.28',
                    'argv': pin.PIN_ARGV, 'timeout_s': 120}
            contents = {'hostpin/io.rs': source, 'hostpin/provenance.json': json.dumps(provenance).encode()}
            # Only the offline metadata/validator shape is synthetic. Never execute this
            # fake artifact; the production inspector requires the real frozen archive hash.
            with mock.patch.object(pin, 'inspect_hostpin', return_value=(info, contents)):
                sidecar = make_payload.generate(self.args)
        else:
            sidecar = make_payload.generate(self.args)
        expected = validate.read(sidecar)
        spec = dict(expected)
        spec.pop('payload_sha256')
        tree = ast.parse(self.args.output.read_text())
        files = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'FILES'))
        root = self.root / 'tree'
        root.mkdir()
        write(root / 'manifest.json', spec)
        (root / 'source').mkdir()
        (root / 'logs').mkdir()
        (root / 'source/payload.py').write_bytes(self.args.output.read_bytes())
        for name, source in files.items():
            (root / 'source' / name).parent.mkdir(parents=True, exist_ok=True)
            (root / 'source' / name).write_text(source)
        operations = []
        for name in validate.native_io.setup_names(spec['requested']):
            (root / f'logs/{name}.stdout').write_text('fixture source setup\n')
            (root / f'logs/{name}.stderr').write_text('')
            operations.append({'name': name, **completed(SETUP_SECONDS)})
        with tarfile.open(root / 'source/comfy-upstream.tar', 'w', format=tarfile.PAX_FORMAT,
                          pax_headers={'comment': make_payload.COMFY}) as archive:
            data = b'# fixture source\n'
            entry = tarfile.TarInfo('comfy/__init__.py')
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        identity = {'ok': True, 'request_id': '987', 'context_device': 'cuda:0', 'python_optimize': 0,
                    'cuda_visible_devices': '0', 'triton_interpret': '', 'runtime_source': '3' * 40,
                    'runtime': '0.28.2', 'native_sha256': spec['native']['sha256'],
                    'comfy_archive_sha256': validate.sha(root / 'source/comfy-upstream.tar'), 'operations': operations,
                    'outer_torch_present': True}
        write(root / 'identity.json', identity)
        parts = []
        for name in spec['requested']:
            if name in validate.native_io.IDS or name == validate.native_io.HOSTPIN:
                pin = validate.native_io
                if name == pin.HOSTPIN:
                    write(root / 'reports/hostpin-artifact.json', spec['hostpin'])
                    stdout = ('running 1 test\ntest ' + pin.PIN_TEST + ' ... ' + pin.PIN_MARKER + '\nok\n'
                              'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 8 filtered out; finished in 0.50s\n')
                    argv = ['/work/hostpin-io', *pin.PIN_ARGV]
                else:
                    row = next(c for c in spec['native_io']['commands'] if c['id'] == name)
                    argv = ['/work/' + name + '/venv/bin/python', '-u', '/receipt/source/' + row['file'], *row['argv'][3:]]
                    clean = row['profile'] == 'clean'
                    environment = {'python': '3.12 fixture', 'executable': argv[0], 'python_optimize': 0, 'no_user_site': 1,
                                   'cuda_visible_devices': '0', 'cublas_workspace_config': ':4096:8', 'profile': row['profile'],
                                   'installed_native_extension_sha256': spec['native']['extension_sha256'],
                                   'torch': None if clean else '2.14.0+cu130', 'numpy': None if clean else '2.3.5', 'triton': None}
                    write(root / f'reports/{name}-environment.json', environment)
                    if name == 'read-engine':
                        stdout = pin.READ_MARKER + '\n'
                    elif name == 'fork-cleanup':
                        stdout = 'CUDA fork cleanup: CPU-only parent import, child allocation/capsule release recovered 33554449 bytes\n'
                    else:
                        records = []
                        for case in pin.CASES:
                            records += [{'case': case, 'tensorfs': spec['native']['version'], 'distribution': spec['native']['version'],
                                         'extension_sha256': spec['native']['extension_sha256'], 'device': 0, 'uuid': UUID},
                                        {'case': case, 'result': 'passed', 'cleanup': {'sets': 0, 'mapped': 0, 'idle': 0, 'slots': 0},
                                         'callback': {'chunks': 1}, 'h2d': [{'bytes': 65536, 'ns': 1}]}]
                        records.append({'result': 'passed', 'cases': pin.CASES, 'timeout_per_child': 45})
                        stdout = ''.join(json.dumps(r) + '\n' for r in records)
                value = pin.parse(name, stdout, spec['native'], UUID)
                self.result(root, name, value)
                (root / f'logs/{name}.stdout').write_text(stdout)
                parts.append({'name': name, 'command': argv, **completed(pin.TIMEOUT[name])})
                continue
            child = {'ok': True, 'mode': name, 'python_optimize': 0, 'triton_interpret': '', 'script_completed': True,
                     'cublas_workspace_config': ':4096:8', 'cuda_visible_devices': '0', 'cuda_available': True,
                     'device_count': 1, 'cuda_initialized_before_probe': False, 'gpu_uuid': UUID,
                     'native_extension_sha256': spec['native']['extension_sha256'],
                     'physical': {'gpu': 0, 'uuid': UUID, 'name': 'fixture GPU', 'driver': '600.0'}}
            if name == 'kernel':
                child.update(torch_present=False, triton_present=False, torch_loaded=False, triton_loaded=False)
                cases = make_payload.kernel_cases(files)
                value = {'schema': make_payload.KERNEL_SCHEMA, 'device': {'type': 'cuda', 'ordinal': 0},
                         'counts': {'full': 209, 'windows': 209, 'total': 418, 'cases': 209, 'schemes': 39},
                         'schemes': sorted({c['scheme'] for c in cases}), 'cases': cases,
                         'corpus_sha256': make_payload.KERNEL_CORPUS,
                         'routing': {'prepared': 'Kernel', 'host_only_exclusions': []},
                         'runtime': {'torch_installed': False, 'triton_installed': False,
                                     'torch_imported': False, 'triton_imported': False}}
            else:
                value = validate.read(HERE / 'fixtures' / f'{name}-cpu.json')['evidence']
                value.update(proof=f'public-tensorfs-{name}-unet-cuda', device='cuda:0', gpu_origin_writer=True,
                             device_info={'type': 'cuda', 'uuid': UUID}, torch='2.14.0+cu130', torch_cuda='13.0',
                             native_source=spec['native']['source'], wheel_sha256=spec['native']['sha256'])
            write(root / f'reports/{name}-identity.json', child)
            self.result(root, name, value)
            parts.append({'name': name, **completed(KERNEL_SECONDS if name == 'kernel' else CONSUMER_SECONDS)})
        write(root / 'verdict.json', {'ok': True, 'requested': spec['requested'], 'unrequested': spec['unrequested'], 'parts': parts})
        self.inventory(root)
        run = {'id': 'run-' + '1' * 24, 'number': 12, 'status': 'completed', 'machine': 'fixture-machine', 'measurements': {'execution': {'degree': 1,
               'gpus': [{'gpu': 0, 'uuid': UUID, 'arch': 'sm_90', 'pid': 123, 'start_us': 10, 'end_us': 20}]}}}
        self.bind(root, run)
        machine = {'name': 'fixture-machine', 'status': {'gpus': [{'index': 0, 'name': 'fixture GPU', 'driver': '600.0'}]}}
        launch = {'payload_sha256': expected['payload_sha256'], 'argv': ['cozy', 'run', 'payload.py', '--gpus', '1', '--on', 'fixture-machine']}
        return root, expected, run, machine, launch

    def result(self, root, name, value):
        write(root / f'reports/{name}.json', value)
        (root / f'logs/{name}.stdout').write_text(json.dumps(value) + '\n')
        (root / f'logs/{name}.stderr').write_text('')

    def inventory(self, root):
        write(root / 'files.json', {str(p.relative_to(root)): validate.sha(p) for p in root.rglob('*')
                                  if p.is_file() and p != root / 'files.json'})

    def bind(self, root, run):
        document = validate.tree_document(root)
        run['output'] = [{'id': f"{run['number']}/value", 'name': 'value', 'index': 0, 'rev': 1,
                          'status': 'completed', 'media_type': 'application/vnd.cozy.tree-manifest',
                          'length': len(document), 'sha256': 'sha256:' + hashlib.sha256(document).hexdigest()}]

    def test_synthetic_contract_only_positive_and_derived_metadata(self):
        root, expected, run, machine, launch = self.fixture()
        self.assertTrue(validate.validate(root, expected, run, machine, launch)['ok'])
        raw = self.args.output.read_text()
        metadata = raw.split('# /// script\n')[1].split('# ///\n')[0]
        metadata = tomllib.loads('\n'.join(line[2:] for line in metadata.splitlines()))
        self.assertIn('cozy-runtime==0.28.2', metadata['dependencies'])
        self.assertNotIn('cozy-runtime[media]==0.30.0', metadata['dependencies'])
        self.assertEqual(set(metadata['tool']['uv']['sources']), {'cozy-runtime', 'tensorfs'})
        self.assertLess(len(raw.encode()), 32 << 20)

    def test_real_cpu_receipts_are_negative_even_with_zero_error(self):
        root, expected, run, machine, launch = self.fixture()
        for name in ('diffusers', 'comfy'):
            original = validate.read(root / f'reports/{name}.json')
            self.result(root, name, validate.read(HERE / 'fixtures' / f'{name}-cpu.json')['evidence'])
            self.inventory(root)
            with self.assertRaisesRegex(ValueError, 'CUDA consumer'):
                validate.validate(root, expected, run, machine, launch)
            self.result(root, name, original)
        self.inventory(root)

    def test_false_positive_consumer_mutations(self):
        root, expected, run, machine, launch = self.fixture()
        original = validate.read(root / 'reports/diffusers.json')
        mutations = [('tensors', 207), ('bytes', 0), ('skipped', True), ('max_abs_error', float('nan')),
                     ('max_abs_error', float('inf')), ('max_abs_error', 1), ('gpu_origin_writer', False),
                     ('native_source', '0' * 40), ('wheel_sha256', '0' * 64), ('script_sha256', '0' * 64),
                     ('producer_released_before_forward', False), ('assigned_without_copy', False), ('torch_cuda', None)]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                self.result(root, 'diffusers', {**original, key: value})
                self.inventory(root)
                with self.assertRaises(ValueError):
                    validate.validate(root, expected, run, machine, launch)

    def test_missing_duplicate_or_failed_child(self):
        root, expected, run, machine, launch = self.fixture()
        original = validate.read(root / 'verdict.json')
        for parts in [original['parts'][:1], [original['parts'][0]] * 2,
                      [{**original['parts'][0], 'rc': 1}, original['parts'][1]]]:
            write(root / 'verdict.json', {**original, 'parts': parts})
            self.inventory(root)
            with self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, launch)

    def test_gpu_count_interpreter_and_environment_fail_closed(self):
        root, expected, run, machine, launch = self.fixture()
        original = validate.read(root / 'reports/diffusers-identity.json')
        for key, value in [('device_count', 2), ('triton_interpret', '1'), ('python_optimize', 1),
                           ('cublas_workspace_config', None), ('cuda_initialized_before_probe', True),
                           ('native_extension_sha256', '0' * 64), ('cuda_visible_devices', '0,1')]:
            write(root / 'reports/diffusers-identity.json', {**original, key: value})
            self.inventory(root)
            with self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, launch)

    def test_public_allocation_missing_or_different(self):
        root, expected, run, machine, launch = self.fixture()
        for rows in [[{'gpu': 1, 'uuid': UUID}],
                     [run['measurements']['execution']['gpus'][0], {'gpu': 1, 'uuid': 'GPU-' + 'f' * 32}]]:
            changed = copy.deepcopy(run)
            changed['measurements']['execution']['gpus'] = rows
            with self.assertRaises(ValueError):
                validate.validate(root, expected, changed, machine, launch)

    def test_cpu_parent_is_recorded_but_never_used_as_physical_proof(self):
        root, expected, run, machine, launch = self.fixture()
        identity = validate.read(root / 'identity.json')
        write(root / 'identity.json', {**identity, 'context_device': 'cpu'})
        self.inventory(root)
        self.bind(root, run)
        result = validate.validate(root, expected, run, machine, launch)
        self.assertEqual(result['physical']['gpu'], 0)
        self.assertEqual(result['physical']['public_measured_gpu_rows'], 1)
        self.assertEqual(result['context_device'], 'cpu')

    def test_missing_or_negative_public_gpu_identity_never_passes(self):
        root, expected, run, machine, launch = self.fixture()
        for rows in [[], [{'gpu': -1, 'uuid': ''}]]:
            run['measurements']['execution'] = {'degree': 0, 'gpus': rows}
            with self.assertRaisesRegex(ValueError, 'positive GPU execution'):
                validate.validate(root, expected, run, machine, launch)

    def test_swapped_public_run_or_receipt_is_refused(self):
        root, expected, run, machine, launch = self.fixture()
        for field, value in [('sha256', 'sha256:' + 'f' * 64), ('length', 0), ('id', '99/value')]:
            changed = copy.deepcopy(run)
            changed['output'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'public.*Tree|Tree.*public'):
                validate.validate(root, expected, changed, machine, launch)
        # Same sources and same machine, but a different actual worker invocation's bytes.
        identity = validate.read(root / 'identity.json')
        write(root / 'identity.json', {**identity, 'request_id': '988'})
        self.inventory(root)
        with self.assertRaisesRegex(ValueError, 'public run Tree digest'):
            validate.validate(root, expected, run, machine, launch)

    def test_timeout_or_failed_reap_cannot_pass_with_good_stdout(self):
        root, expected, run, machine, launch = self.fixture()
        original = validate.read(root / 'verdict.json')
        for field, value in [('timed_out', True), ('reaped', False), ('lingering_descendants', True)]:
            changed = copy.deepcopy(original)
            changed['parts'][0][field] = value
            write(root / 'verdict.json', changed)
            self.inventory(root)
            with self.assertRaisesRegex(ValueError, 'timed out/not reaped'):
                validate.validate(root, expected, run, machine, launch)

    def test_missing_or_wrong_public_one_gpu_launch(self):
        root, expected, run, machine, launch = self.fixture()
        for argv in [[], ['cozy', 'run', 'payload.py', '--gpus'],
                     ['cozy', 'run', 'payload.py', '--gpus', '2', '--on', 'fixture-machine']]:
            with self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, {**launch, 'argv': argv})

    def test_runtime_wheel_source_stamp_is_required(self):
        self.args.runtime_source = '4' * 40
        with self.assertRaisesRegex(ValueError, 'COMMIT differs'):
            make_payload.generate(self.args)

    def test_tampering_and_extra_checkpoint_bytes(self):
        root, expected, run, machine, launch = self.fixture()
        path = root / 'source/child.py'
        path.write_text(path.read_text() + '\n# changed\n')
        self.inventory(root)
        with self.assertRaisesRegex(ValueError, 'source hash'):
            validate.validate(root, expected, run, machine, launch)
        (root / 'weights.safetensors').write_bytes(b'not a receipt')
        self.inventory(root)
        with self.assertRaisesRegex(ValueError, 'unexpected/missing'):
            validate.validate(root, expected, run, machine, launch)

    def test_kernel_is_explicit_and_has_clean_child_case_set(self):
        root, expected, run, machine, launch = self.fixture(kernel=True)
        self.assertTrue(validate.validate(root, expected, run, machine, launch, require_kernel=True)['ok'])
        value = validate.read(root / 'reports/kernel.json')
        for cases in [[], [value['cases'][0]] * 209,
                      [{**value['cases'][0], 'full_passed': False}, *value['cases'][1:]],
                      [{**value['cases'][0], 'window': {'offset': 1, 'length': 13, 'passed': False}}, *value['cases'][1:]]]:
            self.result(root, 'kernel', {**value, 'cases': cases})
            self.inventory(root)
            with self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, launch, require_kernel=True)

    def test_unrequested_kernel_is_not_a_kernel_pass(self):
        with self.assertRaisesRegex(ValueError, 'not requested'):
            validate.validate(*self.fixture(), require_kernel=True)

    def test_kernel_json_counts_ordinals_and_flags_keep_their_types(self):
        root, expected, run, machine, launch = self.fixture(kernel=True)
        original = validate.read(root / 'reports/kernel.json')
        mutations = [dict(counts={**original['counts'], 'full': 209.0}),
                     dict(device={'type': 'cuda', 'ordinal': False}),
                     dict(runtime={**original['runtime'], 'torch_installed': 0}),
                     dict(routing={'prepared': 'Host', 'host_only_exclusions': []})]
        for change in mutations:
            self.result(root, 'kernel', {**original, **change})
            self.inventory(root)
            with self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, launch, require_kernel=True)

    def test_bad_freeze_hash_and_pre_fork_source_are_refused(self):
        self.args.native_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            make_payload.generate(self.args)
        self.args.native_source = '88a49ac8e634035301a3314088a856d214e55db8'
        with self.assertRaisesRegex(ValueError, 'pre-GPU/fork-fix'):
            make_payload.generate(self.args)

    def test_native_io_matrix_and_optional_hostpin_shape(self):
        root, expected, run, machine, launch = self.fixture(native=True, hostpin=True)
        result = validate.validate(root, expected, run, machine, launch)
        self.assertTrue(result['ok'])
        self.assertEqual(expected['requested'], ['diffusers', 'comfy', *validate.native_io.IDS, validate.native_io.HOSTPIN])
        self.assertEqual(expected['unrequested'], ['kernel', 'native-full-element-windows'])
        self.assertIn('hostpin_bundle: FileAsset', self.args.output.read_text())
        self.assertNotIn('full_gpu_arrays', validate.read(root / 'reports/read-engine.json'))
        changed = copy.deepcopy(expected)
        changed['requested'].remove('read-engine')
        with self.assertRaisesRegex(ValueError, 'partial native I/O batch'):
            validate.validate(root, changed, run, machine, launch)
        changed = copy.deepcopy(expected)
        changed['requested'].append('native-full-element-windows')
        with self.assertRaisesRegex(ValueError, 'extra native vectors are out of scope'):
            validate.validate(root, changed, run, machine, launch)

    def test_native_io_direct_interpreter_profile_and_partial_process_fail(self):
        root, expected, run, machine, launch = self.fixture(native=True)
        self.assertTrue(validate.validate(root, expected, run, machine, launch)['ok'])
        original = validate.read(root / 'verdict.json')
        for name, field, value in [('read-engine', 'command', ['python', '-c', 'import torch; print("fallback")']),
                                   ('provider-cleanup', 'timeout_s', 120), ('fork-cleanup', 'timed_out', True),
                                   ('provider-cleanup', 'rc', 1), ('fork-cleanup', 'reaped', False)]:
            changed = copy.deepcopy(original)
            next(p for p in changed['parts'] if p['name'] == name)[field] = value
            write(root / 'verdict.json', changed)
            self.inventory(root)
            with self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, launch)
        write(root / 'verdict.json', original)
        environment = validate.read(root / 'reports/fork-cleanup-environment.json')
        write(root / 'reports/fork-cleanup-environment.json', {**environment, 'torch': '2.14.0'})
        self.inventory(root)
        with self.assertRaisesRegex(ValueError, 'clean profile'):
            validate.validate(root, expected, run, machine, launch)

    def test_native_text_markers_do_not_allow_partial_skip_or_invented_counts(self):
        pin = validate.native_io
        native = {'extension_sha256': 'f' * 64, 'version': '0.11.0'}
        for text in ['', pin.READ_MARKER + '\n' + pin.READ_MARKER, 'SKIP\n' + pin.READ_MARKER,
                     '{"full_gpu_arrays":4}\n' + pin.READ_MARKER]:
            with self.assertRaises(ValueError):
                pin.parse('read-engine', text, native, UUID)
        for text in ['', 'CUDA fork cleanup: CPU-only parent import, child allocation/capsule release recovered 33554448 bytes',
                     'SKIP\nCUDA fork cleanup: CPU-only parent import, child allocation/capsule release recovered 33554449 bytes']:
            with self.assertRaises(ValueError):
                pin.parse('fork-cleanup', text, native, UUID)

    def test_provider_requires_every_real_identity_cleanup_and_h2d_case(self):
        root, expected, *_ = self.fixture(native=True)
        pin = validate.native_io
        original = [json.loads(line) for line in (root / 'logs/provider-cleanup.stdout').read_text().splitlines()]
        mutations = []
        for row, key, value in [(0, 'uuid', 'unavailable'), (0, 'device', False), (0, 'extension_sha256', '0' * 64),
                                (1, 'result', 'skipped'), (1, 'result', 'error'), (1, 'cleanup', {'sets': 1, 'mapped': 0, 'idle': 0, 'slots': 0}),
                                (1, 'callback', {'chunks': 0}), (1, 'h2d', [{'bytes': 65536, 'ns': 0}]),
                                (1, 'h2d', [{'bytes': 65535, 'ns': 1}]), (1, 'h2d', [{'bytes': 65536, 'ns': float('nan')}])]:
            rows = copy.deepcopy(original);rows[row][key] = value;mutations.append(rows)
        mutations += [original[:-1], original + [original[0]], [*original[:2], *original[:2], *original[4:]]]
        for rows in mutations:
            with self.assertRaises(ValueError):
                pin.parse('provider-cleanup', ''.join(json.dumps(r) + '\n' for r in rows), expected['native'], UUID)

    def test_hostpin_exact_single_ignored_selection_and_no_returned_binary(self):
        root, expected, run, machine, launch = self.fixture(hostpin=True)
        pin = validate.native_io
        original = (root / f'logs/{pin.HOSTPIN}.stdout').read_text()
        for text in [original.replace(pin.PIN_MARKER, ''), original + pin.PIN_MARKER,
                     original.replace('1 passed; 0 failed; 0 ignored', '0 passed; 0 failed; 1 ignored'),
                     original.replace('1 passed; 0 failed; 0 ignored', '1 passed; 1 failed; 0 ignored'),
                     original.replace('test ' + pin.PIN_TEST, 'test host_read_speed')]:
            with self.assertRaises(ValueError):
                pin.parse(pin.HOSTPIN, text, expected['native'], UUID)
        (root / 'hostpin-io').write_bytes(b'not a receipt')
        self.inventory(root)
        with self.assertRaisesRegex(ValueError, 'unexpected/missing'):
            validate.validate(root, expected, run, machine, launch)

    def test_hostpin_rejects_bad_hash_and_unsafe_archive_members(self):
        pin = validate.native_io
        with self.assertRaisesRegex(ValueError, 'archive hash/length'):
            pin.inspect_hostpin(b'wrong artifact')
        for name, kind in [('../hostpin-io', tarfile.REGTYPE), ('hostpin/link', tarfile.SYMTYPE)]:
            raw = io.BytesIO()
            with tarfile.open(fileobj=raw, mode='w:gz') as archive:
                entry = tarfile.TarInfo(name);entry.type = kind;entry.linkname = '/elsewhere'
                archive.addfile(entry)
            with self.assertRaises(ValueError):
                pin.archive_files(raw.getvalue())


class OwnedChild(unittest.TestCase):
    def test_success_keeps_complete_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as folder:
            out, err = Path(folder) / 'out', Path(folder) / 'err'
            row = run_owned([sys.executable, '-c', 'import sys; print("done"); print("error-log", file=sys.stderr)'],
                            out, err, timeout=5, term_seconds=0.05, kill_seconds=2)
            self.assertTrue(row['ok'])
            self.assertEqual(out.read_text(), 'done\n')
            self.assertEqual(err.read_text(), 'error-log\n')
            self.assertNotEqual(row['pgid'], os.getpgrp())

    def test_cleanup_hang_after_valid_json_is_killed_and_reaped(self):
        # A native destructor/cleanup hang after reporting success must still fail.
        code = ('import signal,time,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                'print(\'{"proof":"already-printed"}\',flush=True); '
                'print("cleanup stuck",file=sys.stderr,flush=True); time.sleep(60)')
        with tempfile.TemporaryDirectory() as folder:
            out, err = Path(folder) / 'out', Path(folder) / 'err'
            row = run_owned([sys.executable, '-c', code], out, err,
                            timeout=0.2, term_seconds=0.05, kill_seconds=2)
            self.assertFalse(row['ok'])
            self.assertTrue(row['timed_out'] and row['reaped'])
            self.assertEqual(row['signals'], ['SIGTERM', 'SIGKILL'])
            self.assertEqual(json.loads(out.read_text())['proof'], 'already-printed')
            self.assertIn('cleanup stuck', err.read_text())
            self.assertLess(row['elapsed_s'], 3)

    def test_timeout_targets_only_owned_group_including_grandchild(self):
        code = ('import subprocess,sys,time; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                'print(p.pid,flush=True); time.sleep(60)')
        with tempfile.TemporaryDirectory() as folder:
            out, err = Path(folder) / 'out', Path(folder) / 'err'
            row = run_owned([sys.executable, '-c', code], out, err,
                            timeout=0.2, term_seconds=0.1, kill_seconds=2)
            self.assertTrue(row['reaped'] and row['timed_out'])
            grandchild = int(out.read_text().strip())
            stat = Path(f'/proc/{grandchild}/stat')
            # Only the direct child is ours to reap; an orphan zombie is already dead.
            if stat.exists():
                self.assertEqual(stat.read_text().split(') ', 1)[1].split()[0], 'Z')
            self.assertNotEqual(row['pgid'], os.getpgrp())


if __name__ == '__main__':
    unittest.main()
