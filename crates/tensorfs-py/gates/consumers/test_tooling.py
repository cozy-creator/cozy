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
import zipfile

import make_payload
import native_io
import validate
from owned_child import CONSUMER_SECONDS, KERNEL_SECONDS, SETUP_SECONDS, run_owned

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROOF_SOURCE = '25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2'
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
        self.runtime = self.root / f'cozy_runtime-0.28.2-{make_payload.RUNTIME_TAG}.whl'
        self.requirements = self.root / 'requirements.txt'
        self.requirements.write_text('\n'.join([
            'torch==2.14.0+cu130', 'diffusers==0.40.0', 'numpy==2.3.0', 'einops==0.8.0',
            'safetensors==0.8.0', 'Pillow==12.0.0', 'scipy==1.16.0', 'tqdm==4.67.0',
            'psutil==7.0.0', 'pyyaml==6.0.0', 'packaging==25.0',
            'comfy-kitchen==0.2.39', 'comfy-aimdo==0.5.6']) + '\n')
        # Metadata-only test input: never fetched/checked out as a real Git bundle.
        self.bundle = self.root / 'comfy.bundle'
        self.bundle.write_bytes(('# v2 git bundle\n' + make_payload.COMFY + ' refs/heads/proof\n\nPACK synthetic fixture').encode())
        for path, name, version in [(self.native, 'tensorfs', '0.11.0'), (self.runtime, 'cozy-runtime', '0.28.2')]:
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr(f'{name.replace("-", "_")}-{version}.dist-info/METADATA',
                                 f'Name: {name}\nVersion: {version}\nRequires-Python: >=3.12\n')
                if name == 'tensorfs':
                    archive.writestr('tensorfs/_ext.abi3.so', b'unit-fixture-not-a-native-binary')
                else:
                    archive.writestr('cozy_runtime/_build_provenance.py', 'COMMIT = ' + repr('3' * 40))
                    archive.writestr('cozy_runtime-0.28.2.dist-info/WHEEL',
                                     'Root-Is-Purelib: false\nTag: ' + make_payload.RUNTIME_TAG + '\n')
                    for module in sorted(make_payload.RUNTIME_KERNELS):
                        archive.writestr(module, b'synthetic module bytes, not a native binary: ' + module.encode())
        self.args = argparse.Namespace(creator_repo=REPO, proof_source=PROOF_SOURCE,
                                      native_wheel=self.native, native_source=PROOF_SOURCE,
                                      native_sha256=validate.sha(self.native), runtime_wheel=self.runtime,
                                      runtime_source='3' * 40, runtime_sha256=validate.sha(self.runtime),
                                      output=self.root / 'payload.py', include_kernel=False, include_native_io=False,
                                      require_native_mode=[], requirements=self.requirements,
                                      comfy_bundle=self.bundle, comfy_sha256=validate.sha(self.bundle))

    def fixture(self, kernel=False, native=False):
        if kernel:
            self.args.include_kernel = True
        self.args.include_native_io = native
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
        for name in native_io.setup_names(spec['requested']):
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
                    'runtime_kernel_sha256': spec['runtime']['kernel_sha256'],
                    'comfy_archive_sha256': validate.sha(root / 'source/comfy-upstream.tar'), 'operations': operations,
                    'comfy_bundle_sha256': spec['comfy_bundle']['sha256'],
                    'dependencies': spec['dependencies']['versions'],
                    'outer_torch_present': True}
        write(root / 'identity.json', identity)
        parts = []
        for name in spec['requested']:
            if name in native_io.IDS:
                row = next(r for r in spec['native_io']['commands'] if r['id'] == name)
                executable = f'/fixture/{name}/python'
                clean = row['profile'] == 'clean'
                observed = {'executable': executable, 'python_optimize': 0, 'no_user_site': 1,
                            'cuda_visible_devices': '0', 'cublas_workspace_config': ':4096:8',
                            'profile': row['profile'], 'installed_native_extension_sha256': spec['native']['extension_sha256'],
                            'torch': None if clean else '2.14.0+cu130', 'triton': None, 'numpy': None if clean else '2.3.0'}
                write(root / f'reports/{name}-environment.json', observed)
                if name == 'read-engine':
                    stdout = native_io.READ_MARKER + '\n'
                elif name == 'fork-cleanup':
                    stdout = 'CUDA fork cleanup: CPU-only parent import, child allocation/capsule release recovered 33554449 bytes\n'
                else:
                    records = []
                    for case in native_io.CASES:
                        records += [{'case': case, 'device': 0, 'uuid': UUID, 'extension_sha256': spec['native']['extension_sha256'],
                                     'distribution': spec['native']['version'], 'tensorfs': spec['native']['version']},
                                    {'case': case, 'result': 'passed', 'cleanup': dict.fromkeys(('sets', 'mapped', 'idle', 'slots'), 0),
                                     'callback': {'chunks': 1}, 'h2d': [{'bytes': 65536, 'ns': 1}]}]
                        if case in ('pending-consumer-b', 'pending-ptds-consumer'):
                            consumer = {'fill_ready_before_consume': True, 'pending_after_drop': True,
                                        'source_retained_while_pending': True, 'foreign_python_owner_collected': True,
                                        'drop_seconds': 0.01, 'heartbeat_ticks': 3, 'copied_bytes': 65536,
                                        'copied_sha256': native_io.PROVIDER_DATA_SHA256}
                            if case == 'pending-consumer-b':
                                consumer.update(bind_stream=3, consumer_stream=4, distinct_nondefault=True)
                            else:
                                consumer.update(scope='final-release-custody', bind_stream=2, consumer_stream=2,
                                                bind_thread=100, consumer_thread=101, distinct_threads=True,
                                                bind_thread_alive_while_pending=True, bind_thread_source_handles_released=True,
                                                ptds=True, pending_readiness_device_proof=False)
                            records[-1]['consumer'] = consumer
                    records.append({'result': 'passed', 'cases': native_io.CASES, 'timeout_per_child': 45})
                    stdout = ''.join(json.dumps(r) + '\n' for r in records)
                write(root / f'reports/{name}.json', native_io.parse(name, stdout, spec['native'], UUID))
                (root / f'logs/{name}.stdout').write_text(stdout)
                (root / f'logs/{name}.stderr').write_text('')
                parts.append({'name': name, **completed(row['timeout_s']),
                              'command': [executable, '-u', '/fixture/source/' + row['file'], *row['argv']]})
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
                             native_source=spec['native']['source'], wheel_sha256=spec['native']['sha256'],
                             script_sha256=spec['consumers'][name]['sha256'])
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
        sources = metadata['tool']['uv']['sources']
        self.assertEqual(set(sources), {'cozy-runtime', 'tensorfs', 'torch'})
        self.assertEqual(sources['cozy-runtime'], {'path': self.runtime.name})
        self.assertEqual(sources['tensorfs'], {'path': self.native.name})
        self.assertEqual(sources['torch'], {'index': 'pytorch-cu130'})
        self.assertEqual(metadata['tool']['uv']['index'], [{
            'name': 'pytorch-cu130', 'url': 'https://download.pytorch.org/whl/cu130', 'explicit': True,
        }])
        self.assertIn('torch==2.14.0+cu130', metadata['dependencies'])
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
        self.args.proof_source = self.args.native_source
        with self.assertRaisesRegex(ValueError, 'pre-GPU/fork-fix'):
            make_payload.generate(self.args)

    def test_current_source_and_split_crate_layout_are_frozen(self):
        root, expected, run, machine, launch = self.fixture(kernel=True, native=True)
        self.assertEqual(expected['proof_source'], PROOF_SOURCE)
        self.assertEqual(expected['native_io']['source'], PROOF_SOURCE)
        script = root / 'source' / make_payload.KERNEL_FILE
        self.assertTrue((script.parents[2] / 'tensorfs/conformance/vectors/cases').is_dir())
        self.assertNotIn('read_all(', script.read_text())
        self.assertEqual(expected['kernel']['sha256'], validate.sha(script))
        self.assertTrue(validate.validate(root, expected, run, machine, launch)['ok'])

    def test_missing_mode_requirements_bundle_and_source_fail_before_writing(self):
        self.args.require_native_mode = ['future-ab']
        with self.assertRaisesRegex(ValueError, 'mode absent'):
            make_payload.generate(self.args)
        self.args.require_native_mode = []
        self.requirements.write_text('torch>=2.14\n')
        with self.assertRaisesRegex(ValueError, 'exact name==version'):
            make_payload.generate(self.args)
        self.assertFalse(self.args.output.exists())

    def test_native_io_actual_stdout_is_required(self):
        root, expected, run, machine, launch = self.fixture(native=True)
        self.assertTrue(validate.validate(root, expected, run, machine, launch)['ok'])
        (root / 'logs/fork-cleanup.stdout').write_text('skipped\n')
        self.inventory(root)
        with self.assertRaisesRegex(ValueError, 'fork recovery'):
            validate.validate(root, expected, run, machine, launch)

    def test_proof_source_must_be_the_wheel_producer(self):
        self.args.native_source = '2' * 40
        with self.assertRaisesRegex(ValueError, 'wheel producer'):
            make_payload.generate(self.args)

    def test_changed_bundle_fails_before_writing(self):
        self.args.comfy_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Comfy bundle hash'):
            make_payload.generate(self.args)

    def test_runtime_pure_wheel_or_missing_kernel_is_rejected(self):
        with zipfile.ZipFile(self.runtime) as source:
            original = {name: source.read(name) for name in source.namelist()}
        for pure in (True, False):
            files = dict(original)
            if pure:
                files['cozy_runtime-0.28.2.dist-info/WHEEL'] = b'Root-Is-Purelib: true\nTag: py3-none-any\n'
            else:
                del files['cozy_runtime/_kernels/_fa3.abi3.so']
            with zipfile.ZipFile(self.runtime, 'w') as target:
                for name, raw in files.items():
                    target.writestr(name, raw)
            self.args.runtime_sha256 = validate.sha(self.runtime)
            with self.assertRaisesRegex(ValueError, 'non-pure Linux|kernel module set'):
                make_payload.generate(self.args)
        self.assertFalse(self.args.output.exists())

    def test_torch_build_tag_is_required_and_exact(self):
        self.requirements.write_text(self.requirements.read_text().replace('torch==2.14.0+cu130', 'torch==2.14.0'))
        with self.assertRaisesRegex(ValueError, 'exact CUDA build tag'):
            make_payload.generate(self.args)
        self.requirements.write_text(self.requirements.read_text().replace('torch==2.14.0', 'torch==2.14.0+cu130'))
        root, expected, run, machine, launch = self.fixture()
        original = validate.read(root / 'identity.json')
        for version in ('2.14.0+cpu', '2.14.0+cu129', '2.14.0'):
            changed = copy.deepcopy(original)
            changed['dependencies']['torch'] = version
            write(root / 'identity.json', changed)
            self.inventory(root)
            with self.assertRaisesRegex(ValueError, 'installed dependency differs: torch'):
                validate.validate(root, expected, run, machine, launch)

    def test_other_cuda_index_build_is_refused_before_writing(self):
        self.requirements.write_text(self.requirements.read_text().replace('torch==2.14.0+cu130', 'torch==2.14.0+cu128'))
        with self.assertRaisesRegex(ValueError, r'exact CUDA build tag \+cu130'):
            make_payload.generate(self.args)
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.args.output.with_suffix('.manifest.json').exists())

    def test_installed_runtime_kernel_hashes_must_match(self):
        root, expected, run, machine, launch = self.fixture()
        self.assertEqual(set(expected['runtime']['kernel_sha256']), make_payload.RUNTIME_KERNELS)
        identity = validate.read(root / 'identity.json')
        identity['runtime_kernel_sha256']['cozy_runtime/_kernels/_gemm.abi3.so'] = '0' * 64
        write(root / 'identity.json', identity)
        self.inventory(root)
        with self.assertRaisesRegex(ValueError, 'Runtime native kernel hashes'):
            validate.validate(root, expected, run, machine, launch)

    def test_six_provider_cases_and_source_hash_are_frozen(self):
        root, expected, run, machine, launch = self.fixture(native=True)
        contract = next(c for c in expected['native_io']['commands'] if c['id'] == 'provider-cleanup')
        self.assertEqual(contract['timeout_s'], 300)
        self.assertEqual(contract['case_timeout_s'], 45)
        self.assertEqual(contract['cases'], native_io.CASES)
        self.assertEqual(expected['sources'][contract['file']],
                         'bc4f439e5770b6d6f1dacc7adf843a9864f1628962a80e37c9dc3231cb5f8797')
        self.assertEqual(expected['pending'], [])
        records = validate.read(root / 'reports/provider-cleanup.json')['records']
        self.assertEqual(len(records), 13)
        self.assertTrue(validate.validate(root, expected, run, machine, launch)['ok'])

    def test_pending_consumer_evidence_cannot_be_omitted_or_substituted(self):
        root, expected, *_ = self.fixture(native=True)
        records = validate.read(root / 'reports/provider-cleanup.json')['records']
        mutations = [(9, 'pending_after_drop', False), (9, 'consumer_stream', 3),
                     (9, 'copied_sha256', '0' * 64), (9, 'heartbeat_ticks', 2),
                     (11, 'pending_readiness_device_proof', True), (11, 'consumer_thread', 100),
                     (11, 'bind_thread_alive_while_pending', False), (11, 'bind_stream', False)]
        for index, key, value in mutations:
            with self.subTest(key=key):
                changed = copy.deepcopy(records)
                changed[index]['consumer'][key] = value
                with self.assertRaises(ValueError):
                    native_io.parse('provider-cleanup', '\n'.join(json.dumps(r) for r in changed),
                                    expected['native'], UUID)
        with self.assertRaisesRegex(ValueError, 'missing/duplicate/extra'):
            native_io.parse('provider-cleanup', '\n'.join(json.dumps(r) for r in records[:8] + [records[-1]]),
                            expected['native'], UUID)

    def test_old_four_case_provider_source_is_refused(self):
        source = make_payload.git(REPO, 'show',
            '85ab9f3d92736a640548a9c3da70e3f374368c55:crates/tensorfs-py/scripts/cuda_varena_source.py').decode()
        with self.assertRaisesRegex(ValueError, 'provider mode inventory differs'):
            native_io.check_source('provider-cleanup', source)


class OwnedChild(unittest.TestCase):
    def test_tree_document_orders_full_paths_before_prefix_components(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = ['source/crates/tensorfs/z.py', 'source/crates/tensorfs-py/a.py',
                     'source/crates/tensorfs0/a.py']
            for path in paths:
                file = root / path
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'x')
            expected = {'entries': [
                {'blob': {'length': 1, 'sha256': '2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881'},
                 'kind': 'file', 'path': 'source/crates/tensorfs-py/a.py'},
                {'blob': {'length': 1, 'sha256': '2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881'},
                 'kind': 'file', 'path': 'source/crates/tensorfs/z.py'},
                {'blob': {'length': 1, 'sha256': '2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881'},
                 'kind': 'file', 'path': 'source/crates/tensorfs0/a.py'},
            ]}
            self.assertEqual(validate.tree_document(root),
                             json.dumps(expected, sort_keys=True, separators=(',', ':')).encode())

    def test_worker_directories_preserve_existing_comfy_checkout(self):
        # Execute the worker's actual directory setup, without importing Runtime/Torch.
        tree = ast.parse((HERE / 'worker.py').read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and
                    ast.unparse(n.iter) == "SPEC['requested']")
        setup = [n for n in loop.body if
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'directory' for t in n.targets)) or
                 (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
                  isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and
                  n.value.func.value.id == 'directory' and n.value.func.attr == 'mkdir')]
        self.assertEqual(len(setup), 2)
        code = compile(ast.Module(body=setup, type_ignores=[]), 'worker-directory-setup', 'exec')
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            checkout = work / 'comfy'
            checkout.mkdir()
            make_payload.git(checkout, 'init', '--quiet')
            model = checkout / 'model.py'
            original = b'# retained upstream source\n'
            model.write_bytes(original)
            make_payload.git(checkout, 'add', 'model.py')
            model.unlink()
            make_payload.git(checkout, 'checkout-index', '--all')
            directories = []
            for name in native_io.ALL:
                scope = {'work': work, 'name': name}
                exec(code, scope)
                directory = scope['directory']
                self.assertTrue(directory.is_dir())
                self.assertNotEqual(directory, checkout)
                (directory / 'model.py').write_text(name)
                directories.append(directory)
            self.assertEqual(len(set(directories)), len(native_io.ALL))
            self.assertEqual([p.joinpath('model.py').read_text() for p in directories], native_io.ALL)
            self.assertEqual(model.read_bytes(), original)
            self.assertTrue((checkout / '.git').is_dir())
            make_payload.git(checkout, 'diff', '--exit-code')

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
