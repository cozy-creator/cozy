"""Stdlib proof-contract tests. Synthetic GPU-shaped records are NOT device evidence."""
import argparse
import ast
import base64
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
PROVIDER_SOURCE = '7778b76d3e8eb0e9140cbf5fdca7eb487224cfd6'
KERNEL_SOURCE = 'bceb259ac8f4229ea37ef512618d485880680662'
COMBINED_SOURCE = 'def5e7a31c91ce1349c247439e802bf5163d1d8f'
UUID = 'GPU-12345678-1234-1234-1234-123456789abc'
RESIDENT_SOURCE = 'def5e7a31c91ce1349c247439e802bf5163d1d8f'


def resident_report(files, native):
    """Synthetic receipt from corpus metadata; never import or run the subject."""
    corpus = json.loads(files['crates/tensorfs/conformance/vectors/cases/comfy.json'])['cases']
    positive = [r for r in corpus if not r.get('refuse')]
    derived = dict(next(r for r in positive if r['key'] == 'w4a8-correction-g48-bf16'))
    derived['key'] += '-exact-f16-correction'
    records = [{'key': r['key'], 'result': 'passed', 'kind': 'golden',
                'bytes': len(base64.b64decode(r['expect']))} for r in [*positive, derived]]
    records += [{'key': r['key'], 'result': 'passed', 'kind': 'expected-refusal', 'family': 'decode',
                 'value_error_output_unspecified': r['key'] == 's-rel-nan'} for r in corpus if r.get('refuse')]
    records.append({'key': 'zero-output-pending-codebook', 'result': 'passed', 'kind': 'completion-control',
                    'producer_complete_at_return': True, 'output_complete_at_return': True,
                    'codebook_sha256': '58dda328598e2f7fe472621bfc54935aaa354d1a6ebcaf9562cd743fd575eb19'})
    return {'mode': 'gpu', 'passed': 34, 'failed': 0, 'skipped': 0, 'original_positive': 18,
            'derived_f16_positive': 1, 'expected_refusal': 14, 'completion_control': 1, 'raw_calls': 41,
            'cases': records, 'metadata_cast_dtypes': ['BF16', 'F16', 'F32', 'F8_E4M3'],
            'stream_control': {'producer_pending_before': True, 'output_pending_before': True,
                               'producer_complete_at_return': True, 'output_complete_at_return': True,
                               'source_stream': 123456, 'output_stream': 654321},
            'storage_controls': {'byte_offset': 8, 'managed_deletions': 3, 'cuda_managed_deletions': 1,
                                 'mixed_cpu_managed_deletions': 2, 'actual_cuda_owner_released': True,
                                 'used_alias_refused': True, 'overlap_family': 'invalid',
                                 'overlap_output_unchanged': True, 'mixed_host_cuda_family': 'unsupported',
                                 'mixed_cuda_host_family': 'unsupported', 'mixed_outputs_unchanged': True},
            'corpus_sha256': '56758d0c82b7b0d173d8db25cd5362ad9db348839e5f4e22ce664784c0e56043',
            'tensorfs_version': native['version'], 'torch_version': '2.14.0+cu130', 'device_count': 1,
            'gpu_uuid': UUID, 'native_sha256': native['extension_sha256'],
            'script_sha256': '2e46411b9d17c2fac6ddbf226e7a0edb9750c8ebc2033fc5ddaf5aca0b0c86f7',
            'scope': 'resident GPU decode corpus/refusals and checked producer/output ordering; no full model or unknown CUDA-fault injection'}


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

    def fixture(self, kernel=False, native=False, resident=False):
        if kernel:
            self.args.include_kernel = True
        self.args.include_native_io = native
        if native:
            self.args.proof_source = self.args.native_source = PROVIDER_SOURCE
        if kernel:
            self.args.proof_source = self.args.native_source = KERNEL_SOURCE
        if kernel and native:
            self.args.proof_source = self.args.native_source = COMBINED_SOURCE
        self.args.include_resident_decode = resident
        if resident:
            self.args.proof_source = self.args.native_source = RESIDENT_SOURCE
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
                        if case in ('pending-consumer-b', 'pending-ptds-consumer', 'pending-legacy-consumer'):
                            consumer = {'fill_ready_before_consume': True, 'pending_after_drop': True,
                                        'source_retained_while_pending': True, 'foreign_python_owner_collected': True,
                                        'drop_seconds': 0.01, 'heartbeat_ticks': 3, 'copied_bytes': 65536,
                                        'copied_sha256': native_io.PROVIDER_DATA_SHA256}
                            if case == 'pending-consumer-b':
                                consumer.update(bind_stream=3, consumer_stream=4, distinct_nondefault=True)
                            elif case == 'pending-ptds-consumer':
                                consumer.update(scope='final-release-custody', bind_stream=2, consumer_stream=2,
                                                bind_thread=100, consumer_thread=101, distinct_threads=True,
                                                bind_thread_alive_while_pending=True, bind_thread_source_handles_released=True,
                                                ptds=True, pending_readiness_device_proof=False)
                            else:
                                consumer.update(scope='final-release-custody', bind_stream=3, consumer_stream=1,
                                                torch_consumer_stream=0, dlpack_stream=None, default_legacy=True,
                                                distinct_nondefault=False, pending_readiness_device_proof=False)
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
                         'counts': {'full': 227, 'odd_windows': 227, 'partition_windows': 77,
                                    'windows': 304, 'total': 531, 'cases': 227, 'schemes': 42},
                         'partition_windows': {'nonzero-element': 18, 'group': 15, 'rotation': 7,
                                               'row': 18, 'matrix': 1, 'near-end-partial': 18},
                         'schemes': sorted({c['scheme'] for c in cases}), 'cases': cases,
                         'corpus_sha256': make_payload.KERNEL_CORPUS,
                         'routing': {'prepared': 'Kernel', 'host_only_exclusions': []},
                         'runtime': {'torch_installed': False, 'triton_installed': False,
                                     'torch_imported': False, 'triton_imported': False}}
            elif name == native_io.RESIDENT:
                value = resident_report(files, spec['native'])
            else:
                value = validate.read(HERE / 'fixtures' / f'{name}-cpu.json')['evidence']
                value.update(proof=f'public-tensorfs-{name}-unet-cuda', device='cuda:0', gpu_origin_writer=True,
                             device_info={'type': 'cuda', 'uuid': UUID}, torch='2.14.0+cu130', torch_cuda='13.0',
                             native_source=spec['native']['source'], wheel_sha256=spec['native']['sha256'],
                             script_sha256=spec['consumers'][name]['sha256'])
            write(root / f'reports/{name}-identity.json', child)
            self.result(root, name, value)
            part = {'name': name, **completed(KERNEL_SECONDS if name == 'kernel' else CONSUMER_SECONDS)}
            if name == native_io.RESIDENT:
                part['command'] = ['/fixture/python', '/fixture/source/child.py',
                                   '/fixture/reports/resident-decode-identity.json', name,
                                   '/fixture/source/' + expected['resident_decode']['file'], '--gpu', '--vectors',
                                   '/fixture/source/' + expected['resident_decode']['vectors_file']]
            parts.append(part)
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
        for cases in [[], [value['cases'][0]] * 227,
                      [{**value['cases'][0], 'full_passed': False}, *value['cases'][1:]],
                      [{**value['cases'][0], 'window': {'offset': 1, 'length': 13, 'passed': False}}, *value['cases'][1:]]]:
            self.result(root, 'kernel', {**value, 'cases': cases})
            self.inventory(root)
            self.bind(root, run)
            with self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, launch, require_kernel=True)

    def test_unrequested_kernel_is_not_a_kernel_pass(self):
        with self.assertRaisesRegex(ValueError, 'not requested'):
            validate.validate(*self.fixture(), require_kernel=True)

    def test_kernel_json_counts_ordinals_and_flags_keep_their_types(self):
        root, expected, run, machine, launch = self.fixture(kernel=True)
        original = validate.read(root / 'reports/kernel.json')
        mutations = [dict(counts={**original['counts'], 'full': 227.0}),
                     dict(device={'type': 'cuda', 'ordinal': False}),
                     dict(runtime={**original['runtime'], 'torch_installed': 0}),
                     dict(routing={'prepared': 'Host', 'host_only_exclusions': []})]
        for change in mutations:
            self.result(root, 'kernel', {**original, **change})
            self.inventory(root)
            self.bind(root, run)
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
        root, expected, run, machine, launch = self.fixture(kernel=True)
        self.assertEqual(expected['proof_source'], KERNEL_SOURCE)
        script = root / 'source' / make_payload.KERNEL_FILE
        self.assertTrue((script.parents[2] / 'tensorfs/conformance/vectors/cases').is_dir())
        self.assertNotIn('read_all(', script.read_text())
        self.assertEqual(expected['kernel']['sha256'], validate.sha(script))
        self.assertTrue(validate.validate(root, expected, run, machine, launch)['ok'])

    def test_current_kernel_corpus_boundaries_are_frozen(self):
        root, expected, run, machine, launch = self.fixture(kernel=True)
        value = validate.read(root / 'reports/kernel.json')
        self.assertEqual(expected['kernel']['corpus_sha256'],
                         'fa2f92b926ba1f0e1195027eaf4066ae0d72c20ca4119584ca8547d5e91167b1')
        self.assertEqual(expected['kernel']['sha256'],
                         '4ddbd5d6c24f59c38c48e710d4657612a3978c7f3eaa160a8db253b5264afd08')
        self.assertEqual(value['counts'], {'full': 227, 'odd_windows': 227, 'partition_windows': 77,
                                          'windows': 304, 'total': 531, 'cases': 227, 'schemes': 42})
        actual = {}
        for row in value['cases']:
            for window in row.get('partition_windows', []):
                actual[window['kind']] = actual.get(window['kind'], 0) + 1
                self.assertEqual(window['length'], 13)
                self.assertTrue(window['passed'])
        self.assertEqual(actual, {'nonzero-element': 18, 'group': 15, 'rotation': 7,
                                  'row': 18, 'matrix': 1, 'near-end-partial': 18})
        self.assertTrue(validate.validate(root, expected, run, machine, launch, require_kernel=True)['ok'])

    def test_combined_future_modes_use_one_actual_source_snapshot(self):
        root, expected, run, machine, launch = self.fixture(kernel=True, native=True)
        self.assertEqual(expected['proof_source'], COMBINED_SOURCE)
        self.assertEqual(expected['kernel']['source'], COMBINED_SOURCE)
        self.assertEqual(expected['native_io']['source'], COMBINED_SOURCE)
        self.assertEqual(expected['requested'], [name for name in native_io.ALL if name != native_io.RESIDENT])
        self.assertTrue(validate.validate(root, expected, run, machine, launch, require_kernel=True)['ok'])

    def test_frozen_matrix_and_rotation_boundaries_have_independent_offsets(self):
        root, *_ = self.fixture(kernel=True)
        cases = validate.read(root / 'reports/kernel.json')['cases']
        matrix = next(row for row in cases if row['case'].endswith('/w4a8-experts-bf16'))
        rotation = next(row for row in cases if row['case'].endswith('/w4a4-f32'))
        # Hand-calculated from the frozen [2,4,256] BF16 and [4,256] F32 geometry.
        self.assertEqual([(w['kind'], w['offset']) for w in matrix['partition_windows']],
                         [('nonzero-element', 7), ('group', 31), ('row', 511),
                          ('matrix', 2047), ('near-end-partial', 4083)])
        self.assertEqual([(w['kind'], w['offset']) for w in rotation['partition_windows']],
                         [('nonzero-element', 13), ('rotation', 253), ('row', 1021),
                          ('near-end-partial', 4083)])

    def test_wrong_or_missing_kernel_boundary_cannot_hide_behind_good_totals(self):
        root, expected, run, machine, launch = self.fixture(kernel=True)
        original = validate.read(root / 'reports/kernel.json')
        index = next(i for i, row in enumerate(original['cases']) if row.get('partition_windows'))
        shifted = original['cases'][index]['partition_windows'][0]['offset'] + 1
        for field, value in [('kind', 'wrong'), ('offset', shifted), ('length', 12), ('passed', False), ('offset', True)]:
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(original)
                changed['cases'][index]['partition_windows'][0][field] = value
                self.result(root, 'kernel', changed)
                self.inventory(root)
                self.bind(root, run)
                with self.assertRaisesRegex(ValueError, 'kernel missing/duplicate'):
                    validate.validate(root, expected, run, machine, launch, require_kernel=True)
        for remove in ('one', 'all', 'duplicate'):
            changed = copy.deepcopy(original)
            windows = changed['cases'][index]['partition_windows']
            if remove == 'one':
                windows.pop()
            elif remove == 'all':
                del changed['cases'][index]['partition_windows']
            else:
                windows.append(copy.deepcopy(windows[0]))
            self.result(root, 'kernel', changed)
            self.inventory(root)
            self.bind(root, run)
            with self.assertRaisesRegex(ValueError, 'kernel missing/duplicate'):
                validate.validate(root, expected, run, machine, launch, require_kernel=True)

    def test_kernel_old_counts_hash_ordinal_and_boundary_totals_are_refused(self):
        root, expected, run, machine, launch = self.fixture(kernel=True)
        original = validate.read(root / 'reports/kernel.json')
        changes = [{'counts': {'full': 209, 'windows': 209, 'total': 418, 'cases': 209, 'schemes': 39}},
                   {'device': {'type': 'cuda', 'ordinal': 1}},
                   {'corpus_sha256': '92a0f4f01f4490c2f7f8c391e5eb600f5e5b33d647eed48d91ff7c21e7af57d4'},
                   {'partition_windows': {**original['partition_windows'], 'group': 14}}]
        for change in changes:
            with self.subTest(change=change):
                self.result(root, 'kernel', {**original, **change})
                self.inventory(root)
                self.bind(root, run)
                with self.assertRaisesRegex(ValueError, 'kernel (CPU/wrong|missing/duplicate)'):
                    validate.validate(root, expected, run, machine, launch, require_kernel=True)

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

    def test_seven_provider_cases_and_source_hash_are_frozen(self):
        root, expected, run, machine, launch = self.fixture(native=True)
        contract = next(c for c in expected['native_io']['commands'] if c['id'] == 'provider-cleanup')
        self.assertEqual(expected['native_io']['source'], PROVIDER_SOURCE)
        self.assertEqual(contract['timeout_s'], 360)
        self.assertEqual(contract['case_timeout_s'], 45)
        self.assertEqual(contract['cases'], native_io.CASES)
        self.assertEqual(expected['sources'][contract['file']],
                         'e41175a7cfb425d6ace0c3772918370c2fc4e3c79042bd22a57b7b8753973872')
        self.assertEqual(expected['pending'], [])
        records = validate.read(root / 'reports/provider-cleanup.json')['records']
        self.assertEqual(len(records), 15)
        self.assertTrue(validate.validate(root, expected, run, machine, launch)['ok'])

    def test_legacy_none_stream_contract_rejects_missing_or_substituted_facts(self):
        root, expected, *_ = self.fixture(native=True)
        records = validate.read(root / 'reports/provider-cleanup.json')['records']
        legacy = next(i for i, row in enumerate(records)
                      if row.get('case') == 'pending-legacy-consumer' and row.get('result') == 'passed')
        self.assertIsNone(records[legacy]['consumer']['dlpack_stream'])
        mutations = [('dlpack_stream', 1), ('consumer_stream', 0), ('consumer_stream', True),
                     ('torch_consumer_stream', 1), ('torch_consumer_stream', False),
                     ('bind_stream', 2), ('default_legacy', False), ('distinct_nondefault', True),
                     ('pending_readiness_device_proof', True), ('scope', 'pending-readiness'),
                     ('source_retained_while_pending', False), ('copied_bytes', 65535),
                     ('copied_sha256', '0' * 64), ('heartbeat_ticks', 2), ('drop_seconds', 0.25)]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(records)
                changed[legacy]['consumer'][field] = value
                with self.assertRaises(ValueError):
                    native_io.parse('provider-cleanup', '\n'.join(json.dumps(r) for r in changed),
                                    expected['native'], UUID)
        for field in ('dlpack_stream', 'default_legacy', 'pending_readiness_device_proof'):
            with self.subTest(missing=field):
                changed = copy.deepcopy(records)
                del changed[legacy]['consumer'][field]
                with self.assertRaises(ValueError):
                    native_io.parse('provider-cleanup', '\n'.join(json.dumps(r) for r in changed),
                                    expected['native'], UUID)

    def test_old_six_case_receipt_and_timeout_do_not_qualify_seven_cases(self):
        root, expected, *_ = self.fixture(native=True)
        records = validate.read(root / 'reports/provider-cleanup.json')['records']
        old = copy.deepcopy(records[:12] + [records[-1]])
        old[-1]['cases'] = old[-1]['cases'][:6]
        with self.assertRaisesRegex(ValueError, 'missing/duplicate/extra'):
            native_io.parse('provider-cleanup', '\n'.join(json.dumps(r) for r in old),
                            expected['native'], UUID)
        raw = validate.read(HERE / 'native-io-contract.json')
        raw['commands'][1]['timeout_s'] = 300
        with self.assertRaisesRegex(ValueError, 'timeout differs'):
            native_io.contract(json.dumps(raw))

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

    def test_old_six_case_provider_source_is_refused(self):
        source = make_payload.git(REPO, 'show',
            PROOF_SOURCE + ':crates/tensorfs-py/scripts/cuda_varena_source.py').decode()
        with self.assertRaisesRegex(ValueError, 'provider mode inventory differs'):
            native_io.check_source('provider-cleanup', source)

    def test_resident_def5_source_layout_contract_and_all_modes(self):
        root, expected, run, machine, launch = self.fixture(kernel=True, native=True, resident=True)
        self.assertEqual(expected['proof_source'], RESIDENT_SOURCE)
        self.assertEqual(expected['requested'], native_io.ALL)
        contract = expected['resident_decode']
        self.assertEqual(contract['timeout_s'], 180)
        self.assertEqual(contract['argv'], ['--gpu', '--vectors', contract['vectors_file']])
        self.assertEqual(contract['counts']['raw_calls'], 41)
        self.assertEqual(len(contract['cases']), 34)
        self.assertEqual(len(contract['goldens']), 19)
        self.assertEqual({r['dtype'] for r in contract['goldens']}, {'F32', 'F16', 'BF16'})
        self.assertEqual(contract['goldens'][7]['sha256'], contract['goldens'][-1]['sha256'])
        self.assertEqual(Path(contract['file']).parent, Path(contract['helper_file']).parent)
        for path, digest in native_io.RESIDENT_FILES.items():
            self.assertEqual(validate.sha(root / 'source' / path), digest)
        self.assertTrue(validate.validate(root, expected, run, machine, launch,
                                          require_kernel=True, require_resident=True)['ok'])

    def test_resident_is_optional_and_must_be_requested_for_acceptance(self):
        with self.assertRaisesRegex(ValueError, 'resident proof was not requested'):
            validate.validate(*self.fixture(), require_resident=True)

    def test_resident_counts_rows_and_identity_keep_exact_types(self):
        root, expected, *_ = self.fixture(resident=True)
        value = validate.read(root / 'reports/resident-decode.json')
        contract, native = expected['resident_decode'], expected['native']
        changes = [{'mode': 'cpu'}, {'passed': 14}, {'passed': 34.0}, {'failed': False},
                   {'skipped': True}, {'raw_calls': 39}, {'raw_calls': 40}, {'device_count': True},
                   {'device_count': 2}, {'tensorfs_version': 'old'}, {'torch_version': '2.14.0'},
                   {'native_sha256': '0' * 64}, {'script_sha256': '0' * 64},
                   {'corpus_sha256': '0' * 64}, {'gpu_uuid': 'GPU-' + '0' * 32},
                   {'metadata_cast_dtypes': ['F32']}, {'unknown': True}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                native_io.parse_resident(json.dumps({**value, **change}), contract, native, '2.14.0+cu130', UUID)
        mutations = [('missing', lambda rows: rows.pop()),
                     ('duplicate', lambda rows: rows.append(dict(rows[0]))),
                     ('key', lambda rows: rows[0].update(key='other')),
                     ('kind', lambda rows: rows[0].update(kind='skipped')),
                     ('length', lambda rows: rows[0].update(bytes=4095)),
                     ('length-type', lambda rows: rows[0].update(bytes=4096.0)),
                     ('refusal', lambda rows: rows[19].update(family='invalid')),
                     ('value-error', lambda rows: next(r for r in rows if r['key'] == 's-rel-nan').update(value_error_output_unspecified=False)),
                     ('completion', lambda rows: rows[-1].update(producer_complete_at_return=False)),
                     ('codebook', lambda rows: rows[-1].update(codebook_sha256='0' * 64))]
        for label, mutate in mutations:
            changed = copy.deepcopy(value)
            mutate(changed['cases'])
            with self.subTest(label=label), self.assertRaises(ValueError):
                native_io.parse_resident(json.dumps(changed), contract, native, '2.14.0+cu130', UUID)

    def test_resident_completion_ownership_and_mixed_device_cannot_be_omitted(self):
        root, expected, *_ = self.fixture(resident=True)
        value = validate.read(root / 'reports/resident-decode.json')
        changes = [('stream_control', key, False) for key in ('producer_pending_before', 'output_pending_before',
                   'producer_complete_at_return', 'output_complete_at_return')]
        changes += [('stream_control', 'source_stream', v) for v in (True, 1, 2, 654321)]
        changes += [('stream_control', 'output_stream', 123456.0)]
        changes += [('storage_controls', key, replacement) for key, replacement in [
            ('byte_offset', 0), ('managed_deletions', 2), ('cuda_managed_deletions', True),
            ('mixed_cpu_managed_deletions', 0), ('actual_cuda_owner_released', False),
            ('used_alias_refused', False), ('overlap_family', 'decode'), ('overlap_output_unchanged', False),
            ('mixed_host_cuda_family', 'invalid'), ('mixed_cuda_host_family', 'invalid'),
            ('mixed_outputs_unchanged', False)]]
        for group, key, replacement in changes:
            changed = copy.deepcopy(value)
            changed[group][key] = replacement
            with self.subTest(group=group, key=key, replacement=replacement), self.assertRaises(ValueError):
                native_io.parse_resident(json.dumps(changed), expected['resident_decode'], expected['native'], '2.14.0+cu130', UUID)
        for group in ('stream_control', 'storage_controls'):
            changed = copy.deepcopy(value)
            changed.pop(group)
            with self.subTest(missing=group), self.assertRaises(ValueError):
                native_io.parse_resident(json.dumps(changed), expected['resident_decode'], expected['native'], '2.14.0+cu130', UUID)

    def test_resident_stdout_rejects_duplicate_nonfinite_and_extra_records(self):
        root, expected, *_ = self.fixture(resident=True)
        raw = (root / 'logs/resident-decode.stdout').read_text()
        for text in ('', 'null', '{"mode":"cpu","passed":14}', raw + raw, 'ignored\n' + raw,
                     raw.replace('"raw_calls": 41', '"raw_calls": NaN'),
                     raw.replace('"mode": "gpu"', '"mode": "cpu", "mode": "gpu"')):
            with self.subTest(text=text[:60]), self.assertRaises(ValueError):
                native_io.parse_resident(text, expected['resident_decode'], expected['native'], '2.14.0+cu130', UUID)

    def test_resident_source_drift_refused_before_payload_output(self):
        root, expected, *_ = self.fixture(resident=True)
        files = {p: (root / 'source' / p).read_text() for p in native_io.RESIDENT_FILES}
        for path in files:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'resident source/corpus differs'):
                native_io.resident_contract({**files, path: files[path] + '\n'})
        self.args.output = self.root / 'second-payload.py'
        self.args.proof_source = self.args.native_source = PROVIDER_SOURCE
        with self.assertRaisesRegex(ValueError, 'resident source/corpus unavailable'):
            make_payload.generate(self.args)
        self.assertFalse(self.args.output.exists())

    def test_resident_real_stdout_and_invocation_required_after_tree_rebind(self):
        root, expected, run, machine, launch = self.fixture(resident=True)
        original = validate.read(root / 'reports/resident-decode.json')
        changed = {**original, 'raw_calls': 39}
        self.result(root, native_io.RESIDENT, changed)
        self.inventory(root)
        self.bind(root, run)
        with self.assertRaisesRegex(ValueError, 'resident counts/cases/artifact differ'):
            validate.validate(root, expected, run, machine, launch)
        self.result(root, native_io.RESIDENT, original)
        verdict = validate.read(root / 'verdict.json')
        part = next(p for p in verdict['parts'] if p['name'] == native_io.RESIDENT)
        correct = copy.deepcopy(part)
        changes = [{'command': [*part['command'][:5], '--cpu', *part['command'][6:]]},
                   {'command': [*part['command'][:-1], '/other/comfy.json']},
                   {'timed_out': True}, {'rc': 1}, {'reaped': False}, {'lingering_descendants': True}]
        for change in changes:
            part.clear()
            part.update({**correct, **change})
            write(root / 'verdict.json', verdict)
            self.inventory(root)
            self.bind(root, run)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, launch)


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
