"""Stdlib proof-contract tests. Synthetic GPU-shaped records are NOT device evidence."""
import argparse
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import tomllib
import unittest
import zipfile

import make_payload
import validate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
UUID = 'GPU-12345678-1234-1234-1234-123456789abc'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


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
                                      output=self.root / 'payload.py', kernel_script=None, kernel_source=None,
                                      kernel_sha256=None, kernel_manifest=None)

    def fixture(self, kernel=False):
        if kernel:
            script = self.root / 'kernel-input.py'
            script.write_text('# Contract fixture only, never executed as GPU proof.\n')
            manifest = self.root / 'kernel-input.json'
            write(manifest, {'proof': 'fixture-native-kernel', 'cases': ['decode'], 'argv': []})
            self.args.kernel_script, self.args.kernel_manifest = script, manifest
            self.args.kernel_source, self.args.kernel_sha256 = '4' * 40, validate.sha(script)
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
            (root / 'source' / name).write_text(source)
        (root / 'logs/comfy-setup.log').write_text('fixture source setup\n')
        with tarfile.open(root / 'source/comfy-upstream.tar', 'w', format=tarfile.PAX_FORMAT,
                          pax_headers={'comment': make_payload.COMFY}) as archive:
            data = b'# fixture source\n'
            entry = tarfile.TarInfo('comfy/__init__.py')
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        identity = {'ok': True, 'request_id': 'fixture-only', 'context_device': 'cuda:0', 'python_optimize': 0,
                    'cuda_visible_devices': '0', 'triton_interpret': '', 'runtime_source': '3' * 40,
                    'runtime': '0.28.2', 'native_sha256': spec['native']['sha256'],
                    'comfy_archive_sha256': validate.sha(root / 'source/comfy-upstream.tar')}
        write(root / 'identity.json', identity)
        parts = []
        for name in spec['requested']:
            child = {'ok': True, 'mode': name, 'python_optimize': 0, 'triton_interpret': '', 'script_completed': True,
                     'cublas_workspace_config': ':4096:8', 'cuda_visible_devices': '0', 'cuda_available': True,
                     'device_count': 1, 'cuda_initialized_before_probe': False, 'gpu_uuid': UUID,
                     'native_extension_sha256': spec['native']['extension_sha256'],
                     'physical': {'gpu': 0, 'uuid': UUID, 'name': 'fixture GPU', 'driver': '600.0'}}
            if name == 'kernel':
                child.update(torch_present=False, triton_present=False, torch_loaded=False, triton_loaded=False)
                value = {'proof': 'fixture-native-kernel', 'native_source': spec['native']['source'],
                         'wheel_sha256': spec['native']['sha256'], 'script_sha256': spec['kernel']['sha256'],
                         'device_count': 1, 'gpu_uuid': UUID,
                         'cases': [{'name': 'decode', 'executed': True, 'skipped': False,
                                    'kernel': True, 'device': 'cuda:0', 'max_abs_error': 0.0}]}
            else:
                value = validate.read(HERE / 'fixtures' / f'{name}-cpu.json')['evidence']
                value.update(proof=f'public-tensorfs-{name}-unet-cuda', device='cuda:0', gpu_origin_writer=True,
                             device_info={'type': 'cuda', 'uuid': UUID}, torch='2.14.0+cu130', torch_cuda='13.0',
                             native_source=spec['native']['source'], wheel_sha256=spec['native']['sha256'])
            write(root / f'reports/{name}-identity.json', child)
            self.result(root, name, value)
            parts.append({'name': name, 'rc': 0, 'ok': True})
        write(root / 'verdict.json', {'ok': True, 'requested': spec['requested'], 'unrequested': spec['unrequested'], 'parts': parts})
        self.inventory(root)
        run = {'status': 'completed', 'machine': 'fixture-machine', 'measurements': {'execution': {'degree': 1,
               'gpus': [{'gpu': 0, 'uuid': UUID, 'pid': 123, 'start_us': 10, 'end_us': 20}]}}}
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
        run['measurements']['execution'] = {'degree': 0, 'gpus': [{'gpu': -1, 'uuid': ''}]}
        result = validate.validate(root, expected, run, machine, launch)
        self.assertEqual(result['physical']['gpu'], 0)
        self.assertEqual(result['physical']['public_measured_gpu_rows'], 0)
        self.assertEqual(result['context_device'], 'cpu')

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
        for cases in [[], [{**value['cases'][0], 'skipped': True}], [{**value['cases'][0], 'kernel': False}]]:
            self.result(root, 'kernel', {**value, 'cases': cases})
            self.inventory(root)
            with self.assertRaises(ValueError):
                validate.validate(root, expected, run, machine, launch, require_kernel=True)

    def test_unrequested_kernel_is_not_a_kernel_pass(self):
        with self.assertRaisesRegex(ValueError, 'not requested'):
            validate.validate(*self.fixture(), require_kernel=True)

    def test_bad_freeze_hash_and_pre_fork_source_are_refused(self):
        self.args.native_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            make_payload.generate(self.args)
        self.args.native_source = '88a49ac8e634035301a3314088a856d214e55db8'
        with self.assertRaisesRegex(ValueError, 'pre-GPU/fork-fix'):
            make_payload.generate(self.args)


if __name__ == '__main__':
    unittest.main()
