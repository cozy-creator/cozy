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
                                      output=self.root / 'payload.py', include_kernel=False)

    def fixture(self, kernel=False):
        if kernel:
            self.args.include_kernel = True
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
        for name in ['comfy-init', 'comfy-remote', 'comfy-fetch', 'comfy-checkout', 'comfy-archive'] + [n + '-purelib' for n in spec['requested']]:
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
