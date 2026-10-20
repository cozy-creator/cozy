"""Stdlib-only diagnostic-tool tests. Never import TensorFS or execute a probe."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).with_name('prove-cold-cuda.py')
spec = importlib.util.spec_from_file_location('cold_cuda_diagnostic', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Diagnostic(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        wheel = self.root / 'tensorfs.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('tensorfs-0.11.0.dist-info/METADATA', 'Name: tensorfs\nVersion: 0.11.0\n')
            archive.writestr('tensorfs/__init__.py', '# synthetic\n')
            archive.writestr('tensorfs/_ext.abi3.so', b'synthetic-not-executable')
        self.args = argparse.Namespace(wheel=wheel, wheel_sha256=probe.sha(wheel.read_bytes()),
                                       native_source='2' * 40, vectors=self.root / 'awq.json',
                                       out=self.root / 'output', timeout=45)
        repo = SCRIPT.parents[3]
        self.args.vectors.write_bytes(subprocess.check_output([
            'git', '-C', str(repo), 'show',
            '25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2:crates/tensorfs/conformance/vectors/cases/awq.json']))

    def test_real_first_awq_fixture_and_changed_bytes_refuse(self):
        row = probe.first_awq(self.args.vectors)
        self.assertEqual((row['key'], row['encoding']['scheme'], row['shape']),
                         ('autoawq-32x64-g32-f16', 'awq.int4', [32, 64]))
        self.args.vectors.write_bytes(self.args.vectors.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'exact current25'):
            probe.first_awq(self.args.vectors)

    def test_wheel_hash_and_exact_native_member(self):
        version, files, extension = probe.wheel_files(self.args.wheel, self.args.wheel_sha256)
        self.assertEqual(version, '0.11.0')
        self.assertEqual(extension, 'tensorfs/_ext.abi3.so')
        self.assertEqual(files[extension], probe.sha(b'synthetic-not-executable'))
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            probe.wheel_files(self.args.wheel, '0' * 64)

    def test_fresh_child_argv_has_faulthandler_and_exact_inputs(self):
        self.assertEqual(probe.STAGES, ('import', 'host-u8', 'cuda-u8', 'cuda-awq'))
        commands = [probe.command(self.args, stage) for stage in probe.STAGES]
        for stage, command in zip(probe.STAGES, commands):
            self.assertEqual(command[1:4], ['-u', '-X', 'faulthandler'])
            self.assertEqual(command[command.index('--stage') + 1], stage)
            self.assertIn(str(self.args.wheel.resolve()), command)
            self.assertIn(self.args.wheel_sha256, command)

    def test_coordinator_stops_after_first_failed_stage_and_preserves_logs(self):
        calls = []
        def fake_run(argv, **kw):
            stage = argv[argv.index('--stage') + 1]
            calls.append(stage)
            self.assertEqual(kw['env']['PYTHONFAULTHANDLER'], '1')
            self.assertNotIn('PYTHONPATH', kw['env'])
            point = 'stage-complete' if stage == 'import' else 'read-start'
            kw['stdout'].write(json.dumps({'stage': stage, 'point': point}).encode() + b'\n')
            kw['stderr'].write(b'synthetic diagnostic stderr\n')
            return subprocess.CompletedProcess(argv, 0 if stage == 'import' else -11)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': '/not-used'}), \
                patch.object(probe.subprocess, 'run', side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.coordinate(self.args), 1)
        self.assertEqual(calls, ['import', 'host-u8'])
        result = json.loads((self.args.out / 'result.json').read_text())
        self.assertFalse(result['ok'])
        self.assertEqual(result['stages'][-1]['rc'], -11)
        self.assertEqual(result['stages'][-1]['last_marker']['point'], 'read-start')
        self.assertEqual((self.args.out / 'host-u8.stderr').read_bytes(), b'synthetic diagnostic stderr\n')

    def test_timeout_or_partial_stdout_keeps_failure_receipt(self):
        def timeout(argv, **kw):
            kw['stdout'].write(b'{"stage":"import","point":')
            raise subprocess.TimeoutExpired(argv, 45)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                patch.object(probe.subprocess, 'run', side_effect=timeout), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.coordinate(self.args), 1)
        row = json.loads((self.args.out / 'result.json').read_text())['stages'][0]
        self.assertTrue(row['timed_out'])
        self.assertIsNone(row['rc'])
        self.assertTrue(row['unparsed_stdout'])


if __name__ == '__main__':
    unittest.main()
