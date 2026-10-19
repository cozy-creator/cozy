"""Fresh subprocess bootstrap; consumer CUDA starts only after deterministic environment checks."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def uuid(value):
    return str(value).lower().removeprefix('gpu-').replace('-', '')


identity_file, mode, script, *argv = sys.argv[1:]
identity = {'mode': mode, 'python_optimize': sys.flags.optimize,
            'triton_interpret': os.environ.get('TRITON_INTERPRET', ''),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
try:
    assert sys.flags.optimize == 0, 'optimized Python disables proof assertions'
    assert identity['triton_interpret'] in ('', '0'), 'interpreter fallback is forbidden'
    assert identity['cublas_workspace_config'] == ':4096:8', 'deterministic matmul environment must precede CUDA init'
    assert identity['cuda_visible_devices'] and len(identity['cuda_visible_devices'].split(',')) == 1, 'exactly one sealed GPU is required'
    import tensorfs._ext
    identity['native_extension_sha256'] = hashlib.sha256(Path(tensorfs._ext.__file__).read_bytes()).hexdigest()
    assert identity['native_extension_sha256'] == os.environ['COZY_PROOF_NATIVE_EXTENSION_SHA256'], 'wrong imported native extension'
    if mode == 'kernel':
        identity['torch_present'] = importlib.util.find_spec('torch') is not None
        identity['triton_present'] = importlib.util.find_spec('triton') is not None
        assert not identity['torch_present'] and not identity['triton_present'], 'kernel child is TensorFS-only'
    else:
        import torch
        identity['cuda_initialized_before_probe'] = torch.cuda.is_initialized()
        assert identity['cuda_initialized_before_probe'] is False, 'CUDA initialized before the proof bootstrap'
        identity['cuda_available'] = torch.cuda.is_available()
        identity['device_count'] = torch.cuda.device_count()
        assert identity['cuda_available'] and identity['device_count'] == 1, 'exactly one real CUDA GPU is required'
        props = torch.cuda.get_device_properties(0)
        identity['gpu_uuid'] = str(props.uuid)
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,driver_version,name',
                                       '--format=csv,noheader,nounits'], text=True)
        rows = [line.split(',', 3) for line in raw.splitlines() if line.strip()]
        matches = [r for r in rows if len(r) == 4 and uuid(r[1].strip()) == uuid(identity['gpu_uuid'])]
        assert len(matches) == 1, 'Torch UUID has no unique physical NVML mapping'
        row = [v.strip() for v in matches[0]]
        identity['physical'] = {'gpu': int(row[0]), 'uuid': row[1], 'driver': row[2], 'name': row[3]}
        visible = identity['cuda_visible_devices']
        assert (int(visible) == int(row[0])) if visible.isdecimal() else uuid(visible) == uuid(row[1]), 'sealed visibility differs from physical GPU'
    identity['ok'] = True
    sys.argv = [script, *argv]
    try:
        runpy.run_path(script, run_name='__main__')
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    identity['script_completed'] = True
finally:
    identity['torch_loaded'] = 'torch' in sys.modules
    identity['triton_loaded'] = 'triton' in sys.modules
    Path(identity_file).write_text(json.dumps(identity, indent=2) + '\n')
