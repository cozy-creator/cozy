"""Real public Runtime/TensorFS source preparation and checkpoint transport for a live owner proof.

Initial stdout: {request: base64 PrepareModelSourceRequest, software: ...}.
First stdin: {operation_id: Creator-assigned request ID}.
Second stdout: {address: actual RuntimePreparation gRPC address, prepared: base64 result}.
Close stdin to stop. No account credential or grant URL is printed or accepted on argv.
"""
import base64
import gzip
import hashlib
import importlib.metadata
import json
import math
import secrets
from pathlib import Path
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import grpc
from cozy_runtime._build_provenance import COMMIT
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.internal.worker import model_source_prepare, checkpoint_transport
from cozy_runtime.protocol import worker_pb2 as pb, worker_pb2_grpc as pb_grpc

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
wide = '--transfer-proof' in sys.argv[2:]
geometry_path = Path(__file__).with_name('h3-native-source-geometry.json.gz')
geometry = json.loads(gzip.decompress(geometry_path.read_bytes()))
request = pb.PrepareModelSourceRequest(
    source_selection_digest=hashlib.sha256(b'builtin-h3-source-live-custody-proof').digest(),
    profiles=[pb.ModelSourceProfile(slot='dits', profile='hf/minimax-h3/native-dual-bf16/1')])
for task, tensors in geometry.items():
    sizes = {key: math.prod(row['shape']) * {'BF16': 2, 'F32': 4}[row['dtype']] for key, row in tensors.items()}
    chosen = sorted((key for key, n in sizes.items() if n >= (65536 if wide else 257)), key=lambda key: sizes[key])[:4 if wide else 1]
    assert len(tensors) == 535
    small = b''.join((secrets.token_bytes(32) * ((sizes[key] + 31) // 32))[:sizes[key]] if wide else bytes(sizes[key]) for key in sorted(chosen))
    index = json.dumps({'weight_map': {key: 'small.safetensors' if key in chosen else 'rest.safetensors' for key in tensors}}, sort_keys=True, separators=(',', ':')).encode()
    for filename, selected in [('model.safetensors.index.json', None), ('small.safetensors', chosen), ('rest.safetensors', [key for key in tensors if key not in chosen])]:
        member = f'{task}/transformer/{filename}'
        if selected is None:
            prefix = index
            length = len(index)
            data = index
        else:
            offset = 0
            head = {}
            for key in sorted(selected):
                head[key] = {**tensors[key], 'data_offsets': [offset, offset + sizes[key]]}
                offset += sizes[key]
            encoded = json.dumps(head, sort_keys=True, separators=(',', ':')).encode()
            prefix = struct.pack('<Q', len(encoded)) + encoded
            length = len(prefix) + offset
            data = prefix + small if filename == 'small.safetensors' else None
        header = root / (task + '-' + filename + '.header')
        header.write_bytes(prefix)
        path = root / (task + '-' + filename)
        if data is not None:
            path.write_bytes(data)
        digest = hashlib.sha256(data if data is not None else member.encode()).hexdigest()
        request.files.append(pb.LocalModelSourceFile(member=member, object_id='sha256:' + digest, length=length,
            path=str(path), header_path=str(header), verified=data is not None))
request.files.sort(key=lambda row: row.member)
print(json.dumps({'request': base64.b64encode(request.SerializeToString()).decode(), 'software': {
    'runtime': importlib.metadata.version('cozy-runtime'), 'tensorfs': importlib.metadata.version('tensorfs'), 'commit': COMMIT}}), flush=True)
request.operation_id = json.loads(sys.stdin.readline())['operation_id']
store = root / 'store'
prepared = model_source_prepare.prepare_model_source(request, tensorfs_root=store)
if prepared.outcome != pb.MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE or len(prepared.checkpoints) != 1:
    raise RuntimeError(f'actual partial source preparation refused: {prepared.safe_code} {prepared.safe_detail}')
def serve(selected_store):
    service = _PreparationServicer(None, None,
        lambda value: model_source_prepare.prepare_model_source(value, tensorfs_root=selected_store), None, None,
        lambda value: checkpoint_transport.page(value, tensorfs_root=selected_store),
        lambda value: checkpoint_transport.transfer(value, tensorfs_root=selected_store, allow_local=wide))
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_RuntimePreparationServicer_to_server(service, server)
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()
    return server, f'127.0.0.1:{port}'
server, address = serve(store)
restore, restore_address = serve(root / 'restored')
loop_restore, loop_address = serve(root / 'loop-restored')
print(json.dumps({'address': address, 'restore_address': restore_address,
    'loop_restore_address': loop_address, 'prepared': base64.b64encode(prepared.SerializeToString()).decode()}), flush=True)
sys.stdin.readline()
server.stop(0).wait()
restore.stop(0).wait()

loop_restore.stop(0).wait()
