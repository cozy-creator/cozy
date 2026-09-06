"""Actual Runtime/native partial-role custody; commands are fixture coordination only."""
import base64
import io
import json
from pathlib import Path
import sys
import struct
import threading
from concurrent import futures

import grpc
import tensorfs
from cozy_runtime.author import WeightsTarget, WeightsTensor, WeightsPart
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal.weights_sink import WeightsTransactionHost, WeightsHostBinding
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.internal.worker import checkpoint_transport
from cozy_runtime.protocol import worker_pb2 as pb, worker_pb2_grpc as rpc

def failure(kind, error, trace):
 while trace.tb_next is not None: trace=trace.tb_next
 print(json.dumps({'proof_error':kind.__name__,'line':trace.tb_lineno}),flush=True)
sys.excepthook=failure

root, request_id, phase = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
scale = struct.pack('<f', float(int(request_id[-6:], 16) + 1)) * 1024
root.mkdir(parents=True)
store = tensorfs.Store.init(root / 'store')
plan = next(digest for alias,digest in tensorfs.seed_digests() if alias=='fp8-rowwise/1')
request = WeightsCommit('model',{}, {'model':WeightsTarget(add={'weight':WeightsTensor(
 logical_dtype='bf16',shape=(1024,4096),encoding=plan,parts={
 'data':WeightsPart(dtype='f8_e4m3fn',shape=(1024,4096)),
 'scale':WeightsPart(dtype='f32',shape=(1024,))})})},{},(('model','weight'),),8<<20)
observed=[]
host = WeightsTransactionHost(store=store,owner_scope='record-owner/local/default',request_id=request_id,
 invocation_spec_digest='sha256:'+'a'*64,work_fingerprint='sha256:'+'b'*64,writer_session_id=1,
 allowed_sources={},output_bounds={'model':8<<20},record_checkpoint=observed.append)
transaction_id=host.transaction_id('model')
# The current declaration is always produced by the native encoder.
targets={key:host._target(value) for key,value in request.targets.items()}
declaration=store.derived_declaration({},targets,{},list(request.order),8<<20,work_fingerprint=host.work_fingerprint)
import hashlib
status=pb.WeightsTransactionStatus(weights_transaction_id=transaction_id,request_id=request_id,attempt_ordinal=1 if phase=='initial' else 2,
 invocation_spec_digest=host.invocation_spec_digest,output_slot='model',writer_epoch=1,
 tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),state=pb.WEIGHTS_TRANSACTION_STATE_INTENT)
server=grpc.server(futures.ThreadPoolExecutor(max_workers=8))
rpc.add_RuntimePreparationServicer_to_server(_PreparationServicer(None,None,None,None,None,
 checkpoint_pager=lambda ask:checkpoint_transport.page(ask,tensorfs_root=root/'store'),
 checkpoint_transfer=lambda ask:checkpoint_transport.transfer(ask,tensorfs_root=root/'store',allow_local=True)),server)
port=server.add_insecure_port('127.0.0.1:0');server.start()

def emit(**values):
 print(json.dumps(values),flush=True)

def point():
 row=observed[-1]
 return pb.CheckpointRef(head=pb.Ref(digest=bytes.fromhex(row.head[7:]),length=row.head_length),
  plan_digest=bytes.fromhex(row.plan_digest[7:]),index=row.index,bytes=row.bytes)

if phase=='initial':
 transaction=host.open(request)
 transaction.add_part('model','weight','data',io.BytesIO(b'\x31'*(4<<20)))
 transaction.checkpoint()
 status.checkpoint.CopyFrom(point())
 entered=threading.Event()
 class InterruptedScale:
  def read(self,size=-1):
   entered.set()
   threading.Event().wait()
 threading.Thread(target=lambda:transaction.add_part('model','weight','scale',InterruptedScale()),daemon=True).start()
 assert entered.wait(5),'partial role did not enter native reader'
emit(address=f'127.0.0.1:{port}',status=base64.b64encode(status.SerializeToString()).decode(),phase=phase)
for line in sys.stdin:
 command=json.loads(line)
 if command['action']=='resume':
  checkpoint=pb.CheckpointRef.FromString(base64.b64decode(command['checkpoint']))
  head='sha256:'+checkpoint.head.digest.hex()
  assert store.derived_lookup(transaction_id)['state']=='absent'
  store.validate_derived_checkpoint(transaction_id,declaration,head,checkpoint.head.length,
   operation_id=request_id,slot='model',plan_digest='sha256:'+checkpoint.plan_digest.hex())
  assert store.derived_lookup(transaction_id)['state']=='absent'
  host.bind_intent=lambda *args:WeightsHostBinding(transaction_id,1,checkpoint=(head,checkpoint.head.length))
  transaction=host.open(request)
  assert transaction.completed_parts==frozenset({('model','weight','data')})
  transaction.add_part('model','weight','scale',io.BytesIO(scale))
  transaction.checkpoint()
  status.checkpoint.CopyFrom(point());status.intent_ready=True
  emit(status=base64.b64encode(status.SerializeToString()).decode(),reused_bytes=4<<20,new_role_bytes=4096)
 elif command['action']=='commit':
  receipt=transaction.commit()
  store.derived_adopt(transaction_id,'creator-chosen-checkpoint-proof')
  tensorfs.gc(str(root/'store'))
  facts=json.loads(receipt.tensorfs_receipt)
  manifest='sha256:'+facts['manifest']['sha256']
  with store.acquire_cozytensors(manifest) as lease:
   header=store.manifest(manifest)['header']
   plan=tensorfs.plan(header,[('model','weight')],window=1<<20)
   slots=[memoryview(bytearray(1<<20)) for _ in range(2)]
   chunks={}
   def got(batch):chunks[batch.index]=bytes(slots[batch.slot][:batch.nbytes]);batch.release()
   lease.stream(plan,slots,got)
  expected=b'\x31'*(4<<20)+scale
  assert b''.join(chunks[index] for index in sorted(chunks))==expected
  emit(committed=True,manifest=manifest,verified_bytes=len(expected))
  break
server.stop(0)
