"""Build a real tiny immutable source through Runtime/native, then spend exact Hub grants."""
import io
import json
from pathlib import Path
import sys
import re

import tensorfs
from cozy_runtime.author import WeightsTarget,WeightsTensor,WeightsPart
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
def failure(kind,error,trace):
 while trace.tb_next is not None:trace=trace.tb_next
 print(json.dumps({'proof_error':kind.__name__,'line':trace.tb_lineno,'code':str(getattr(error,'code','')),'detail':re.sub(r'https?://\S+','<url>',str(error))[:500]}),flush=True)
sys.excepthook=failure
root=Path(sys.argv[1]);root.mkdir(parents=True)
store=tensorfs.Store.init(root/'store')
plain=next(digest for alias,digest in tensorfs.seed_digests() if alias=='plain/1')
host=WeightsTransactionHost(store=store,owner_scope='record-owner/recovery-source-fixture',request_id='recovery-source-fixture-20260906',
 invocation_spec_digest='sha256:'+'1'*64,work_fingerprint='sha256:'+'2'*64,writer_session_id=1,allowed_sources={},output_bounds={'source':8192})
request=WeightsCommit('source',{}, {'seed':WeightsTarget(add={'value':WeightsTensor(logical_dtype='f32',shape=(512,),encoding=plain,parts={'value':WeightsPart('f32',(512,))})})},{},(('seed','value'),),8192)
writer=host.open(request);writer.add_part('seed','value','value',io.BytesIO(b'\x00\x00\x80?'*512));receipt=writer.commit()
facts=json.loads(receipt.tensorfs_receipt)
manifest='sha256:'+facts['manifest']['sha256'];length=facts['manifest']['length']
lease=store.acquire_cozytensors(manifest)
refs=dict(lease.objects);refs[manifest]=length
print(json.dumps({'manifest_id':manifest,'manifest_length':length,'objects':[{'object_id':key,'length':value} for key,value in sorted(refs.items())]}),flush=True)
for line in sys.stdin:
 command=json.loads(line)
 if command['action']=='stop':break
 grant=command['grant']
 headers=grant['required_headers']
 text='\n'.join([grant['url'],*[key+': '+value for key,value in headers.items()]])
 result=store.checkpoint_push(grant['object_id'],grant['length'],grant['object_id']==manifest,text)
 print(json.dumps(result),flush=True)
lease.release()
