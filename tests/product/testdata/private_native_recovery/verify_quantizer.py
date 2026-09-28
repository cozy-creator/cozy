import hashlib,json,struct,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
p=json.loads((root/'shared-quantizer-proof.json').read_text());q=json.loads((root/'shared-quantizer-edited-proof.json').read_text())
a=p['parents']['job-d8bdfbb8c027cce3bcd50722'];b=q['parents']['job-8410ba95953b141d918d1bd2']
assert [c['executions'] for c in a]==[1,1]
assert [c['executions'] for c in b]==[0,0]
for before,after in zip(a,b,strict=True):
 for key in ['result','intent','computation','revision','native']:assert before[key]==after[key],key
assert a[0]['result']['manifest']['digest']=='sha256:59c0599c42928bce7c9c6679b0b25a45313510f37f95d2d454412e1bfd0ad2c8'
assert a[1]['result']['manifest']['digest']=='sha256:bce67f12bff1b854f4d457f54c9aed16c0d1fa7dc10493b9d8a98f734eef5cf2'
expected={'sha256:'+hashlib.sha256(struct.pack('<e',v)*(512*1024)).hexdigest() for v in [1.25,2.25]}
assert {part['id'] for part in a[0]['native']}==expected
assert sum(part['bytes'] for part in a[0]['native'])==2097152
assert sum(part['bytes'] for part in a[1]['native'])==1052672
rows=[json.loads(line) for line in (root/'native-events.jsonl').read_text().splitlines()]
assert not any(r['kind']=='part' and r.get('operation') in {c['child_request'] for c in b} for r in rows)
events=[r for r in rows if r.get('operation')==a[1]['child_request']]
assert len([r for r in events if r['kind']=='part'])==4
receipt,=[r['facts'] for r in events if r['kind']=='receipt_before_ack']
assert len(receipt['added_objects'])==3 and sum(o['length'] for o in receipt['added_objects'])==528384
assert 'bias' not in receipt['declaration']['components']['text_encoder']['drop']
assert receipt['declaration']['order'][-1]==['text_encoder','bias']
value={'passed':True,'versions':q['versions'],'execution_counts':[[1,1],[0,0]],'source_payload_bytes':2097152,'quantized_logical_part_bytes':1052672,'distinct_added_native_bytes':528384,'result':b[1]['result'],'matches_historical_local_fixture_manifest':True,'scope':'bounded genuine shared-quantizer private completed reuse, not full source/conversion/checkpoint or trained quality'}
(root/'verified-quantizer.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
