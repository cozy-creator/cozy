import hashlib,json,struct,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
proof=json.loads((root/'changed-callee-proof.json').read_text())
old,new=(proof['parents'][p] for p in ['job-182e30ce414c8fcdb648fddc','job-d2f79bbbd3c3eb992b834807'])
assert old[0]['executions']==new[0]['executions']==0
for key in ['result','intent','computation','revision']:
 assert old[0][key]==new[0][key],key
assert old[1]['intent']==new[1]['intent']
assert old[1]['computation']!=new[1]['computation'] and old[1]['revision']!=new[1]['revision']
assert new[1]['executions']==1 and len(new[1]['native'])==2
expected='sha256:'+hashlib.sha256(struct.pack('<512f',*([22.0]*512))).hexdigest()
assert all(p=={'bytes':2048,'id':expected,'value':22.0,'values':512} for p in new[1]['native'])
events=[json.loads(line) for line in (root/'native-events.jsonl').read_text().splitlines()]
operations={old[1]['child_request'],new[1]['child_request']}
events=[e for e in events if e.get('operation') in operations]
assert [e['key'] for e in events if e['kind']=='part']==['a','a','b']
assert [e['parts'] for e in events if e['kind']=='completed']==[[],[]]
faults=[e for e in events if e['kind']=='fault'];assert len(faults)==1 and faults[0]['facts']['index']==0
result={'passed':True,'scope':'private same-version callee edit rejects incompatible partial','versions':proof['versions'],'write_order':['a','a','b'],'result':new[1]['result']}
(root/'verified-changed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
