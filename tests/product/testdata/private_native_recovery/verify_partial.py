import hashlib,json,struct,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
proof=json.loads((root/'partial-proof.json').read_text())
old,new=(proof['parents'][p] for p in ['job-1557ba1bb6affd0ff1b4d1e2','job-82f7d42d8cfb53b0527692f5'])
assert old[0]['executions']==1 and new[0]['executions']==0
for key in ['result','intent','computation','revision']:
 assert old[0][key]==new[0][key],key
for key in ['intent','computation','revision']:
 assert old[1][key]==new[1][key],key
assert new[1]['executions']==1 and len(new[1]['native'])==2
expected='sha256:'+hashlib.sha256(struct.pack('<512f',*([14.0]*512))).hexdigest()
assert all(p=={'bytes':2048,'id':expected,'value':14.0,'values':512} for p in new[1]['native'])
events=[json.loads(line) for line in (root/'native-events.jsonl').read_text().splitlines()]
operations={c['child_request'] for c in old+new}
events=[e for e in events if e.get('operation') in operations]
assert [e['key'] for e in events if e['kind']=='part']==['weight','a','b']
assert [e['parts'] for e in events if e['kind']=='completed']==[[],[['model','a','value']]]
faults=[e for e in events if e['kind']=='fault'];assert len(faults)==1 and faults[0]['facts']['index']==0
result={'passed':True,'scope':'explicit private CLI retry adopts native partial after Runtime crash','versions':proof['versions'],'write_order':['weight','a','b'],'result':new[1]['result'],'automatic_retry':'failed on Runtime24 admission_epoch_stale; Runtime611 candidate not yet privately qualified'}
(root/'verified-partial.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
