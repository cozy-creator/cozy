import hashlib,json,struct,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
p=json.loads((root/'automatic-partial-proof.json').read_text());parent='job-89fd763eadfab64ee0292aea'
a,b=p['parents'][parent];assert a['executions']==1 and b['executions']==3
rows=[json.loads(line) for line in (root/'native-events.jsonl').read_text().splitlines()]
events=[r for r in rows if r.get('operation') in {a['child_request'],b['child_request']}]
assert [r['key'] for r in events if r['kind']=='part']==['weight','a','b']
assert [r['parts'] for r in events if r['kind']=='completed']==[[],[['model','a','value']]]
assert len([r for r in events if r['kind']=='fault' and r['fault']=='checkpoint'])==1
for child,value,count in [(a,11.,1),(b,23.,2)]:
 digest='sha256:'+hashlib.sha256(struct.pack('<512f',*([value]*512))).hexdigest()
 assert child['native']==[{'bytes':2048,'id':digest,'value':value,'values':512}]*count
attempts=json.loads((root/'automatic-partial-attempts.json').read_text())
assert attempts[b['child_request']][1]['cause']=='CAUSE_CODE_ADMISSION_EPOCH_STALE'
assert attempts[b['child_request']][-1]['status']=='OUTCOME_STATUS_SUCCEEDED'
assert len(attempts[parent])==2 and attempts[parent][-1]['status']=='OUTCOME_STATUS_SUCCEEDED'
result={'passed':True,'versions':p['versions'],'write_order':['weight','a','b'],'automatic_recovery':True,'candidate':b['result']}
(root/'verified-automatic-partial.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
