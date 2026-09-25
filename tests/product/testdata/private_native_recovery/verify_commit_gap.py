import hashlib,json,struct,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
proof=json.loads((root/'commit-gap-proof.json').read_text())
parent='job-d1f3a273fa911869abeac111'
child,=proof['parents'][parent];operation=child['child_request']
events=[json.loads(line) for line in (root/'native-events.jsonl').read_text().splitlines()]
events=[e for e in events if e.get('operation')==operation]
parts=[e for e in events if e['kind']=='part'];assert len(parts)==1 and parts[0]['key']=='weight'
faults=[e for e in events if e['kind']=='fault'];assert len(faults)==1
fault=faults[0];assert fault['fault']=='receipt_before_ack'
row,=[e['row'] for e in events if e['kind']=='pre_fault_workspace'];assert row['state']=='intent' and row['receipt_bytes']==0
facts=fault['facts'];assert row['id']==facts['transaction_id']
result=child['result'];assert result['manifest']=={'digest':'sha256:'+facts['manifest']['sha256'],'length':facts['manifest']['length']}
receipt='sha256:'+hashlib.sha256(json.dumps(facts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert result['tensorfs_receipt_digest']==receipt
payload='sha256:'+hashlib.sha256(struct.pack('<512f',*([19.0]*512))).hexdigest()
assert child['native']==[{'bytes':2048,'id':payload,'value':19.0,'values':512}]
attempts=json.loads((root/'commit-gap-attempts.json').read_text())
assert [a['cause'] for a in attempts[operation]][1]=='CAUSE_CODE_ADMISSION_EPOCH_STALE'
assert attempts[operation][1]['status']=='OUTCOME_STATUS_REFUSED' and attempts[operation][1]['origin']=='CAUSE_ORIGIN_WORKER'
assert attempts[operation][-1]['status']=='OUTCOME_STATUS_SUCCEEDED' and len(attempts[operation])==3
assert attempts[parent][-1]['status']=='OUTCOME_STATUS_SUCCEEDED' and len(attempts[parent])==2
value={'passed':True,'versions':proof['versions'],'payload_productions':1,'pre_crash_runtime_row':row,'native_receipt_digest':receipt,'automatic_stale_admission_recovery':True,'result':result}
(root/'verified-commit-gap.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
