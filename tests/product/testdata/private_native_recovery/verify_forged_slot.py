import json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve()
proof=json.loads((p/'forged-slot-proof.json').read_text());rows=proof['parents']['job-8d6020e92989d6a72beab4a9']
assert rows[0]['executions']==1 and rows[0]['result'] is not None
assert rows[1]['executions']==0 and rows[1]['result'] is None
assert rows[1]['safe_code']=='child_admission_refused'
assert rows[1]['safe_detail']=='model artifact has no exact owned producer provenance'
result={'passed':True,'versions':proof['versions'],'source_executions':1,'candidate_executions':0,'refusal':rows[1]['safe_code'],'detail':rows[1]['safe_detail'],'scope':'wrong output slot on genuine artifact denied before candidate execution; source reexecuted after Runtime cohort change, not a memo hit'}
(p/'verified-forged-slot.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
