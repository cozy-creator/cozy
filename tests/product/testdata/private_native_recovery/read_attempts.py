import json,sqlite3,sys
from cozy_runtime.protocol import documents,worker_pb2 as pb
c=sqlite3.connect('file:/var/lib/tensorfs/.cozy-workspace/journal.sqlite3?mode=ro',uri=True);c.row_factory=sqlite3.Row
p=sys.argv[1];assert p.startswith('job-') and p[4:].isalnum()
requests=[p]+[r[0] for r in c.execute('SELECT child_request FROM execution_calls WHERE parent_request=? ORDER BY call_index',(p,))]
result={}
for r in requests:
 rows=[]
 for a in c.execute('SELECT ordinal,state,outcome FROM attempts WHERE request=? ORDER BY ordinal',(r,)):
  outcome=documents.read(a['outcome'],pb.AttemptOutcomeBody) if a['outcome'] else {}
  rows.append({'ordinal':a['ordinal'],'state':a['state'],'status':pb.OutcomeStatus.Name(outcome.get('status',0)),'cause':pb.CauseCode.Name(outcome.get('cause',{}).get('code',0)),'origin':pb.CauseOrigin.Name(outcome.get('cause',{}).get('origin',0)),'safe_message':outcome.get('safe_message','')})
 result[r]=rows
print(json.dumps(result,sort_keys=True))
