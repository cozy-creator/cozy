"""Read only the named qualification requests and their tiny native results."""
from pathlib import Path
import importlib.metadata,json,sqlite3,struct,sys,hashlib
import tensorfs
parents=sys.argv[1:]
assert parents and all(p.startswith('job-') and p[4:].isalnum() for p in parents)
db=sqlite3.connect('file:/var/lib/tensorfs/.cozy-workspace/journal.sqlite3?mode=ro',uri=True)
db.row_factory=sqlite3.Row;db.execute('PRAGMA busy_timeout=5000')
store=tensorfs.Store.open('/var/lib/tensorfs')
result={'versions':{n:importlib.metadata.version(n) for n in ('cozy-runtime','tensorfs')},'parents':{}}
for parent in parents:
 rows=db.execute('''SELECT c.child_request,COALESCE(e.state,CASE WHEN c.safe_code<>'' THEN 'refused' ELSE 'memo_hit' END) state,hex(c.intent_digest) intent,c.prepared,c.result,c.safe_code,c.safe_detail,
 (SELECT count(*) FROM attempts a WHERE a.owner=c.owner AND a.request=c.child_request) executions
 FROM execution_calls c LEFT JOIN executions e ON e.owner=c.owner AND e.request=c.child_request
 WHERE c.parent_request=? ORDER BY c.call_index''',(parent,)).fetchall()
 children=[]
 for row in rows:
  plan=json.loads(row['prepared']) if row['prepared'] else {}
  child={k:row[k] for k in ('child_request','state','intent','executions','safe_code','safe_detail')}
  child.update(revision=plan.get('revision'),computation=plan.get('computation'))
  value=json.loads(row['result']) if row['result'] else None
  child['result']=value
  if value and 'manifest' in value:
   manifest=value['manifest']['digest']
   with store.acquire_cozytensors(manifest) as lease:
    parts=[p for p in store.walk_cozytensors(manifest) if p['kind']=='part']
    verified=[]
    for part in parts:
     length=part['length'];assert 0<length<=2097152,part
     raw=bytearray(length);lease.read_into(part['id'],length,0,length,raw)
     digest='sha256:'+hashlib.sha256(raw).hexdigest();assert digest==part['id']
     verified.append({'id':part['id'],'bytes':length,'sha256':digest})
    child['native']=verified
  children.append(child)
 result['parents'][parent]=children
print(json.dumps(result,sort_keys=True))
