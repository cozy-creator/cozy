import json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve()
final=json.loads((p/'final-after-gc-proof.json').read_text())
checks=[('partial-proof.json','job-82f7d42d8cfb53b0527692f5'),('changed-callee-proof.json','job-d2f79bbbd3c3eb992b834807'),('commit-gap-proof.json','job-d1f3a273fa911869abeac111'),('automatic-partial-proof.json','job-89fd763eadfab64ee0292aea'),('shared-quantizer-proof.json','job-d8bdfbb8c027cce3bcd50722'),('shared-quantizer-edited-proof.json','job-8410ba95953b141d918d1bd2')]
for name,parent in checks:
 old=json.loads((p/name).read_text())['parents'][parent][-1];new=final['parents'][parent][-1]
 assert 'native_absent' not in new,parent
 assert old['result']==new['result']
 assert [(v['id'],v['bytes']) for v in old['native']]==[(v['id'],v['bytes']) for v in new['native']]
 for value in new['native']:assert value['sha256']==value['id']
prune=json.loads((p/'final-prune.json').read_text());assert prune['removed_entries']==2 and prune['reclaimed_bytes']==2107185 and not prune['store_busy']
assert json.loads((p/'rental-end.json').read_text())['state']=='ended'
last=json.loads((p/'rental-list-after-end.json').read_text());assert last['machines_running']==0 and last['rentals']==[] and last['hourly_spend_usd_micros']==0
print('PASS: six exact returned roots survive native GC; owned rental ended and absent.')
