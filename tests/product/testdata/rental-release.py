from pathlib import Path
import base64,hashlib,http.server,json,os,subprocess,tempfile,threading,sys
binary=sys.argv[1]
results=[]
for arm in ['readonly','release','wrong-boot','initial-absent','released-absent','changed-worker','unexpected-state']:
    events=[]
    class Hub(http.server.BaseHTTPRequestHandler):
        def answer(self,status,value):
            raw=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_POST(self):
            if self.path.endswith('/begin'):self.answer(200,{'challenge_id':'fixture','challenge':base64.urlsafe_b64encode(bytes(32)).rstrip(b'=').decode(),'expires_at':'2099-01-01T00:00:00Z'})
            elif self.path.endswith('/finish'):self.answer(200,{'token_set':{'access_token':'fixture-token','token_type':'Bearer','expires_in':3600},'device_key':{'id':'fixture'}})
            else:self.answer(404,{})
        def do_GET(self):
            assert self.headers.get('Authorization')=='Bearer fixture-token'
            if arm=='initial-absent' or arm=='released-absent' and events:self.answer(404,{'error':{'code':'rental.not_found','message':'fixture absent'}});return
            value={'rental_id':'pr-22222222222222222222','name':'proof','state':'released' if events else 'ready','hourly_rate_usd_micros':1,'worker_id':'fixture-worker','worker_boot_id':'fixture-boot','provider_state':'gone' if events else 'running','container_state':'gone' if events else 'running'}
            if arm=='wrong-boot':value['worker_boot_id']='foreign-boot'
            if arm=='changed-worker' and events:value['worker_id']='foreign-worker'
            if arm=='unexpected-state':value['state']='unknown-new-state'
            self.answer(200,value)
        def do_DELETE(self):
            assert self.headers.get('Authorization')=='Bearer fixture-token'
            events.append(self.path);self.send_response(204);self.end_headers()
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Hub);threading.Thread(target=server.serve_forever,daemon=True).start()
    hub=f'http://127.0.0.1:{server.server_port}'
    with tempfile.TemporaryDirectory() as tmp:
        home=Path(tmp);auth=home/'auth';auth.mkdir()
        credential=auth/(hashlib.sha256(hub.encode()).hexdigest()+'.json')
        credential.write_text(json.dumps({'version':1,'hub':hub,'email':'fixture@example.test','device_key_id':'fixture','private_key':base64.urlsafe_b64encode(bytes(range(32))).rstrip(b'=').decode()}));credential.chmod(0o600)
        env=dict(os.environ,COZY_HOME=tmp,TENSORHUB_URL=hub)
        args=[binary,'--rental','pr-22222222222222222222','--boot','fixture-boot']+([] if arm=='readonly' else ['--release'])
        result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
        expected=arm in ('readonly','release','released-absent')
        assert (result.returncode==0)==expected,(arm,result.stderr,result.stdout)
        assert len(events)==(1 if arm in ('release','released-absent','changed-worker') else 0),(arm,events)
        assert not (home/'daemon.lock').exists(),arm
        results.append({'arm':arm,'exit':result.returncode,'deletes':len(events),'output':result.stdout.strip(),'refusal':result.stderr.strip()})
    server.shutdown();server.server_close()
print(json.dumps(results,indent=2))
