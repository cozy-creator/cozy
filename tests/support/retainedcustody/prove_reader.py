import base64,hashlib,http.server,io,json,subprocess,sys,tempfile,threading
from pathlib import Path
import tensorfs

with tempfile.TemporaryDirectory(prefix='cozy-serial-recovery-proof-') as directory:
    root=Path(directory);store=tensorfs.Store.init(root/'store')
    plain=next(d for n,d in tensorfs.seed_digests() if n=='plain/1')
    writer=store.begin_derived('sha256:'+'1'*64,1,{}, {'model':{'drop':[],'add':{'weight':{'logical_dtype':'f32','shape':[1024],'encoding':plain,'parts':{'value':{'dtype':'f32','shape':[1024]}}}}}}, {'model':{'kind':'add'}}, [('model','weight')],8192)
    body=b'\0\0\x80?'*1024
    writer.add_part('model','weight','value',io.BytesIO(body));writer.add_config('model',io.BytesIO(b'{}'))
    receipt=writer.commit();manifest={'digest':'sha256:'+receipt['manifest']['sha256'],'length':receipt['manifest']['length']}
    store.derived_adopt('sha256:'+'1'*64,'operator-proof')
    objects=[{'digest':str(x['id']),'length':x['length']} for x in store.walk(manifest['digest'])]+[manifest]
    uploaded={};attempts=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_PUT(self):
            digest=self.path.split('?')[0].split('/')[-1];n=int(self.headers['Content-Length']);raw=self.rfile.read(n)
            assert self.headers['If-None-Match']=='*'
            assert self.headers['X-Amz-Checksum-Sha256']==base64.b64encode(bytes.fromhex(digest)).decode()
            assert hashlib.sha256(raw).hexdigest()==digest
            attempts.append(digest);status=412 if digest in uploaded else 200;uploaded[digest]=raw
            self.send_response(status);self.send_header('Content-Length','0');self.send_header('ETag','proof-etag');self.end_headers()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('reader.py')),'--store',str(root/'store'),'--tfs',str(Path(sys.executable).parent/'tfs'),'--allow-local'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        def ask(obj):
            request={'manifest':manifest,'object':obj,'grant':{'url':f'http://127.0.0.1:{server.server_port}/objects/{obj["digest"][7:]}?secret=never-print','required_headers':{'if-none-match':'*','x-amz-checksum-sha256':base64.b64encode(bytes.fromhex(obj['digest'][7:])).decode()}}}
            child.stdin.write(json.dumps(request)+'\n');child.stdin.flush();raw=child.stdout.readline();assert raw and 'never-print' not in raw;return json.loads(raw)
        for obj in objects:
            result=ask(obj);assert result['ok'] and result['http_status']==200,result
            assert result['manifest_digest']==manifest['digest'] and result['checksum_sha256']==obj['digest']
            assert result['transferred_bytes']==obj['length']
        replay=ask(objects[0]);assert replay['ok'] and replay['http_status']==412
        before=len(attempts);wrong=ask({'digest':'sha256:'+'f'*64,'length':5});assert not wrong['ok'] and len(attempts)==before
        assert uploaded[hashlib.sha256(body).hexdigest()]==body
        child.stdin.close();assert child.wait(timeout=5)==0;errors=child.stderr.read();assert 'never-print' not in errors
        assert store.derived_lookup('sha256:'+'1'*64)['disposition']['kind']=='adopted'
        print(json.dumps({'tensorfs':tensorfs.__version__,'objects':len(objects),'new_bytes':sum(len(v) for v in uploaded.values()),'serial_actual_native_http':True,'unknown_member_refused_before_socket':True,'replay_412_observed':True,'original_disposition_retained':True,'secrets_not_printed':True}))
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)
        server.shutdown();server.server_close()
