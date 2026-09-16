import base64,json,os,socket,subprocess,sys,uuid
from pathlib import Path
authority_path=Path(sys.argv[1]).resolve(strict=True)
authority=json.loads(authority_path.read_text());root=authority_path.parent
ports=[]
for _ in range(2):
 with socket.socket() as listener:
  listener.bind(('127.0.0.1',0));ports.append(listener.getsockname()[1])
control,media=ports
boot_key=base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip('=')
environment={'COZY_WORKER_ID':authority['worker_id'],'COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL':boot_key,'COZY_WORKER_INTERNAL_PORT':str(control),'COZY_MEDIA_INTERNAL_PORT':str(media),'COZY_RECORD_OWNER_AUTH_JSON':json.dumps({'control_public_key_ed25519_b64url':authority['control_public_key_ed25519_b64url'],'media_token_sha256':authority['media_token_sha256']},separators=(',',':')),'TENSORHUB_ORIGIN':'https://tensorhub.invalid'}
env=root/'host-boot.env'
with env.open('x') as file:
 os.chmod(env,0o600);file.write(''.join(k+'='+v+'\n' for k,v in environment.items()))
name='cozy-h3-old55-' +uuid.uuid4().hex[:10]
image=json.loads((Path(__file__).resolve().parent/'manifest.json').read_text())['image_id']
container=subprocess.check_output(['docker','--context','default','run','-d','--runtime=runc','-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=','--name',name,'--network','host','--label','com.cozy.task=h3-old55-compat','--memory','4g','--cpus','2','--pids-limit','512','--env-file',str(env),image],text=True).strip()
print(json.dumps({'container':container,'name':name,'control_address':'127.0.0.1:'+str(control),'media_url':'https://127.0.0.1:'+str(media),'readiness_url':'https://127.0.0.1:'+str(media)+'/v1/bootstrap/receipt','tls_certificate_path_in_container':'/run/cozy/bootstrap/tls.crt'}))
