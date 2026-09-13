"""Boot one task-only CPU Host through its normal supervisor entrypoint."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

IMAGE = sys.argv[2] if len(sys.argv) > 2 else "tensorhub/worker@sha256:1b5c8593fd4f8680df9c6e568416f941d5a75ac881b626e1f4028d8c0454c2ed"
authority_path = Path(sys.argv[1]).resolve(strict=True)
authority = json.loads(authority_path.read_text())
root = authority_path.parent
name = "cozy-proto051-" + uuid.uuid4().hex[:12]
bootstrap_key = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
environment = {
    "COZY_WORKER_ID": authority["worker_id"],
    "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL": bootstrap_key,
    "COZY_WORKER_INTERNAL_PORT": "19781",
    "COZY_MEDIA_INTERNAL_PORT": "19782",
    "COZY_RECORD_OWNER_AUTH_JSON": json.dumps({
        "control_public_key_ed25519_b64url": authority["control_public_key_ed25519_b64url"],
        "media_token_sha256": authority["media_token_sha256"],
    }, separators=(",", ":")),
    "TENSORHUB_ORIGIN": "https://tensorhub.invalid",
}
env_path = root / "host-boot.env"
with env_path.open("x") as file:
    os.chmod(env_path, 0o600)
    file.write("".join(key + "=" + value + "\n" for key, value in environment.items()))
gpu = ["--gpus", "device=0"] if len(sys.argv) > 3 and sys.argv[3] == "gpu" else []
container = subprocess.check_output([
    "docker", "run", "-d", "--name", name,
    "--label", "com.cozy.task=proto051-disconnected-owner",
    "--memory", "4g", "--cpus", "2", "--pids-limit", "512",
    "-p", "127.0.0.1::19781", "-p", "127.0.0.1::19782",
    "--env-file", str(env_path), *gpu, IMAGE,
], text=True).strip()
ports = json.loads(subprocess.check_output([
    "docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", container,
], text=True))
control = "127.0.0.1:" + ports["19781/tcp"][0]["HostPort"]
media = "https://127.0.0.1:" + ports["19782/tcp"][0]["HostPort"]
print(json.dumps({
    "container": container, "name": name, "image": IMAGE,
    "control_address": control, "media_url": media,
    "readiness_url": media + "/v1/bootstrap/receipt",
    "tls_certificate_path_in_container": "/run/cozy/bootstrap/tls.crt",
    "readiness_envelope_path_in_container": "/run/cozy/bootstrap/readiness-envelope.json",
}, sort_keys=True))
