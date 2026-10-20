"""Read public model metadata only. Native A11 inventories v1 assets before phase two."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import urllib.error
import urllib.parse
import urllib.request

A11 = "e2c8ded23669b7b18185808c1c1c0934693def3b"
TARGETS = {
    "minimax-h3": "1.0.0-rc.3",
    "minimax-h3-turbo-lora": "1.0.0-audit.1",
    "sdxl": "1.0.0",
    "qwen-image-2.1": "0.1.0",
    "flux-3-klein": "0.1.0",
    "flux-3-klein-base": "0.1.0",
    "flux-3-klein-upsampler": "0.2.0",
    "qwen3.5-9b": "0.2.0",
    "qwen3.5-9b-mtp": "0.2.0",
}
CAP = 32 << 20
TOTAL = 128 << 20


def require(ok, detail):
    if not ok:
        raise RuntimeError(detail)


def reference(at):
    sha = at["sha256"]
    require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid metadata digest")
    require(isinstance(at["length"], int) and 0 <= at["length"] <= CAP, "metadata object exceeds cap")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_):
        return None


class Reader:
    def __init__(self, output, origin):
        self.output, self.origin = output, origin.rstrip("/")
        self.log = output / "reads.ndjson"
        self.total = sum(json.loads(line)["bytes"] for line in self.log.read_text().splitlines()) if self.log.exists() else 0
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, url, cap, kind, sha, body=None):
        for _ in range(6):
            parsed = urllib.parse.urlsplit(url)
            host = parsed.hostname or ""
            require(parsed.scheme == "https" and parsed.port in (None, 443), "metadata requires HTTPS")
            require(host == "tensorhub.com" or host.endswith((".tensorhub.com", ".r2.cloudflarestorage.com", ".r2.dev")), "metadata redirect host refused")
            request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"} if body else {}, method="POST" if body else "GET")
            try:
                with self.opener.open(request, timeout=45) as response:
                    declared = response.headers.get("Content-Length")
                    require(declared is None or int(declared) <= cap, "metadata Content-Length exceeds cap")
                    data = response.read(cap + 1)
                    self.total += len(data)
                    require(len(data) <= cap and self.total <= TOTAL, "metadata read budget exceeded")
                    with self.log.open("a") as log:
                        log.write(json.dumps({"kind": kind, "sha256": sha, "status": response.status, "bytes": len(data)}) + "\n")
                    return data
            except urllib.error.HTTPError as error:
                if error.code in (301, 302, 303, 307, 308):
                    url = urllib.parse.urljoin(url, error.headers["Location"])
                    body = None
                    continue
                raise RuntimeError(f"{kind} {sha}: HTTP {error.code}") from None
        raise RuntimeError("metadata redirect limit exceeded")

    def object(self, sha, manifest, length=None, kind="metadata"):
        require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid object digest")
        path = self.output / "objects" / sha
        if path.is_file():
            data = path.read_bytes()
        else:
            body = json.dumps({"digests": [sha], "manifest": manifest}).encode()
            doc = json.loads(self.request(self.origin + "/v1/tensorfs/presign", 1 << 20, "read-presign", sha, body))
            require(set(doc["urls"]) == {sha}, "presign widened the metadata selection")
            data = self.request(doc["urls"][sha], length if length is not None else CAP, kind, sha)
        require(len(data) <= CAP and (length is None or len(data) == length), "metadata length mismatch")
        require(hashlib.sha256(data).hexdigest() == sha, "metadata hash mismatch")
        path.write_bytes(data)
        return data, {"sha256": sha, "length": len(data)}


def metadata_path(name):
    parts = pathlib.PurePosixPath(name).parts
    require(parts and ".." not in parts and not name.startswith("/"), "unsafe source file name")
    require(parts[0] not in ("chunks", "blobs", "weights"), "tensor directory is not metadata")
    require(not name.lower().endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".zip", ".tar", ".gz")), "tensor/archive file is not metadata")


def collect(args):
    output = args.output.resolve()
    (output / "objects").mkdir(parents=True, exist_ok=True)
    (output / "repos").mkdir(exist_ok=True)
    reader = Reader(output, args.origin)
    bundle_path = output / "bundle.json"
    bundle = json.loads(bundle_path.read_text()) if bundle_path.exists() else {"format": "cozy.a11.proof-source/1", "a11_source": A11, "objects": {}, "repos": []}
    if args.inventory:
        inventory = json.loads(args.inventory.read_text())
        require(inventory["a11_source"] == A11 and inventory["tensor_chunk_reads"] == 0, "unapproved asset inventory")
        for row in inventory["assets"].values():
            at = row["object"]
            reference(at)
            _, verified = reader.object(at["sha256"], row["manifest"], at["length"], "header-asset")
            bundle["objects"][at["sha256"]] = verified
    else:
        require(args.cozy is not None, "initial collection requires the approved cozy client")
        repos = []
        for name, version in TARGETS.items():
            full = "fidika/" + name
            result = subprocess.run([str(args.cozy), "model", "info", full, "--json"], check=True, capture_output=True)
            require(len(result.stdout) <= 2 << 20, "model information exceeds cap")
            info = json.loads(result.stdout)
            (output / "repos" / (full.replace("/", "__") + ".info.json")).write_bytes(result.stdout)
            releases = [release for release in info["releases"] if release["release"] == version]
            require(len(releases) == 1, f"{full}: requested release is absent or ambiguous")
            release = releases[0]
            lanes = {}
            for lane in release["lanes"]:
                sha = lane["checkpoint"].split("#sha256:")[-1]
                raw, at = reader.object(sha, sha, kind="manifest")
                bundle["objects"][sha] = at
                doc = json.loads(raw)
                if "entries" in doc:
                    require(all(entry["kind"] in ("file", "cozytensors") for entry in doc["entries"]), "unexpected source manifest entry kind")
                    files = [(entry["path"], entry["blob"]) for entry in doc["entries"]]
                else:
                    require(doc.get("format") == "tensorfs.manifest/1", "unknown source manifest")
                    files = list(doc["files"].items())
                for filename, file in files:
                    metadata_path(filename)
                    reference(file)
                    _, verified = reader.object(file["sha256"], sha, file["length"], "header" if filename.endswith(".cozytensors") else "manifest-file")
                    bundle["objects"][file["sha256"]] = verified
                lanes[lane["lane"]] = at
            repos.append({"name": full, "releases": [{"version": version, "revision": release["revision"], "yanked": release["yanked"], "lanes": lanes}], "checkpoints": []})
            bundle["repos"] = repos
            bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    receipt = {"repos": [repo["name"] for repo in bundle["repos"]], "metadata_objects": len(bundle["objects"]), "metadata_bytes": sum(at["length"] for at in bundle["objects"].values()), "tensor_chunk_downloads": 0, "hub_mutations": 0}
    (output / "collection-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--origin", default="https://tensorhub.com")
    parser.add_argument("--cozy", type=pathlib.Path)
    parser.add_argument("--inventory", type=pathlib.Path)
    try:
        collect(parser.parse_args())
    except Exception as error:
        # URLs containing signed query strings and credential values never reach logs.
        print(json.dumps({"failed": type(error).__name__, "detail": str(error) if isinstance(error, RuntimeError) else "metadata collection failed"}))
        raise SystemExit(1) from None
