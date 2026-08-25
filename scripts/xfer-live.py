#!/usr/bin/env python3
"""cl-012 live driver: real hub, real R2, real artifacts, the real `cozy` binary.

No automated tests exist in v2 (decisions.md #160). This runs the transfer verbs
end to end against a live tensorhub and observes what they do, plants each refusal
for real, removes the plant, and benchmarks.

One thing here stands in for something that does not exist yet. th-002 landed the
whole WRITE side of the transfer protocol and no read side: the hub signs PUTs at
final content keys and streams objects back for its own verification, but exposes no
route that hands a client a presigned GET. So a Launch-1 hub can take custody of
bytes and cannot give them back, and `cozy pull` has nothing to fetch from.

The shim below is that missing route, and nothing else: it forwards every request to
the real hub untouched, and answers exactly one path — POST
/v1/repos/{org}/{name}/checkpoints/{snapshot}/reads — with real presigned GETs at
the real final content keys. Forty lines. The client code under test is the product's,
unmodified, dialling one hub; what the shim proves is that the ONLY thing standing
between Launch 1 and a working pull is those forty lines living in tensorhub instead
of here. Every pull number below is honest about its source: the bytes are real R2
bytes, the signature is a real signature, and the grant is not the hub's.
"""
import argparse
import datetime
import hashlib
import hmac
import http.server
import json
import os
import pathlib
import re
import shutil
import socketserver
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

PASS, FAIL, NOTES = 0, 0, []
PRINTED = []


def out(line=""):
    PRINTED.append(str(line))
    print(line)


def section(name):
    out()
    out(f"\033[1m== {name}\033[0m")


def ok(what, detail=""):
    global PASS
    PASS += 1
    out(f"  \033[32mPASS\033[0m {what}" + (f" — {detail}" if detail else ""))


def bad(what, detail=""):
    global FAIL
    FAIL += 1
    out(f"  \033[31mFAIL\033[0m {what}" + (f" — {detail}" if detail else ""))


def note(what):
    out(f"  ---- {what}")


def first(r):
    """The first line a run said, whichever stream carried it. Under --json a typed
    refusal goes to stdout, because stdout must carry exactly one document."""
    said = (r.stderr or "").strip() or (r.stdout or "").strip()
    return said.splitlines()[0][:170] if said else f"exit {r.returncode}, said nothing"


def check(cond, what, detail=""):
    (ok if cond else bad)(what, detail)
    return cond


# ---------------------------------------------------------------- SigV4 presign

EMPTY = hashlib.sha256(b"").hexdigest()


def creds(secrets):
    p = pathlib.Path(secrets)
    return (
        (p / "storage.access_key_id").read_text().strip(),
        (p / "storage.secret_access_key").read_text().strip(),
        (p / "storage.endpoint_url").read_text().strip().rstrip("/"),
        (p / "storage.region").read_text().strip() or "auto",
    )


def signing_key(secret, date, region):
    k = ("AWS4" + secret).encode()
    for part in (date, region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return k


def presign_get(secrets, bucket, key, ttl=900):
    ak, sk, ep, region = creds(secrets)
    host = urllib.parse.urlparse(ep).netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{date}/{region}/s3/aws4_request"
    q = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{ak}/{scope}",
        "X-Amz-Date": stamp,
        "X-Amz-Expires": str(ttl),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
        for k, v in sorted(q.items()))
    path = "/" + bucket + "/" + urllib.parse.quote(key, safe="/")
    creq = "\n".join(["GET", path, canonical_query, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"])
    sts = "\n".join(["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(creq.encode()).hexdigest()])
    sig = hmac.new(signing_key(sk, date, region), sts.encode(), hashlib.sha256).hexdigest()
    return f"{ep}{path}?{canonical_query}&X-Amz-Signature={sig}", stamp


def object_key(prefix, object_id):
    """The hub's own content-key layout (th-002). Composing it here is exactly what
    the missing route would not have to do — that is the point of the shim."""
    h = object_id.replace("sha256:", "")
    return f"{prefix}objects/sha256/{h[0:2]}/{h[2:4]}/{h}"


# ---------------------------------------------------------------- the read shim

class Shim(http.server.BaseHTTPRequestHandler):
    hub = ""
    secrets = ""
    bucket = ""
    prefix = ""
    corrupt = {}         # object id -> the id whose REAL bytes to serve instead
    swap_manifest = None  # snapshot id whose manifest is answered mutated
    reads_armed = True

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    READS = re.compile(r"^/v1/repos/([^/]+)/([^/]+)/checkpoints/(sha256:[0-9a-f]{64})/reads$")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _answer(self, status, body, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        m = self.READS.match(self.path)
        if m and self.reads_armed:
            body = json.loads(self._body() or b"{}")
            reads = []
            for oid in body.get("object_ids", []):
                # A corrupted source serves ANOTHER object's REAL bytes: a live,
                # well-formed, correctly-signed answer that is simply not the object
                # asked for. Answering a 404 would prove only that 404s are handled.
                key = object_key(self.prefix, self.corrupt.get(oid, oid))
                url, _ = presign_get(self.secrets, self.bucket, key)
                reads.append({"object_id": oid, "length": 0, "url": url,
                              "expires_at": ""})
            return self._answer(200, json.dumps({"reads": reads}).encode())
        return self._forward("POST")

    def do_GET(self):
        return self._forward("GET")

    def do_DELETE(self):
        return self._forward("DELETE")

    def do_PUT(self):
        return self._forward("PUT")

    def _forward(self, method):
        body = self._body()
        req = urllib.request.Request(self.hub + self.path, data=body or None, method=method)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "connection"):
                continue
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                raw, status, ctype = r.read(), r.status, r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            raw, status, ctype = e.read(), e.code, e.headers.get("Content-Type", "application/json")
        except urllib.error.URLError as e:
            raw, status, ctype = json.dumps({"error": {"code": "shim.upstream", "message": str(e)}}).encode(), 502, "application/json"
        if self.swap_manifest and self.path.endswith(f"/checkpoints/{self.swap_manifest}/manifest"):
            raw = raw.replace(b"model.cozytensors", b"model.cozytensorS", 1) if b"model.cozytensors" in raw else raw + b" "
        self._answer(status, raw, ctype)


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------- process helpers

class Cozy:
    def __init__(self, binary, base, tfs, token):
        self.binary, self.base, self.tfs, self.token = binary, base, tfs, token

    def env(self, home, token=True, url=None):
        e = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
             "COZY_HOME": home, "COZY_TFS": self.tfs,
             "TENSORHUB_URL": url or self.base}
        if token:
            e["TENSORHUB_TOKEN"] = self.token
        return e

    def run(self, home, *args, token=True, url=None, stdin=None, timeout=2400):
        r = subprocess.run([self.binary, *args], env=self.env(home, token, url),
                           capture_output=True, text=True, input=stdin, timeout=timeout)
        PRINTED.append(r.stdout)
        PRINTED.append(r.stderr)
        return r

    def json(self, home, *args, **kw):
        r = self.run(home, *args, "--json", **kw)
        try:
            return r.returncode, json.loads(r.stdout or "{}"), r
        except json.JSONDecodeError:
            return r.returncode, {}, r


def tfs_run(binary, *args):
    r = subprocess.run([binary, *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def build_artifact(tfs, store, blocks, scale, seed):
    rc, so, se = tfs_run(tfs, "checkpoint", "write", store,
                         "--blocks", str(blocks), "--scale", str(scale), "--seed", str(seed))
    if rc != 0:
        sys.exit("checkpoint write: " + se.strip())
    snap = re.search(r"checkpoint_id\s+sha256:([0-9a-f]{64})", so).group(1)
    total = int(re.search(r"tensor bytes (\d+)", so).group(1))
    return "sha256:" + snap, total


def hub_get(base, path, token=None):
    req = urllib.request.Request(base + path)
    if token:
        req.add_header("Authorization", "Bearer " + token)
        req.add_header("X-Tensorhub-Reason", "cl-012 live verification")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def field(doc, name, default=None):
    return doc.get(name, default)


def ms(fn, reps=3):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return min(t), statistics.median(t)


# ---------------------------------------------------------------- the run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", required=True)
    ap.add_argument("--shim-port", type=int, required=True)
    ap.add_argument("--cozy", required=True)
    ap.add_argument("--tfs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--secrets", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--hub-sha", default="")
    ap.add_argument("--tfs-sha", default="")
    ap.add_argument("--phases", default="")
    a = ap.parse_args()
    want = set(a.phases.split()) if a.phases.strip() else set()

    def on(phase):
        return not want or phase in want

    token = os.environ["CL012_TOKEN"]
    work = pathlib.Path(a.work)
    pub = str(work / "publisher")
    dst = str(work / "puller")
    for d in (pub, dst):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

    Shim.hub, Shim.secrets, Shim.bucket, Shim.prefix = a.hub, a.secrets, a.bucket, a.prefix
    server = Threaded(("127.0.0.1", a.shim_port), Shim)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{a.shim_port}"

    cozy = Cozy(a.cozy, base, a.tfs, token)

    section("subjects")
    note(f"tensorhub {a.hub_sha[:12]} · tensorfs {a.tfs_sha[:12]} · prefix {a.prefix}")
    # The publisher's store is a real tensorfs CAS; the artifact is written into it by
    # tfs itself, which is the two-step model (decisions #58): ingest locally, then
    # declare-first upload. cozy-creator never converts anything.
    pathlib.Path(pub, "cas").mkdir(parents=True, exist_ok=True)
    tfs_run(a.tfs, "store", "init", pub + "/cas")
    small, small_bytes = build_artifact(a.tfs, pub + "/cas", 1, 4, 7)
    note(f"small artifact {small[:19]}… {small_bytes/2**20:.1f} MiB")

    for org_name, kind in (("acme/small", "model"), ("acme/twin", "model"), ("acme/large", "model")):
        rc, doc, r = cozy.json(pub, "repo", "create", org_name, "--kind", kind,
                               "--reason", "cl-012 live verification")
        if rc not in (0, 13):
            bad("repo create " + org_name, r.stderr.strip()[:160])

    result = {}

    # ------------------------------------------------------------ push
    if on("push"):
        section("push — declare-first, upload only what the hub lacks")
        rc, plan, r = cozy.json(pub, "push", "acme/small", small,
                                "--family", "sdxl", "--reason", "cl-012 plan", "--dry-run")
        check(rc == 0, "push --dry-run", f"exit {rc}")
        check(plan.get("kind") == "push plan", "the plan is the HUB's answer",
              f"{plan.get('missing')} missing / {plan.get('held')} held, {plan.get('moved')} to move")

        t0 = time.perf_counter()
        rc, doc, r = cozy.json(pub, "push", "acme/small", small,
                               "--family", "sdxl", "--reason", "cl-012 first publish")
        wall = time.perf_counter() - t0
        check(rc == 0, "push round trip", f"exit {rc} in {wall:.2f}s")
        if rc != 0:
            note((r.stdout + r.stderr).strip()[:900])
        result["small"] = doc
        check(doc.get("snapshot") == small, "the installed root is the snapshot published",
              str(doc.get("snapshot"))[:24])
        check(doc.get("grade") in ("COMPATIBLE", "CONVERTIBLE"), "hermetic verdict", str(doc.get("grade")))
        check(str(doc.get("verified", "")) == str(doc.get("uploaded", "")),
              "every uploaded object was proved by the hub itself",
              f"{doc.get('verified')} verified of {doc.get('uploaded')} uploaded")
        check(doc.get("moved") not in (None, "0B"), "bytes actually moved", str(doc.get("moved")))

        status, cat = hub_get(a.hub, "/v1/repos/acme/small/checkpoints")
        rows = cat.get("checkpoints", [])
        check(any(c["snapshot_id"] == small for c in rows),
              "the public catalog read returns it", f"{len(rows)} row(s)")

    # ------------------------------------------------------------ dedup
    if on("dedup"):
        section("dedup — a second publish of the same set moves 0 bytes")
        t0 = time.perf_counter()
        rc, doc, r = cozy.json(pub, "push", "acme/twin", small,
                               "--family", "sdxl", "--reason", "cl-012 dedup arm")
        wall = time.perf_counter() - t0
        check(rc == 0, "publish the same set into a second repo", f"exit {rc} in {wall:.2f}s")
        check(doc.get("moved") == "0B", "0 bytes moved", f"moved {doc.get('moved')} · deduped {doc.get('deduped')}")
        check(doc.get("uploaded") == 0, "0 objects uploaded", str(doc.get("uploaded")))
        check(int(doc.get("reingested") or 0) > 0,
              "the hub re-verified every held object anyway (law 18)",
              f"reingested {doc.get('reingested')}")
        check(doc.get("snapshot") == small, "and still installed the same root")
        result["dedup_wall"] = wall

        rc2, doc2, r2 = cozy.json(pub, "push", "acme/small", small,
                                  "--family", "sdxl", "--reason", "cl-012 duplicate completion")
        check(rc2 == 0 and doc2.get("duplicate") is True,
              "an exact duplicate completion replays the ledger", f"duplicate={doc2.get('duplicate')}")
        check(doc2.get("catalog_root") == result.get("small", {}).get("catalog_root"),
              "same catalog root id, byte for byte")

    # ------------------------------------------------------------ resume
    if on("resume"):
        section("resume — an interrupted publish converges on a re-run")
        mid, mid_bytes = build_artifact(a.tfs, pub + "/cas", 1, 2, 11)
        note(f"artifact {mid[:19]}… {mid_bytes/2**20:.1f} MiB")
        rc, doc, r = cozy.json(pub, "repo", "create", "acme/resume", "--kind", "model",
                               "--reason", "cl-012 resume arm")
        rc, doc, r = cozy.json(pub, "push", "acme/resume", mid, "--family", "sdxl",
                               "--reason", "cl-012 interrupted publish", "--crash-after", "3")
        check(rc == 1, "the publish was killed mid-upload", f"exit {rc}")
        check("crash_after_upload" in (r.stderr + r.stdout), "and said so typed")

        status, mid_state = hub_get(a.hub, "/v1/repos/acme/resume/checkpoints")
        check(not any(c["snapshot_id"] == mid for c in mid_state.get("checkpoints", [])),
              "no root is visible after the kill", "0 checkpoints")

        rc, doc, r = cozy.json(pub, "push", "acme/resume", mid, "--family", "sdxl",
                               "--reason", "cl-012 resumed publish")
        check(rc == 0, "the re-run completes", f"exit {rc}")
        check(doc.get("deduped") != "0B",
              "and resumes: the objects already uploaded are HELD, not re-sent",
              f"moved {doc.get('moved')} · deduped {doc.get('deduped')}")
        check(doc.get("snapshot") == mid, "exactly one root, and it is the right one")
        # A killed publish can leave a ranged upload open at the backend, and an
        # unaborted one is billed storage no object listing can show. The client does
        # not own that cleanup and must not: the hub's reaper does. What this observes
        # is where the boundary actually falls today.
        status, before = hub_get(a.hub, "/v1/admin/dangling-uploads", token)
        req = urllib.request.Request(a.hub + "/v1/admin/reap", data=b"", method="POST")
        req.add_header("Authorization", "Bearer " + token)
        req.add_header("X-Tensorhub-Reason", "cl-012 resume arm")
        try:
            with urllib.request.urlopen(req, timeout=300) as rr:
                swept = json.loads(rr.read() or b"{}")
        except urllib.error.HTTPError as e:
            swept = json.loads(e.read() or b"{}")
        status, after = hub_get(a.hub, "/v1/admin/dangling-uploads", token)
        n_before = len(before.get("dangling_multipart_uploads") or [])
        n_after = len(after.get("dangling_multipart_uploads") or [])
        check(n_before == n_after,
              "the client leaves no cleanup of its own behind: the hub owns the sweep",
              f"{n_before} dangling before the sweep, {n_after} after · {json.dumps(swept.get('reaped', {}))}")
        if n_after:
            note("FINDING for tensorhub (th-002): a resumed publish re-grants an object whose")
            note("  ranged upload is still open, and `openMultipart` opens a SECOND upload for the")
            note("  same key instead of adopting or aborting the first. The orphan is invisible to")
            note("  any object listing and is billed until publish.multipart_ttl_seconds (24h by")
            note("  default) lets the reaper abort it. Nothing on the client side can see it, and")
            note("  nothing on the client side should: the fix belongs where the upload is opened.")
        result["mid"] = (mid, mid_bytes, doc)

    # ------------------------------------------------------------ pull
    if on("pull"):
        section("pull — hub checkpoint into a fresh local store, verified")
        note("the read grant comes from the shim: th-002 landed no object-read route (see the header)")
        rc, plan, r = cozy.json(dst, "pull", "acme/small", "--dry-run")
        check(rc == 0, "pull --dry-run", f"exit {rc}")
        check(plan.get("kind") == "pull plan", "a plan moves nothing",
              f"{plan.get('objects')} objects / {plan.get('bytes')}")

        t0 = time.perf_counter()
        rc, doc, r = cozy.json(dst, "pull", "acme/small@" + small)
        wall = time.perf_counter() - t0
        check(rc == 0, "pull round trip", f"exit {rc} in {wall:.2f}s")
        if rc != 0:
            note((r.stdout + r.stderr).strip()[:900])
        check(doc.get("snapshot") == small, "the snapshot asked for is the snapshot installed")
        check(int(doc.get("tensors") or 0) > 0, "every declared byte verified",
              f"{doc.get('tensors')} tensors, {doc.get('parts')} parts")
        # The proof, independently: run the byte plane's own whole-checkpoint verify
        # against the puller's store, from outside the CLI.
        rc2, so, se = tfs_run(a.tfs, "snapshot", "verify", dst + "/cas", small.replace("sha256:", ""))
        check(rc2 == 0, "tfs agrees, asked directly", so.strip().splitlines()[-2].strip() if rc2 == 0 else se.strip()[:120])
        rc3, so3, _ = tfs_run(a.tfs, "root", "list", dst + "/cas")
        check("hub_published=true" in so3, "the local root is noted hub_published (durability, not a pin)")
        result["pull_wall"] = wall

    # ------------------------------------------------------------ pull resume
    if on("pullresume"):
        section("pull resume — an interrupted fetch converges from verified state")
        mid = result.get("mid", (None,))[0]
        if not mid:
            mid, mid_bytes = build_artifact(a.tfs, pub + "/cas", 1, 2, 11)
            cozy.json(pub, "repo", "create", "acme/resume", "--kind", "model", "--reason", "cl-012")
            cozy.json(pub, "push", "acme/resume", mid, "--family", "sdxl", "--reason", "cl-012")
        home = str(work / "puller2")
        pathlib.Path(home).mkdir(parents=True, exist_ok=True)
        rc, doc, r = cozy.json(home, "pull", "acme/resume@" + mid, "--crash-after", "4")
        check(rc == 1, "the fetch was killed mid-transfer", f"exit {rc}")
        check("crash_after_fetch" in (r.stderr + r.stdout), "and said so typed")
        rc2, so, _ = tfs_run(a.tfs, "root", "list", home + "/cas")
        check(mid.replace("sha256:", "") not in so or "root " not in so,
              "no local root points at a half-fetched checkpoint")

        rc, doc, r = cozy.json(home, "pull", "acme/resume@" + mid)
        check(rc == 0, "the re-run completes", f"exit {rc}")
        check(doc.get("skipped", 0) > 0,
              "and resumes: already-verified objects are skipped, not re-fetched",
              f"admitted {doc.get('admitted')} · skipped {doc.get('skipped')} · moved {doc.get('moved')} · deduped {doc.get('deduped')}")
        rc2, so, se = tfs_run(a.tfs, "snapshot", "verify", home + "/cas", mid.replace("sha256:", ""))
        check(rc2 == 0, "the converged checkpoint verifies whole")

    # ------------------------------------------------------------ refusals
    if on("refusals"):
        section("refusals — every one a real condition, planted and observed")

        def arm(what, expect_exit, expect_text, home, *args, **kw):
            r = cozy.run(home, *args, **kw)
            said = (r.stderr + r.stdout)
            got = expect_text in said
            check(r.returncode == expect_exit and got, what,
                  f"exit {r.returncode}: " + first(r))

        arm("tokenless push refuses BEFORE the dial", 5, "hub.token_missing",
            pub, "push", "acme/small", small, "--family", "sdxl", "--reason", "x", token=False)
        r = cozy.run(pub, "push", "acme/small", small, "--family", "sdxl", "--reason", "x",
                     token=False, stdin="not-the-token")
        # A wrong credential is the hub's own refusal, verbatim.
        r = subprocess.run([a.cozy, "push", "acme/small", small, "--family", "sdxl",
                            "--reason", "x", "--token-stdin"],
                           env=cozy.env(pub, token=False), input="not-the-token",
                           capture_output=True, text=True)
        PRINTED.append(r.stderr)
        check(r.returncode == 5 and "auth.token_invalid" in r.stderr,
              "a WRONG token gets the hub's own refusal, verbatim",
              first(r))
        check("not-the-token" not in r.stderr and "not-the-token" not in r.stdout,
              "and the credential appears in no byte of the output")

        arm("--token-stdin with empty stdin", 5, "token.empty_stdin",
            pub, "push", "acme/small", small, "--family", "sdxl", "--reason", "x",
            "--token-stdin", token=False, stdin="")
        arm("push without --family", 2, "needs --family",
            pub, "push", "acme/small", small, "--reason", "x")
        arm("push without --reason", 2, "needs --reason",
            pub, "push", "acme/small", small, "--family", "sdxl")
        arm("a PATH is not publishable", 2, "not publishable",
            pub, "push", "acme/small", "./some/dir", "--family", "sdxl", "--reason", "x")
        arm("a malformed snapshot id", 2, "is not a snapshot id",
            pub, "push", "acme/small", "sha256:nope", "--family", "sdxl", "--reason", "x")
        arm("an unknown repo", 4, "not_found",
            pub, "push", "acme/absent", small, "--family", "sdxl", "--reason", "x")
        arm("an unknown checkpoint", 4, "holds no checkpoint",
            dst, "pull", "acme/small@" + "0" * 64)
        arm("a release pinned BY NAME", 2, "th-003",
            dst, "pull", "acme/small@v1")
        arm("an unknown repo on pull", 4, "not_found", dst, "pull", "acme/absent")

        # The family lock: a second family into a locked repo.
        arm("a second model family into a locked repo", 13, "family_locked",
            pub, "push", "acme/small", small, "--family", "wan", "--reason", "cl-012 lock arm")

        # A hub that serves no read plane: point pull at the REAL hub, past the shim.
        # It must run against an EMPTY store — a puller that already holds every object
        # never asks for a read grant, so a warm store would make this arm vacuous.
        empty = str(work / "noreadplane")
        shutil.rmtree(empty, ignore_errors=True)
        pathlib.Path(empty).mkdir(parents=True, exist_ok=True)
        r = cozy.run(empty, "pull", "acme/small@" + small, url=a.hub)
        check(r.returncode == 9 and "hub.no_read_plane" in r.stderr,
              "a hub with no object-read route refuses by NAME", first(r))

        # A source that hands back the wrong bytes. The identity is the digest, so the
        # byte plane must refuse it — nothing enters the store under a name it did not earn.
        rc, listing, _ = cozy.json(dst, "pull", "acme/small@" + small, "--dry-run")
        rc2, so, _ = tfs_run(a.tfs, "cloud", "closure", pub + "/cas", small.replace("sha256:", ""), "s-arm",
                             "--out", str(work / "arm-closure.json"))
        objs = json.loads((work / "arm-closure.json").read_text())["objects"]
        ids = sorted("sha256:" + o["sha256"] for o in objs)
        victim, substitute = ids[-1], ids[0]
        home = str(work / "corrupt")
        shutil.rmtree(home, ignore_errors=True)
        pathlib.Path(home).mkdir(parents=True, exist_ok=True)
        Shim.corrupt = {victim: substitute}
        r = cozy.run(home, "pull", "acme/small@" + small)
        Shim.corrupt = {}
        check(r.returncode == 3 and ("tfs_refused" in r.stderr + r.stdout or
                                     "fetch.object_refused" in r.stderr + r.stdout),
              "a source serving ANOTHER object's real bytes is refused by the byte plane",
              first(r))
        rc3, so3, _ = tfs_run(a.tfs, "root", "list", home + "/cas")
        check("root " not in so3 or small.replace("sha256:", "") not in so3,
              "and no local root points at it")

        # A hub that lies about a manifest. It is admitted under the snapshot id or
        # not at all, so the document proves itself.
        home = str(work / "badmanifest")
        shutil.rmtree(home, ignore_errors=True)
        pathlib.Path(home).mkdir(parents=True, exist_ok=True)
        Shim.swap_manifest = small
        r = cozy.run(home, "pull", "acme/small@" + small)
        Shim.swap_manifest = None
        check(r.returncode != 0, "a mutated manifest never enters the store",
              first(r))

        # A hub that is not there.
        r = cozy.run(pub, "push", "acme/small", small, "--family", "sdxl", "--reason", "x",
                     url="http://127.0.0.1:1")
        check(r.returncode == 9, "an unreachable hub is exit 9",
              first(r))

        # A missing byte plane.
        r = subprocess.run([a.cozy, "pull", "acme/small", "--dry-run"],
                           env={**cozy.env(dst), "COZY_TFS": "tfs-does-not-exist"},
                           capture_output=True, text=True)
        PRINTED.append(r.stderr)
        check(r.returncode == 6 and "tfs_missing" in r.stderr,
              "no tensorfs CLI is a STRUCTURAL refusal naming what to install",
              first(r))

    # ------------------------------------------------------------ bench
    if on("bench"):
        section("bench — nice -n 19, shared box, residential uplink to R2")
        big, big_bytes = build_artifact(a.tfs, pub + "/cas", 1, 1, 23)
        note(f"artifact {big[:19]}… {big_bytes/2**20:.1f} MiB")
        cozy.json(pub, "repo", "create", "acme/bench", "--kind", "model", "--reason", "cl-012 bench")

        t0 = time.perf_counter()
        rc, doc, r = cozy.json(pub, "push", "acme/bench", big, "--family", "sdxl", "--reason", "cl-012 bench push")
        push_wall = time.perf_counter() - t0
        check(rc == 0, "push a large set", f"exit {rc}")
        note(f"push {big_bytes/2**20:.1f} MiB in {push_wall:.2f}s = {big_bytes/2**20/push_wall:.2f} MiB/s")

        t0 = time.perf_counter()
        rc2, doc2, _ = cozy.json(pub, "push", "acme/bench", big, "--family", "sdxl",
                                 "--reason", "cl-012 dedup-hit latency")
        dedup_wall = time.perf_counter() - t0
        note(f"dedup-hit publish (duplicate completion): {dedup_wall*1000:.0f} ms")

        home = str(work / "benchpull")
        shutil.rmtree(home, ignore_errors=True)
        pathlib.Path(home).mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        rc3, doc3, r3 = cozy.json(home, "pull", "acme/bench@" + big)
        pull_wall = time.perf_counter() - t0
        check(rc3 == 0, "pull the same set into an empty store", f"exit {rc3}")
        note(f"pull {big_bytes/2**20:.1f} MiB in {pull_wall:.2f}s = {big_bytes/2**20/pull_wall:.2f} MiB/s")

        t0 = time.perf_counter()
        rc4, doc4, _ = cozy.json(home, "pull", "acme/bench@" + big)
        warm = time.perf_counter() - t0
        check(rc4 == 0 and doc4.get("moved") == "0B",
              "a pull into a store that already holds it moves 0 bytes",
              f"{warm*1000:.0f} ms, moved {doc4.get('moved')}")

    # ------------------------------------------------------------ show
    if on("show"):
        section("show — what a person actually sees")
        cozy.json(pub, "repo", "create", "acme/show", "--kind", "model", "--reason", "cl-012 show")
        shown, _ = build_artifact(a.tfs, pub + "/cas", 1, 4, 41)
        r = cozy.run(pub, "push", "acme/show", shown, "--family", "sdxl",
                     "--reason", "cl-012 human-readable output")
        for line in r.stdout.rstrip().splitlines():
            out("  | " + line)
        check(r.returncode == 0, "push, rendered", f"exit {r.returncode}")
        home = str(work / "show")
        shutil.rmtree(home, ignore_errors=True)
        pathlib.Path(home).mkdir(parents=True, exist_ok=True)
        r = cozy.run(home, "pull", "acme/show@" + shown)
        for line in r.stdout.rstrip().splitlines():
            out("  | " + line)
        check(r.returncode == 0, "pull, rendered", f"exit {r.returncode}")

    # ------------------------------------------------------------ secrecy
    section("secrecy")
    body = "\n".join(PRINTED)
    check(token not in body, "the raw admin token appears in NO byte this session printed",
          f"{len(body)} bytes checked")

    section("verdict")
    out(f"  {PASS} checks, {FAIL} failed")
    server.shutdown()
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
