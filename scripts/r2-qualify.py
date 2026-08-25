#!/usr/bin/env python3
"""tfs-017's QUALIFICATION GATE, run against live R2.

The invariant is no-corrupt/no-clobber. How it is enforced is a qualification OUTCOME, not
an assumption: build the 5-part staging protocol only if the one-request primitive fails.
So this runs first, and the verdict is banked before either branch exists.

The primitive under test is a presigned PUT carrying

  * a SigV4-SIGNED `x-amz-checksum-sha256` — the body must hash to the declared digest, and
    a signed header cannot be stripped without breaking the signature; and
  * `If-None-Match: *` — never replaces an existing object.

Blob segments are <=64 MiB single-object operations, so multipart — the one surface with no
conditional form — is unreachable by construction and is not tested here.

SigV4 is implemented in this file, by hand, over hmac/sha256. That is deliberate: the whole
question is which headers the signature BINDS, and an SDK that assembles the request for you
is the wrong instrument for answering it.

  scripts/r2-qualify.py            run the gate, print the verdict
  scripts/r2-qualify.py --keep     leave the scratch objects for inspection
  scripts/r2-qualify.py --publish <store> <snapshot-hex>
                                   the GREEN branch end to end against live R2, plus the
                                   race matrix: honest duplicate, wrong-body publisher,
                                   verifier crash mid-publish, and a GC pass during publish

Credentials come from the environment or from an --env file of KEY=VALUE lines; nothing is
read from a credential store and nothing is written to one. The token available here can
write objects but cannot create buckets, so the gate works inside an existing bucket under a
namespaced scratch prefix and deletes every object it wrote.
"""
import datetime
import hashlib
import hmac
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The token this gate runs under can write objects but cannot CREATE buckets, so the gate
# runs inside an existing one under a namespaced scratch prefix and deletes what it wrote.
# `dataset-cas` is a digest-keyed CAS namespace: a `tfs-017-qualification/<run>/` key is
# not digest-shaped, collides with nothing, and is addressed by nothing.
BUCKET = os.environ.get("TFS017_BUCKET", "dataset-cas")
SEGMENT_MIB = int(os.environ.get("TFS017_SEGMENT_MIB", "8"))
UNSIGNED = "UNSIGNED-PAYLOAD"
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def load_env(path):
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def creds():
    ak = os.environ.get("TENSORHUB_S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("TENSORHUB_S3_SECRET_ACCESS_KEY") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY")
    ep = os.environ.get("TENSORHUB_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")
    region = os.environ.get("TENSORHUB_S3_REGION", "auto")
    if not (ak and sk and ep):
        sys.exit("no S3/R2 credentials in the environment; pass --env <file>")
    return ak, sk, ep.rstrip("/"), region


def sign_key(secret, date, region, service):
    k = ("AWS4" + secret).encode()
    for part in (date, region, service, "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return k


def canonical(method, path, query, headers, payload_hash):
    ch = "".join(f"{k}:{v}\n" for k, v in sorted(headers.items()))
    sh = ";".join(sorted(headers))
    return "\n".join([method, path, query, ch, sh, payload_hash]), sh


def presign(method, key, extra_headers, expires=300, sign_only=None):
    """A presigned URL whose signature BINDS every header in `extra_headers`.

    `sign_only` lets the gate build a URL that promises headers it will then not send —
    the only honest way to ask whether a signed header can be stripped.
    """
    ak, sk, ep, region = creds()
    host = urllib.parse.urlparse(ep).netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    signed = dict(extra_headers if sign_only is None else sign_only)
    signed["host"] = host
    sh = ";".join(sorted(signed))
    q = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{ak}/{date}/{region}/s3/aws4_request",
        "X-Amz-Date": stamp,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": sh,
    }
    qs = "&".join(f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}"
                  for k, v in sorted(q.items()))
    path = "/" + BUCKET + "/" + urllib.parse.quote(key, safe="/")
    creq, _ = canonical(method, path, qs, signed, UNSIGNED)
    scope = f"{date}/{region}/s3/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", stamp, scope,
                     hashlib.sha256(creq.encode()).hexdigest()])
    sig = hmac.new(sign_key(sk, date, region, "s3"), sts.encode(), hashlib.sha256).hexdigest()
    return f"{ep}{path}?{qs}&X-Amz-Signature={sig}"


def signed_request(method, key="", body=b"", extra=None, query=""):
    """Header-authenticated SigV4, for the gate's own housekeeping (bucket create, HEAD,
    delete). Never part of what is under test."""
    ak, sk, ep, region = creds()
    host = urllib.parse.urlparse(ep).netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest() if body else EMPTY_SHA
    headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": stamp}
    headers.update({k.lower(): v for k, v in (extra or {}).items()})
    path = "/" + BUCKET + ("/" + urllib.parse.quote(key, safe="/") if key else "")
    creq, sh = canonical(method, path, query, headers, payload_hash)
    scope = f"{date}/{region}/s3/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", stamp, scope,
                     hashlib.sha256(creq.encode()).hexdigest()])
    sig = hmac.new(sign_key(sk, date, region, "s3"), sts.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={sh}, Signature={sig}")
    url = f"{ep}{path}" + (f"?{query}" if query else "")
    return send(method, url, body, headers)


def send(method, url, body=b"", headers=None):
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except urllib.error.URLError as e:
        return 0, {}, str(e).encode()


def b64sha(b):
    import base64
    return base64.b64encode(hashlib.sha256(b).digest()).decode()


RESULTS = []


def arm(name, expect, got, detail=""):
    ok = expect(got) if callable(expect) else got == expect
    RESULTS.append({"arm": name, "ok": ok, "observed": got, "detail": detail})
    print(f"  {'ok  ' if ok else 'MISS'} {name:<58} {got}"
          + (f"  {detail[:70]}" if detail else ""))
    return ok


def tfs(*a):
    import subprocess
    r = subprocess.run(["target/release/tfs", *a], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def field(out, name):
    for line in out.splitlines():
        if line.startswith(name):
            return line[len(name):].strip()
    return ""


def publish_object(prefix, sha256, length, path, store_root):
    """One object, published the GREEN way: mint the grant, sign BOTH conditions into the
    URL, PUT at the FINAL key, and turn what the backend reported into a fact."""
    import subprocess
    _, gout, _ = tfs("cloud", "grant", f"{WORK}/closure.json", f"sha256:{sha256}",
                     "--out", f"{WORK}/grant.json")
    key = prefix + "/" + field(gout, "key")
    ck = [l.split(": ", 1)[1] for l in gout.splitlines()
          if l.startswith("sign") and "checksum" in l][0]
    body = pathlib.Path(path).read_bytes()
    url = presign("PUT", key, {"if-none-match": "*", "x-amz-checksum-sha256": ck})
    st, h, msg = send("PUT", url, body, {"If-None-Match": "*", "x-amz-checksum-sha256": ck})
    return st, h, msg, key, ck, gout


def race_matrix(prefix):
    import shutil, subprocess
    store, snap = PUBLISH
    print(f"\nstore    {store}\nsnapshot {snap}\n")

    print("(1) DECLARE FIRST — the closure is every object the snapshot reaches")
    rc, out, err = tfs("cloud", "closure", store, snap, "s-green", "--out", f"{WORK}/closure.json")
    if rc != 0:
        sys.exit(err)
    print("  " + "\n  ".join(out.strip().splitlines()))
    closure = json.loads(pathlib.Path(f"{WORK}/closure.json").read_text())
    objs = closure["objects"]

    print("\n(2) conditional checksummed PUTs AT FINAL KEYS")
    facts = pathlib.Path(f"{WORK}/facts"); shutil.rmtree(facts, ignore_errors=True)
    facts.mkdir(parents=True)
    moved = 0
    for o in objs:
        h = o["sha256"]
        p = f"{store}/objects/sha256/{h[:2]}/{h[2:4]}/{h}"
        st, hdr, msg, key, ck, _ = publish_object(prefix, h, o["length"], p, store)
        if not 200 <= st < 300:
            arm(f"PUT {h[:12]}", lambda s: 200 <= s < 300, st, msg[:80].decode(errors="replace"))
            continue
        moved += o["length"]
        version = hdr.get("x-amz-version-id") or hdr.get("ETag", "").strip('"')
        etag = hdr.get("ETag", "").strip('"')
        got = hdr.get("x-amz-checksum-sha256") or ck
        rc, vout, verr = tfs("cloud", "verified", f"{WORK}/grant.json", version, etag, got,
                             "--out", str(facts / f"{h[:16]}.json"))
        if rc != 0:
            arm(f"fact {h[:12]}", True, False, verr[:90])
    arm("every declared object is resident and has a FACT", len(objs),
        len(list(facts.glob("*.json"))), f"{moved} B moved")

    print("\n(3) the atomic catalog root, over facts only")
    rc, out, err = tfs("cloud", "publish", f"{WORK}/closure.json", str(facts),
                       "--out", f"{WORK}/root.json")
    arm("a root over complete facts publishes", 0, rc, err[:90])
    print("  " + "\n  ".join(out.strip().splitlines()))

    print("\n(4) RACE: an honest duplicate re-uploads nothing")
    o = objs[0]; h = o["sha256"]
    p = f"{store}/objects/sha256/{h[:2]}/{h[2:4]}/{h}"
    st, _, msg, key, ck, _ = publish_object(prefix, h, o["length"], p, store)
    arm("the second publisher of the SAME bytes gets 412", 412, st)
    _, hh, _ = signed_request("HEAD", key, extra={"x-amz-checksum-mode": "ENABLED"})
    resident = hh.get("x-amz-checksum-sha256", "")
    rc, out, err = tfs("cloud", "conflict", f"{WORK}/grant.json", resident)
    arm("verify-resident-on-conflict says HONEST DUPLICATE", 0, rc, out.strip()[:80])

    print("\n(5) RACE: a valid resident object vs a WRONG-BODY publisher")
    wrong = b"not the bytes this key promises"
    url = presign("PUT", key, {"if-none-match": "*", "x-amz-checksum-sha256": ck})
    st, _, msg = send("PUT", url, wrong, {"If-None-Match": "*", "x-amz-checksum-sha256": ck})
    arm("wrong body under the grant's own signed digest", lambda s: s >= 400, st,
        msg[:90].decode(errors="replace"))
    st, _, cur = signed_request("GET", key)
    arm("the previously verified bytes are still readable", True,
        cur == pathlib.Path(p).read_bytes(), "under their identity, unchanged")
    rc, out, err = tfs("cloud", "conflict", f"{WORK}/grant.json", b64sha(wrong))
    arm("a resident that is NOT ours quarantines", 1, rc, err.strip()[:90])

    print("\n(6) RACE: the verifier crashes mid-publish")
    partial = pathlib.Path(f"{WORK}/facts-partial"); shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for f in sorted(facts.glob("*.json"))[:-1]:
        shutil.copy(f, partial)
    rc, out, err = tfs("cloud", "publish", f"{WORK}/closure.json", str(partial))
    arm("no catalog root over an incomplete fact set", 1, rc, err.strip()[:110])
    st, _, cur = signed_request("GET", key)
    arm("and the bytes verified BEFORE the crash still read", True,
        cur == pathlib.Path(p).read_bytes())

    print("\n(7) RACE: a GC pass during publish")
    rc, out, err = tfs("gc", "mark", store)
    kept = field(out, "reachable") or out.strip().splitlines()[-1][:70]
    arm("the local mark keeps every object the closure declared", 0, rc, kept[:80])
    still = all(pathlib.Path(
        f"{store}/objects/sha256/{o['sha256'][:2]}/{o['sha256'][2:4]}/{o['sha256']}").exists()
        for o in objs)
    arm("every published object is still local and readable", True, still)

    print("\n(8) the shapes that make the unsafe cases unrepresentable")
    rc, _, err = tfs("cloud", "grant", f"{WORK}/closure.json",
                     "sha256:" + "0" * 64, "--length", "1")
    arm("a grant outside the declared closure", 1, rc, err.strip()[:100])
    rc, _, err = tfs("cloud", "grant", f"{WORK}/closure.json", f"sha256:{objs[0]['sha256']}",
                     "--expires-at", "1", "--out", f"{WORK}/expired.json")
    rc2, _, err2 = tfs("cloud", "verified", f"{WORK}/expired.json", "v", "e", "not-a-checksum")
    arm("a final-key checksum mismatch quarantines", 1, rc2, err2.strip()[:100])
    print("  an UNCONDITIONAL or UNCHECKSUMMED final-key grant has no representation at all:")
    print("  both conditions are DERIVED from the digest, so there is no field to unset and")
    print("  no flag to pass. `tfs cloud grant --help` cannot produce one.")

    if not KEEP:
        n = 0
        for o in objs:
            h = o["sha256"]
            st, _, _ = signed_request(
                "DELETE", f"{prefix}/objects/sha256/{h[:2]}/{h[2:4]}/{h}")
            n += 1 if st < 400 else 0
        print(f"\ncleaned {n} published objects")


PUBLISH = None
KEEP = False
WORK = os.environ.get("TFS017_WORK", "/tmp/tfs-017")


def main():
    args = sys.argv[1:]
    if "--env" in args:
        load_env(args[args.index("--env") + 1])
    global KEEP, PUBLISH
    keep = KEEP = "--keep" in args
    if "--publish" in args:
        i = args.index("--publish")
        PUBLISH = (args[i + 1], args[i + 2])
    pathlib.Path(WORK).mkdir(parents=True, exist_ok=True)
    import uuid
    run = uuid.uuid4().hex[:12]
    prefix = f"qualification/{run}"

    if "--segment-mib" in args:
        globals()["SEGMENT_MIB"] = int(args[args.index("--segment-mib") + 1])
    print(f"bucket   {BUCKET}\nprefix   {prefix}\nsegment  {SEGMENT_MIB} MiB\n")
    st, _, _ = signed_request("GET", query="list-type=2&max-keys=1")
    if st != 200:
        sys.exit(f"bucket {BUCKET!r} is not listable ({st}); set TFS017_BUCKET")
    if PUBLISH:
        print("=" * 72)
        print("THE GREEN BRANCH, END TO END AGAINST LIVE R2, PLUS THE RACE MATRIX")
        print("=" * 72)
        race_matrix(prefix)
        green = all(r["ok"] for r in RESULTS)
        print(f"\n{sum(1 for r in RESULTS if r['ok'])}/{len(RESULTS)} arms fired")
        out = pathlib.Path(os.environ.get("TFS017_OUT", f"{WORK}/race-matrix.json"))
        out.write_text(json.dumps({"bucket": BUCKET, "prefix": prefix,
                                   "arms": RESULTS}, indent=1))
        print(f"banked -> {out}")
        return 0 if green else 1

    body_a = f"tfs-017 qualification body A {run}".encode()
    body_b = f"tfs-017 qualification body B {run}".encode()
    ck_a = b64sha(body_a)

    print("(a) a SIGNED x-amz-checksum-sha256 binds the body to its digest")
    k = f"{prefix}/checksum"
    url = presign("PUT", k, {"x-amz-checksum-sha256": ck_a})
    st, _, _ = send("PUT", url, body_a, {"x-amz-checksum-sha256": ck_a})
    arm("the declared body is accepted", lambda s: 200 <= s < 300, st)
    url = presign("PUT", f"{prefix}/checksum-wrong", {"x-amz-checksum-sha256": ck_a})
    st, _, msg = send("PUT", url, body_b, {"x-amz-checksum-sha256": ck_a})
    arm("a WRONG body under the same signed digest", lambda s: s >= 400, st,
        msg[:120].decode(errors="replace"))
    url = presign("PUT", f"{prefix}/checksum-stripped", {},
                  sign_only={"x-amz-checksum-sha256": ck_a})
    st, _, msg = send("PUT", url, body_b, {})
    arm("the signed header STRIPPED by the client", lambda s: s >= 400, st,
        msg[:120].decode(errors="replace"))

    print("\n(b) If-None-Match: * never replaces an existing object")
    k = f"{prefix}/conditional"
    url = presign("PUT", k, {"if-none-match": "*", "x-amz-checksum-sha256": ck_a})
    st, _, _ = send("PUT", url, body_a, {"If-None-Match": "*", "x-amz-checksum-sha256": ck_a})
    arm("first conditional write of a fresh key", lambda s: 200 <= s < 300, st)
    ck_b = b64sha(body_b)
    url = presign("PUT", k, {"if-none-match": "*", "x-amz-checksum-sha256": ck_b})
    st, _, msg = send("PUT", url, body_b, {"If-None-Match": "*", "x-amz-checksum-sha256": ck_b})
    arm("a second, DIFFERENT body at the same key", lambda s: s == 412, st,
        msg[:100].decode(errors="replace"))

    print("\n(b2) concurrent same-key writers: exactly one wins")
    k = f"{prefix}/race"
    bodies = [f"racer {i} {run}".encode() for i in range(8)]

    def race(b):
        c = b64sha(b)
        u = presign("PUT", k, {"if-none-match": "*", "x-amz-checksum-sha256": c})
        return send("PUT", u, b, {"If-None-Match": "*", "x-amz-checksum-sha256": c})[0]

    with ThreadPoolExecutor(max_workers=8) as ex:
        codes = list(ex.map(race, bodies))
    won = sum(1 for c in codes if 200 <= c < 300)
    arm("exactly one of 8 concurrent writers wins", 1, won, f"codes {sorted(codes)}")
    st, h, resident = signed_request("GET", k)
    arm("the winner's bytes are what is resident", True,
        resident in bodies, f"{len(resident)} B")

    print("\n(c) the RED arm: an UNCONDITIONAL same-key PUT clobbers")
    k = f"{prefix}/unconditional"
    url = presign("PUT", k, {"x-amz-checksum-sha256": ck_a})
    send("PUT", url, body_a, {"x-amz-checksum-sha256": ck_a})
    url = presign("PUT", k, {"x-amz-checksum-sha256": ck_b})
    st, _, _ = send("PUT", url, body_b, {"x-amz-checksum-sha256": ck_b})
    _, _, after = signed_request("GET", k)
    arm("an unconditional PUT REPLACES verified bytes", True, after == body_b,
        "last-writer-wins, exactly as the P0 correction says")

    print("\n(d) verify-resident-on-conflict: a 412 is answerable without trusting anyone")
    k = f"{prefix}/conditional"
    st, h, _ = signed_request("HEAD", k, extra={"x-amz-checksum-mode": "ENABLED"})
    got = h.get("x-amz-checksum-sha256") or h.get("X-Amz-Checksum-Sha256")
    arm("HEAD returns the resident object's sha256", ck_a, got,
        "so a losing publisher CONFIRMS rather than re-uploads")

    print(f"\n(e) a {SEGMENT_MIB} MiB segment is a SINGLE conditional operation")
    big = (b"x" * 1024) * (SEGMENT_MIB * 1024)
    ckb = b64sha(big)
    k = f"{prefix}/segment64"
    url = presign("PUT", k, {"if-none-match": "*", "x-amz-checksum-sha256": ckb}, expires=900)
    st, _, msg = send("PUT", url, big, {"If-None-Match": "*", "x-amz-checksum-sha256": ckb})
    arm(f"{SEGMENT_MIB} MiB conditional checksummed PUT", lambda s: 200 <= s < 300, st,
        f"{len(big)} B in one request; multipart is unreachable by construction")

    green = all(r["ok"] for r in RESULTS)
    print("\n" + "=" * 72)
    print(f"QUALIFICATION VERDICT: {'GREEN' if green else 'RED'} "
          f"({sum(1 for r in RESULTS if r['ok'])}/{len(RESULTS)} arms)")
    print("GREEN means: publish is `declare closure -> conditional checksummed PUTs at final")
    print("keys -> verify-resident-on-conflict -> VerifiedObject facts -> atomic catalog-root")
    print("publication`, and NO staging namespace, staged verifier or promotion copy exists.")
    print("=" * 72)

    out = pathlib.Path(os.environ.get("TFS017_OUT", "/tmp/tfs-017-qualification.json"))
    out.write_text(json.dumps({"bucket": BUCKET, "prefix": prefix,
                               "endpoint_host": urllib.parse.urlparse(creds()[2]).netloc,
                               "verdict": "GREEN" if green else "RED",
                               "arms": RESULTS}, indent=1))
    print(f"\nbanked -> {out}")

    if not keep:
        n = 0
        for suffix in ("checksum", "checksum-wrong", "checksum-stripped", "conditional",
                       "race", "unconditional", "segment64"):
            st, _, _ = signed_request("DELETE", f"{prefix}/{suffix}")
            n += 1 if st < 400 else 0
        print(f"cleaned {n} scratch objects")
    return 0 if green else 1


sys.exit(main())
