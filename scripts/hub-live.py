#!/usr/bin/env python3
"""cl-011 live verification: drive the REAL `cozy` binary against a REAL tensorhub.

Not a test suite (decisions.md #160) — an orchestration script that runs the shipped
binary against a running hub and prints what it observed. Every refusal arm is a real
condition (a wrong token, an absent model, a closed port, a hub that never answers),
never a mock.

    scripts/hub-live.py --cozy ./cozy --hub http://127.0.0.1:18099 --token <admin token>
    scripts/hub-live.py ... --only refusals

Sections: reads · writes · refusals · transport · secrecy · bench.
The secrecy section is cumulative: it asserts the raw token never appeared in ANY
byte this session printed.
"""
import argparse
import http.server
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time

CHECKS = []          # (section, name, ok, detail)
TRANSCRIPT = []      # every byte every command printed, for the secrecy check


def run(cozy, args, env=None, home=None):
    e = dict(os.environ)
    e.pop("TENSORHUB_TOKEN", None)
    e["COZY_HOME"] = home
    e.update(env or {})
    p = subprocess.run([cozy] + args, capture_output=True, text=True, env=e)
    TRANSCRIPT.append(p.stdout)
    TRANSCRIPT.append(p.stderr)
    return p


def check(section, name, ok, detail=""):
    CHECKS.append((section, name, ok, detail))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail and not ok else ""))


def expect(section, name, p, code, has=(), lacks=()):
    out = p.stdout + p.stderr
    problems = []
    if p.returncode != code:
        problems.append(f"exit {p.returncode}, wanted {code}")
    for s in has:
        if s not in out:
            problems.append(f"missing {s!r}")
    for s in lacks:
        if s in out:
            problems.append(f"leaked {s!r}")
    check(section, name, not problems, "; ".join(problems) + f" :: {out.strip()[:300]}")
    return p


# ---------------------------------------------------------------- sections

def reads(a, home):
    """Public catalog reads: no credential anywhere in the environment."""
    expect("reads", "hub status (reachable)",
           run(a.cozy, ["hub", "status"], {"TENSORHUB_URL": a.hub}, home),
           0, has=["reachable: true", "status:", "url:", "no accounts at Launch 1"])
    expect("reads", "model search (tokenless)",
           run(a.cozy, ["model", "search"], {"TENSORHUB_URL": a.hub}, home),
           0, has=["ref", "models:", "results:"])
    expect("reads", "endpoint search (tokenless)",
           run(a.cozy, ["endpoint", "search"], {"TENSORHUB_URL": a.hub}, home),
           0, has=["results:"])
    p = expect("reads", "model search --json is one document",
               run(a.cozy, ["model", "search", "--json"], {"TENSORHUB_URL": a.hub}, home), 0)
    try:
        doc = json.loads(p.stdout)
        check("reads", "model search --json parses",
              doc.get("kind") == "model search" and "rows" in doc, str(doc)[:200])
    except Exception as exc:  # noqa: BLE001
        check("reads", "search --json parses", False, str(exc))
    expect("reads", "search miss is exit 0 with an empty state",
           run(a.cozy, ["model", "search", "zzzz-no-such-model"], {"TENSORHUB_URL": a.hub}, home),
           0, has=["0 results for"])
    expect("reads", "model show resolves through the typed route",
           run(a.cozy, ["model", "show", a.model], {"TENSORHUB_URL": a.hub}, home),
           0, has=["model", "ref:"])
    expect("reads", "the generic search command is absent",
           run(a.cozy, ["search"], {"TENSORHUB_URL": a.hub}, home),
           2, has=["unknown command"])


def writes(a, home):
    """First-party writes under the ONE static admin token."""
    name = f"cozy/cl011-live-{int(time.time())}"
    env = {"TENSORHUB_URL": a.hub, "TENSORHUB_TOKEN": a.token}
    expect("writes", "model create",
           run(a.cozy, ["model", "create", name,
                        "--reason", "cl-011 live verification"], env, home),
           0, has=["ref:", name, "admin audit"], lacks=[a.token])
    expect("writes", "the created model reads back publicly",
           run(a.cozy, ["model", "show", name], {"TENSORHUB_URL": a.hub}, home),
           0, has=[name])
    expect("writes", "duplicate model create is the hub's typed conflict",
           run(a.cozy, ["model", "create", name,
                        "--reason", "cl-011 conflict arm"], env, home),
           13, has=["already exists"])
    endpoint = name + "-endpoint"
    expect("writes", "endpoint create",
           run(a.cozy, ["endpoint", "create", endpoint,
                        "--reason", "cl-011 live verification"], env, home),
           0, has=["ref:", endpoint, "admin audit"], lacks=[a.token])
    expect("writes", "the created endpoint reads back publicly",
           run(a.cozy, ["endpoint", "show", endpoint], {"TENSORHUB_URL": a.hub}, home),
           0, has=[endpoint])
    expect("writes", "model show refuses an endpoint name",
           run(a.cozy, ["model", "show", endpoint], {"TENSORHUB_URL": a.hub}, home),
           4, has=["not_found"])
    expect("writes", "endpoint show refuses a model name",
           run(a.cozy, ["endpoint", "show", name], {"TENSORHUB_URL": a.hub}, home),
           4, has=["not_found"])
    expect("writes", "hub config renders provenance, secrets digested",
           run(a.cozy, ["hub", "config"], env, home),
           0, has=["admin.token", "sha256:", "source"], lacks=[a.token])
    p = run(a.cozy, ["hub", "status"], env, home)
    TRANSCRIPT.append(p.stdout)
    digest_local = next((l.split()[1] for l in p.stdout.splitlines() if l.startswith("token:")), "")
    q = run(a.cozy, ["hub", "config", "--json"], env, home)
    try:
        rows = json.loads(q.stdout)["rows"]
        digest_hub = next(r["value"] for r in rows if r["key"] == "admin.token")
        check("writes", "the local token digest equals the hub's admin.token digest",
              digest_local == digest_hub and digest_local.startswith("sha256:"),
              f"{digest_local} vs {digest_hub}")
    except Exception as exc:  # noqa: BLE001
        check("writes", "digest comparison", False, str(exc))


def refusals(a, home):
    """Every arm a real condition; the hub's own envelope reaches the user verbatim."""
    expect("refusals", "tokenless write refuses BEFORE the dial (exit 5)",
           run(a.cozy, ["model", "create", "cozy/never", "--reason", "x"],
               {"TENSORHUB_URL": a.hub}, home),
           5, has=["hub.token_missing", "TENSORHUB_TOKEN"])
    expect("refusals", "tokenless admin READ refuses (exit 5)",
           run(a.cozy, ["hub", "config"], {"TENSORHUB_URL": a.hub}, home),
           5, has=["hub.token_missing"])
    expect("refusals", "wrong token: the hub's 401 rendered verbatim (exit 5)",
           run(a.cozy, ["model", "create", "cozy/never", "--reason", "x"],
               {"TENSORHUB_URL": a.hub, "TENSORHUB_TOKEN": "not-the-configured-token"}, home),
           5, has=["auth.token_invalid", "the presented admin token is not the configured one"],
           lacks=[a.token])
    expect("refusals", "unknown model is typed not-found (exit 4)",
           run(a.cozy, ["model", "show", "cozy/no-such-model"], {"TENSORHUB_URL": a.hub}, home),
           4, has=["not_found", "no model"])
    expect("refusals", "a release pin refuses by name until th-003",
           run(a.cozy, ["model", "show", "cozy/sdxl@v1"], {"TENSORHUB_URL": a.hub}, home),
           2, has=["th-003"])
    expect("refusals", "model create without --reason refuses (exit 2)",
           run(a.cozy, ["model", "create", "cozy/never"],
               {"TENSORHUB_URL": a.hub, "TENSORHUB_TOKEN": a.token}, home),
           2, has=["--reason"])
    expect("refusals", "typed create rejects the retired --kind discriminator",
           run(a.cozy, ["model", "create", "cozy/never", "--kind", "model", "--reason", "x"],
               {"TENSORHUB_URL": a.hub, "TENSORHUB_TOKEN": a.token}, home),
           2, has=["unknown flag", "--kind"])
    expect("refusals", "the generic repo command is absent",
           run(a.cozy, ["repo", "show", "cozy/never"], {"TENSORHUB_URL": a.hub}, home),
           2, has=["unknown command"])
    expect("refusals", "login is deferred, and says why",
           run(a.cozy, ["login"], {"TENSORHUB_URL": a.hub}, home),
           2, has=["not_implemented", "th-031"])
    expect("refusals", "logout is deferred with it",
           run(a.cozy, ["logout"], {"TENSORHUB_URL": a.hub}, home),
           2, has=["th-031"])


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Deaf(threading.Thread):
    """Accepts a connection and never answers: the deadline arm, not the dead one."""
    def __init__(self, port):
        super().__init__(daemon=True)
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(4)
        self.held = []

    def run(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self.held.append(c)


def transport(a, home):
    """Failures that never become an HTTP answer, plus answers that are not ours."""
    dead = free_port()
    expect("transport", "hub down: a read is typed unavailable (exit 9)",
           run(a.cozy, ["model", "search"], {"TENSORHUB_URL": f"http://127.0.0.1:{dead}"}, home),
           9, has=["unavailable", "unreachable", "connection refused"])
    expect("transport", "hub down: a write is typed unavailable (exit 9)",
           run(a.cozy, ["model", "create", "a/b", "--reason", "x"],
               {"TENSORHUB_URL": f"http://127.0.0.1:{dead}", "TENSORHUB_TOKEN": a.token}, home),
           9, has=["unreachable"])
    expect("transport", "hub down: `hub status` REPORTS it, exit 0",
           run(a.cozy, ["hub", "status"], {"TENSORHUB_URL": f"http://127.0.0.1:{dead}"}, home),
           0, has=["reachable: false", "unreachable"])
    expect("transport", "unresolvable host is typed unavailable (exit 9)",
           run(a.cozy, ["model", "search"], {"TENSORHUB_URL": "http://cl011.invalid:1"}, home),
           9, has=["no such host"])

    # A hub that is up but not serving: accepted, then silence.
    deaf = Deaf(free_port())
    deaf.start()
    t0 = time.time()
    expect("transport", "a hub that never answers is a DEADLINE (exit 10), not unavailable",
           run(a.cozy, ["model", "search"], {"TENSORHUB_URL": f"http://127.0.0.1:{deaf.sock.getsockname()[1]}"}, home),
           10, has=["did not answer within"])
    check("transport", "the deadline arm actually waited", time.time() - t0 > 5,
          f"{time.time() - t0:.1f}s")
    deaf.sock.close()

    # Something that answers HTTP but is not a tensorhub.
    root = pathlib.Path(tempfile.mkdtemp(prefix="cl011-notahub-"))
    (root / "v1").mkdir()
    (root / "v1" / "models").write_text("<html>please log in</html>")
    port = free_port()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=str(root), **kw)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    expect("transport", "a 200 that is not our document is refused, not parsed",
           run(a.cozy, ["model", "search"], {"TENSORHUB_URL": f"http://127.0.0.1:{port}"}, home),
           1, has=["hub.unreadable_answer"])
    expect("transport", "an untyped 404 is reported as untyped, not invented",
           run(a.cozy, ["model", "show", "a/b"], {"TENSORHUB_URL": f"http://127.0.0.1:{port}/nope"}, home),
           4, has=["hub.untyped_refusal"])
    srv.shutdown()


def secrecy(a, home):
    """The credential never appeared in anything this session printed."""
    blob = "".join(TRANSCRIPT)
    check("secrecy", f"the raw token appears in none of the {len(blob)} bytes printed this session",
          a.token not in blob)
    p = run(a.cozy, ["hub", "status"], {"TENSORHUB_URL": a.hub, "TENSORHUB_TOKEN": a.token}, home)
    check("secrecy", "`hub status` renders the credential as a digest",
          "sha256:" in p.stdout and a.token not in p.stdout, p.stdout[:200])
    # A token can only enter through the environment: there is no flag that takes one.
    p = run(a.cozy, ["model", "create", "a/b", "--token", a.token], {"TENSORHUB_URL": a.hub}, home)
    check("secrecy", "no verb accepts a credential on argv",
          p.returncode == 2 and "unknown flag" in (p.stdout + p.stderr), p.stderr[:200])


def bench(a, home):
    """Wall clock per invocation. The /bin/true row is this harness's own floor —
    subtract it to read what cozy costs, and never quote a row without it."""
    env = {"TENSORHUB_URL": a.hub, "TENSORHUB_TOKEN": a.token}
    n, t0 = 30, time.time()
    for _ in range(n):
        subprocess.run(["/bin/true"], capture_output=True)
    print(f"  {(time.time() - t0) * 1000 / n:7.2f} ms/run  /bin/true (harness floor)  (n={n})")
    for name, args in (("cozy version (no hub call)", ["version"]),
                       ("cozy hub status", ["hub", "status"]),
                       ("cozy model search", ["model", "search"]),
                       ("cozy model show", ["model", "show", a.model])):
        n, t0 = 30, time.time()
        for _ in range(n):
            run(a.cozy, args, env, home)
        ms = (time.time() - t0) * 1000 / n
        print(f"  {ms:7.2f} ms/run  {name}  (n={n})")


SECTIONS = {"reads": reads, "writes": writes, "refusals": refusals,
            "transport": transport, "secrecy": secrecy, "bench": bench}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cozy", default="./cozy")
    ap.add_argument("--hub", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--model", default="cozy/sdxl", help="a model the hub already holds")
    ap.add_argument("--only", action="append", choices=sorted(SECTIONS))
    a = ap.parse_args()
    a.cozy = str(pathlib.Path(a.cozy).resolve())

    home = tempfile.mkdtemp(prefix="cl011-home-")
    for name in (a.only or ["reads", "writes", "refusals", "transport", "secrecy", "bench"]):
        print(f"\n[{name}]")
        SECTIONS[name](a, home)

    failed = [c for c in CHECKS if not c[2]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
          + (f", {len(failed)} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
