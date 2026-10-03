#!/usr/bin/env python3
"""The shared canonical corpus, run through the PYTHON FACADE (tfs-007 / tfs-013).

Same vectors and frozen `.verdict` files as `cargo test --test vectors`, so agreement is the
cross-language proof. There is exactly ONE codec underneath (that is the
design), so a divergence here is a defect in the marshal boundary, never a profile
difference: a refusal that arrives untyped, an identity that does not survive the FFI, a
document kind the binding cannot reach.

    scripts/facade-conform.py [vectors-dir]
"""

from __future__ import annotations

import pathlib
import sys
import time

import tensorfs
from tensorfs import errors

ROOT = pathlib.Path(__file__).resolve().parent.parent
VEC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "vectors"


def verdict(path: pathlib.Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k] = v
    return out


def truncate(s: str, n: int) -> str:
    return s if len(s.encode()) <= n else s.encode()[:n].decode(errors="ignore") + "…"


def main() -> int:
    started = time.time()
    closure = tensorfs.Closure()
    for p in sorted((VEC / "closure").iterdir()):
        closure.insert(p.read_bytes())

    cases = sorted(
        [*(VEC / "cases").glob("*/*.json"), *(VEC / "cases").glob("*/*.cbor")]
    )
    outcomes: dict[str, dict[str, str | None]] = {}
    refusals: dict[str, str] = {}
    npass = nfail = 0

    for f in cases:
        name = f"{f.parent.name}/{f.stem}"
        v = verdict(f.with_suffix(".verdict"))
        expect = v["expect"]
        data = f.read_bytes()
        ok: dict[str, str | None] | None = None
        code = detail = ""
        try:
            ok = tensorfs.run_doc(v["doc"], data, closure)
            outcomes[name] = ok
        except errors.Refusal as e:
            # The refusal must be TYPED: the class is the code, not a parsed string.
            code, detail = e.code, e.detail
            assert type(e).code == code, f"{name}: class/instance code disagree"
            refusals[name] = code

        if expect == "ok":
            if ok is None:
                good, why = False, f"expected ok, refused {code}: {detail}"
            else:
                bad = []
                if ok["id"] != v.get("id"):
                    bad.append(f"id {ok['id']} != frozen {v.get('id')!r}")
                if "tensor_schema" in v and ok["tensor_schema"] != v["tensor_schema"]:
                    bad.append(
                        f"tensor schema {ok['tensor_schema']!r} "
                        f"!= frozen {v['tensor_schema']}"
                    )
                if "manifest" in v and ok["manifest"] != v["manifest"]:
                    bad.append(f"manifest {ok['manifest']!r} != frozen {v['manifest']}")
                good = not bad
                why = "; ".join(bad) if bad else f"id={str(ok['id'])[7:19]}"
        elif expect == "refuse":
            want = v.get("code", "")
            if ok is not None:
                good, why = False, f"expected {want} but it passed"
            elif code != want:
                good, why = False, f"refused {code} but the frozen code is {want}"
            else:
                good, why = True, f"{code} ({truncate(detail, 72)})"
        elif expect == "mutant":
            if ok is None:
                good, why = True, f"refused {code}"
            else:
                frm = v.get("note", "")
                if str(ok["id"]) in frm:
                    good, why = False, "a flipped byte kept the same object id"
                else:
                    good, why = True, f"distinct id={str(ok['id'])[7:19]}"
        else:
            good, why = False, f"unknown expectation {expect!r}"

        if good:
            npass += 1
            print(f"PASS {name:<44} {expect:<7} {why}")
        else:
            nfail += 1
            print(f"FAIL {name:<44} {expect:<7} {why}")

    # ---------------------------------------------------------- cross-vector assertions

    def idof(n: str) -> str:
        return str(outcomes.get(n, {}).get("id") or "")

    def schemaof(n: str) -> str:
        return str(outcomes.get(n, {}).get("tensor_schema") or "")

    def manifest_of(n: str) -> str:
        return str(outcomes.get(n, {}).get("manifest") or "")

    cross: list[tuple[str, bool, str]] = []
    cross.append(
        (
            "manifest identity is the hash of its exact bytes",
            manifest_of("manifest/checkpoint") == idof("manifest/checkpoint")
            and bool(manifest_of("manifest/checkpoint")),
            manifest_of("manifest/checkpoint"),
        )
    )
    cross.append(
        (
            "header ObjectId != manifest id",
            idof("header/plain-fp8") != manifest_of("manifest/checkpoint"),
            f"{idof('header/plain-fp8')[:19]} vs {manifest_of('manifest/checkpoint')[:19]}",
        )
    )
    cross.append(
        (
            "a config change flips header id, not tensor_schema_digest",
            idof("header/plain-fp8") != idof("header/plain-fp8-noconfig")
            and schemaof("header/plain-fp8") == schemaof("header/plain-fp8-noconfig"),
            schemaof("header/plain-fp8"),
        )
    )
    cross.append(
        (
            "tensor reorder flips header id, not tensor_schema_digest",
            idof("header/plain-fp8") != idof("header/plain-fp8-reordered")
            and schemaof("header/plain-fp8")
            == schemaof("header/plain-fp8-reordered"),
            schemaof("header/plain-fp8-reordered"),
        )
    )
    cross.append(
        (
            "one more ordinary file flips only the manifest id",
            manifest_of("manifest/checkpoint")
            != manifest_of("manifest/checkpoint-extra-file"),
            manifest_of("manifest/checkpoint-extra-file"),
        )
    )

    rcpt = (VEC / "cases/ingest/receipt.json").read_bytes()
    subj = (VEC / "cases/ingest/subject.json").read_bytes()
    manifest = (VEC / "cases/manifest/checkpoint.json").read_bytes()
    hdr = (VEC / "cases/header/plain-fp8.cbor").read_bytes()
    stamp = (VEC / "cases/ingest/stamp.json").read_bytes()
    try:
        tensorfs.receipt_binds(rcpt, subj, manifest, hdr, stamp)
        cross.append(("receipt binds its exact subject/manifest/header", True, ""))
    except errors.Refusal as e:
        cross.append(("receipt binds its exact subject/manifest/header", False, e.detail))

    # The replay arm needs a DIFFERENT subject. Plant one: the tenant is a value, so
    # swapping it leaves canonical bytes canonical — and the facade re-reads them through
    # the one reader before they are used, so a broken plant announces itself.
    other = subj.replace(b'"acme"', b'"other-tenant"')
    assert other != subj, "the tenant plant did not apply"
    tensorfs.run_doc("subject", other, closure)
    try:
        tensorfs.receipt_binds(rcpt, other, manifest, hdr, stamp)
        cross.append(("cross-subject replay refuses", False, "it bound"))
    except errors.Refusal as e:
        cross.append(("cross-subject replay refuses", e.code == "CROSS_SUBJECT_REPLAY", e.detail))

    have = set(closure.digests())
    seeds = tensorfs.seed_digests()
    cross.append(
        (
            "every seed digest resolves in the corpus closure",
            all(d in have for _, d in seeds),
            "",
        )
    )

    for what, good, detail in cross:
        if good:
            npass += 1
            print(f"PASS cross: {what:<52} {detail}")
        else:
            nfail += 1
            print(f"FAIL cross: {what:<52} {detail}")

    print(f"\n{npass} pass, {nfail} fail, {len(closure)} closure specs, {time.time() - started:.3f}s")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
