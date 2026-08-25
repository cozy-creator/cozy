#!/usr/bin/env python3
"""Transcribe pinned MiniMax-H3 headers out of the h3-evidence bank into NEUTRAL measured
rows, which `tfs compat import` turns into canonical documents.

The bank (`~/cozy_v2/h3-evidence`, READ-ONLY here) holds the exact upstream header bytes of
33 real artifacts, each pinned to a repo revision and a sha256 of the bytes read. This
script reads those bytes and emits component/key/dtype/shape/role rows. It writes NO
canonical document and names NO format mark: there is one canonical writer in this repo and
it is compiled (`scripts/fence.py`, fence 4).

Weights never transit this machine, so the rows carry logical geometry only. `tfs compat
import` fills part object ids with derived placeholders and says so on every run — nothing
below the topology layer is read by `check_compatibility`, which is what makes the pairs
honest evidence for a layout verdict and nothing else.

  scripts/h3-pairs.py list                 what the bank holds, by dialect
  scripts/h3-pairs.py rows <path> <out>    one artifact -> measured rows
  scripts/h3-pairs.py fold <path> <out> <logical-from>
                                           the same, with markers dropped and scales folded
                                           into roles (what the border would have produced).
                                           An encoded tensor's LOGICAL dtype is not in its
                                           own file -- fp8 is the carrier -- so it is read
                                           from the publisher's own unquantized twin, named
                                           explicitly. There is no default and no guess: a
                                           key the twin does not carry refuses.
"""
import json
import pathlib
import re
import sys

BANK = pathlib.Path.home() / "cozy_v2/h3-evidence"
# The marker/role sibling conventions this ecosystem actually ships, measured in
# h3-evidence's "Marker schemas, measured" table. Markers carry no value; role siblings do.
MARKERS = (".comfy_quant", ".quantization_metadata")
ROLE_SUFFIXES = (".weight_scale", ".input_scale")
DTYPES = {"F64": "f64", "F32": "f32", "F16": "f16", "BF16": "bf16",
          "F8_E4M3": "f8_e4m3fn", "F8_E5M2": "f8_e5m2",
          "I64": "i64", "I32": "i32", "I16": "i16", "I8": "i8", "U8": "u8", "BOOL": "bool"}


def bank():
    """path -> (row_id, source record, key map). Cached bytes only; no network."""
    out = {}
    for r in sorted((BANK / "rows").glob("*.json")):
        d = json.loads(r.read_text())
        for f in d.get("banked_files", []):
            if f.get("mode") != "st":
                continue
            p = BANK / "cache" / f["fetch"]["bytes_sha256"]
            if not p.exists():
                continue
            h = json.loads(p.read_bytes())
            keys = {k: v for k, v in h.items()
                    if k != "__metadata__" and isinstance(v, dict) and "dtype" in v}
            if keys:
                out[f["path"]] = (d["row_id"], {
                    "repo": d["source"].get("repo"),
                    "revision": d["source"].get("revision"),
                    "path": f["path"],
                    "header_bytes_sha256": f["fetch"]["bytes_sha256"],
                    "artifact_sha256": f.get("declared_sha256"),
                }, keys)
    return out


def component_of(path):
    p = path.lower()
    for needle, name in (("text_encoder", "text_encoder"), ("audio_vae", "audio_vae"),
                         ("video_vae", "vae"), ("vae", "vae"), ("lora", "lora")):
        if needle in p:
            return name
    return "transformer"


def rows(path, keys, source, fold, logical_from=None):
    """Measured rows. `fold` is what the border does: drop pure markers, fold role siblings
    under their logical tensor. Unfolded is the RAW un-bordered shape, where every sibling
    is still a top-level key — which is exactly the extra-key case the lane rule tiers."""
    logical, roles = {}, {}
    for k, v in keys.items():
        if fold and k.endswith(MARKERS):
            continue
        suffix = next((s for s in ROLE_SUFFIXES if k.endswith(s)), None)
        if fold and suffix:
            roles.setdefault(k[: -len(suffix)] + ".weight", []).append((suffix[1:], v))
            continue
        logical[k] = v
    out = []
    for k, v in sorted(logical.items()):
        dt = DTYPES.get(v["dtype"])
        if dt is None:
            sys.exit(f"unmapped carrier dtype {v['dtype']!r} at {k}")
        row = {"key": k, "dtype": dt, "shape": v["shape"]}
        if k in roles:
            twin = (logical_from or {}).get(k)
            if twin is None:
                sys.exit(f"{k}: encoded, and the named unquantized twin does not carry it — "
                         "a logical dtype is measured or it is not stated")
            if twin["shape"] != v["shape"]:
                sys.exit(f"{k}: twin shape {twin['shape']} != encoded shape {v['shape']}")
            row["dtype"] = DTYPES[twin["dtype"]]
            # The canonical spelling: the logical tensor's own bytes become the `data` role
            # and its scales become sibling roles under the same tensor.
            row["roles"] = sorted(
                [{"role": "data", "dtype": dt, "shape": v["shape"]}]
                + [{"role": n, "dtype": DTYPES[rv["dtype"]], "shape": rv["shape"]}
                   for n, rv in roles[k]],
                key=lambda r: r["role"])
        out.append(row)
    return {
        "component": component_of(path),
        "provenance": (f"{source['repo']}@{source['revision'][:8]} {source['path']} "
                       f"header sha256:{source['header_bytes_sha256'][:16]}"
                       f"{' folded' if fold else ' raw'}"),
        "tensors": out,
    }


def main():
    b = bank()
    verb = sys.argv[1] if len(sys.argv) > 1 else "list"
    if verb == "list":
        print(f"{len(b)} pinned safetensors headers with full key maps\n")
        for path, (row_id, src, keys) in b.items():
            dt = {}
            for v in keys.values():
                dt[v["dtype"]] = dt.get(v["dtype"], 0) + 1
            markers = sum(1 for k in keys if k.endswith(MARKERS))
            scales = sum(1 for k in keys if k.endswith(ROLE_SUFFIXES))
            print(f"{row_id:24} {path[:56]:56} {len(keys):5} keys "
                  f"({len(keys) - markers - scales} logical, {markers} markers, "
                  f"{scales} role siblings) {dt}")
        return
    if verb not in ("rows", "fold") or len(sys.argv) < 4:
        sys.exit(__doc__)

    def resolve(name):
        if name in b:
            return name
        m = [p for p in b if re.search(re.escape(name), p)]
        if len(m) != 1:
            sys.exit(f"{name!r} matches {len(m)} banked artifacts")
        return m[0]

    path, out = resolve(sys.argv[2]), pathlib.Path(sys.argv[3])
    twin = None
    if len(sys.argv) > 4:
        tp = resolve(sys.argv[4])
        twin = b[tp][2]
        print(f"logical dtypes from {tp}")
    row_id, src, keys = b[path]
    doc = rows(path, keys, src, fold=(verb == "fold"), logical_from=twin)
    out.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    print(f"{doc['provenance']}\n  {len(doc['tensors'])} rows -> {out}")


main()
