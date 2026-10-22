"""Independent header-only selection reference for the shared conformance vectors (spec §7)."""
from collections import defaultdict
import json


def family(kind):
    if kind in ("BF16", "F16", "F32"):
        return {"BF16": "bf16", "F16": "fp16", "F32": "f32"}[kind], 16
    if kind in ("nvfp4", "mxfp4"):
        return kind, 4
    for prefix, width in (("int8", 8), ("fp8", 8), ("int4", 4)):
        if kind.startswith(prefix + "."):
            return prefix, width
    return "", 0


def profile(header):
    held = defaultdict(int)
    for component in header["components"].values():
        for tensor in component["tensors"].values():
            if "encoding" in tensor:
                kind = tensor["encoding"]["scheme"]
                size = sum(part["length"] for part in tensor["parts"].values())
            else:
                kind, size = tensor["dtype"], tensor["data"]["length"]
            held[kind] += size
    lanes = [kind for kind in held if family(kind)[1]]
    lane = min(lanes, key=lambda kind: (family(kind)[1], -held[kind], kind)) if lanes else ""
    name, width = family(lane)
    return dict(family=name, bits=width, bytes=sum(held.values()), kinds=sorted(held))


def select(profiles, constraints, model, plain):
    width = constraints.get("bits")
    width = 8 if width is None else width
    if width not in (4, 8, 16):
        raise ValueError("invalid bits")
    devices = constraints.get("gpus", [])
    def capability(text):
        parts = text.split(".")
        if len(parts) == 2 and all(0 < len(p) <= 4 and p.isascii() and p.isdigit() for p in parts):
            return int(parts[0]) * 10 + int(parts[1])
        return None
    cc = [capability(device) for device in devices]
    all_devices = lambda predicate: bool(cc) and all(n is not None and predicate(n) for n in cc)
    mma = all_devices(lambda n: n >= 80)
    int8 = all_devices(lambda n: n >= 80 and n != 103)
    fp8 = all_devices(lambda n: n >= 89)
    fp4 = all_devices(lambda n: n >= 100)
    bf16 = not any(n is not None and n < 80 for n in cc)
    supported = {"int8.tensorwise", "int8.rowwise", "int8.columnwise", "int8.convrot",
                 "fp8.tensorwise", "fp8.rowwise", "fp8.columnwise", "fp8.blockwise",
                 "nvfp4", "mxfp4", "int4.convrot"}
    def runs(kind):
        return kind in plain or (width != 16 and kind in supported and
               {"int8": int8, "fp8": fp8, "int4": mma, "nvfp4": mma, "mxfp4": mma}.get(family(kind)[0], False))
    order = (["nvfp4"] if width == 4 and mma else [])
    order += (["int8"] if width <= 8 and int8 else []) + (["fp8"] if width <= 8 and fp8 else [])
    order += ["bf16", "fp16", "f32"] if bf16 else ["fp16", "bf16", "f32"]
    ranked = []
    for path, p in profiles.items():
        group = "nvfp4" if p["family"] in ("mxfp4", "int4") else p["family"]
        if group in order and all(runs(kind) for kind in p["kinds"]):
            weight_only = p["family"] == "int4" or (p["family"] in ("nvfp4", "mxfp4") and not fp4)
            ranked.append((order.index(group), weight_only, p["bytes"], path))
    chosen = min(ranked)[-1] if ranked else None
    if chosen is None and len(profiles) == 1:
        path, p = next(iter(profiles.items()))
        if not p["family"] and all(runs(kind) for kind in p["kinds"]):
            chosen = path
    reasons = []
    if chosen is None:
        return chosen, reasons
    served = profiles[chosen]["family"]
    if served not in ("bf16", "fp16", "int8", "fp8", "nvfp4"):
        return chosen, reasons
    gpu_name = "+".join(sorted({"sm" + device.replace(".", "") for device in devices if device})) or "no GPU"
    missing = order[:order.index(served)]
    if missing:
        names = ", ".join(missing[:-1]) + " or " + missing[-1] if len(missing) > 1 else missing[0]
        reasons.append(f"precision {width}: {model} has no {names} lane for {gpu_name}; served {served}")
    if served == "fp16" and not bf16:
        reasons.append(f"precision {width}: {gpu_name} runs no bf16; served fp16")
    return chosen, reasons


def check(vectors, parse_header, plain):
    spec = json.loads((vectors / "select.json").read_text())
    profiles, failures, passed = {}, [], 0
    for name, want in spec["profiles"].items():
        got = profile(parse_header((vectors / "select" / f"{name}.cbor").read_bytes()))
        profiles[name] = got
        passed += got == want
        if got != want:
            failures.append(f"selection profile: {name}: {got} != {want}")
    for case in spec["cases"]:
        try:
            got = select({p: profiles[n] for p, n in case["variants"].items()}, case["constraints"], spec["model"], plain)
            ok = not case.get("error") and got == (case["variant"], case["reasons"])
        except ValueError:
            ok = case.get("error") == "invalid"
        passed += ok
        if not ok:
            failures.append(f"selection: {case['why']}")
    for case in spec["lookups"]:
        asked = case["ask"]
        candidates = [asked, asked + ".cozytensors", f"model.{asked}.cozytensors" if asked else "model.cozytensors"]
        got = next((p for p in candidates if p in case["paths"] and p.endswith(".cozytensors")), None)
        passed += got == case["variant"]
        if got != case["variant"]:
            failures.append(f"selection lookup: {asked}")
    return passed, failures
