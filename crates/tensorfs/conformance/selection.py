"""Independent header-only selection reference for the shared conformance vectors (spec §7)."""
from collections import defaultdict
import json

from schemes import SCHEMES


def family(kind):
    if kind in ("BF16", "F16", "F32"):
        return {"BF16": "bf16", "F16": "fp16", "F32": "f32"}[kind], 16
    if kind in ("nvfp4", "mxfp4"):
        return kind, 4
    if kind == "w4a8.convrot":
        return "w4a8", 4
    for prefix, width in (("int8", 8), ("fp8", 8), ("int4", 4)):
        if kind.startswith(prefix + "."):
            return prefix, width
    return "", 0


def profile(header, plain):
    held = defaultdict(int)
    for component in header["components"].values():
        for tensor in component["tensors"].values():
            if "encoding" in tensor:
                kind = tensor["encoding"]["scheme"]
                size = sum(part["length"] for part in tensor["parts"].values())
            else:
                kind, size = tensor["dtype"], tensor["data"]["length"]
            held[kind] += size
    untabled = any(kind not in plain and not family(kind)[1] for kind in held)
    lanes = [] if untabled else [kind for kind in held if family(kind)[1]]
    lane = min(lanes, key=lambda kind: (family(kind)[1], -held[kind], kind)) if lanes else ""
    name, width = family(lane)
    return dict(family=name, bits=width, bytes=sum(held.values()), kinds=sorted(held))


def capability(text):
    parts = text.split(".")
    if len(parts) == 2 and all(0 < len(p) <= 4 and p.isascii() and p.isdigit() for p in parts):
        return int(parts[0]) * 10 + int(parts[1])
    return None


def gpus(devices):
    """What every one of the devices runs, and how a reason names them."""
    cc = [capability(device) for device in devices]
    every = lambda predicate: bool(cc) and all(n is not None and predicate(n) for n in cc)
    name = "+".join(sorted({"sm" + device.replace(".", "") for device in devices if device})) or "no GPU"
    return dict(mma=every(lambda n: n >= 80), int8=every(lambda n: n >= 80 and n != 103), fp8=every(lambda n: n >= 89),
                fp4=every(lambda n: n >= 100), bf16=not any(n is not None and n < 80 for n in cc), name=name)


NEEDS = {"int8": "sm80+ except sm103", "w4a8": "sm80+ except sm103", "fp8": "sm89+", "nvfp4": "sm80+", "mxfp4": "sm80+",
         "int4": "sm80+"}


def runs(kind, device):
    """Whether one device runs an encoded kind: one of the table where its family does, another
    scheme this reader implements always (it decodes anywhere), an unknown one never."""
    fam = family(kind)[0]
    if kind not in SCHEMES:
        return False
    if fam not in NEEDS:
        return True
    g = gpus([device])
    return g["int8"] if fam in ("int8", "w4a8") else g["fp8"] if fam == "fp8" else g["mma"]


def select(profiles, constraints, model, plain):
    for key in ("bits", "prefer", "min_bits"):
        if constraints.get(key) is not None and constraints[key] not in (4, 8, 16):
            raise ValueError("invalid bits")
    asked, devices, takes = constraints.get("bits"), constraints.get("gpus", []), not constraints.get("plain", False)
    width = asked or constraints.get("prefer") or 8
    g = gpus(devices)

    def refusal(p):
        encoded = [kind for kind in p["kinds"] if kind not in plain]
        if asked is not None and p["bits"] != asked:
            return f"is not {asked}-bit"
        if encoded and (asked == 16 or not takes):
            return "holds encoded " + ", ".join(encoded)
        floor = constraints.get("min_bits")
        if asked is None and floor and 0 < p["bits"] < floor:
            return f"is {p['bits']}-bit, under {floor}"
        unrun = [kind for kind in encoded if devices and not any(runs(kind, device) for device in devices)]
        if unrun:
            return "needs " + " and ".join(f"{NEEDS[family(k)[0]] if k in SCHEMES and family(k)[0] in NEEDS else 'a GPU reader'} for {k}" for k in unrun)
        return None

    def lanes(encoded):
        four = [lane for lane, runs in (("nvfp4", g["mma"]), ("w4a8", g["int8"])) if runs]
        decode = asked is None and constraints.get("prefer") == 4
        native = (four if g["fp4"] or decode else four[::-1]) if width == 4 else []
        native += (["int8"] if width <= 8 and g["int8"] else []) + (["fp8"] if width <= 8 and g["fp8"] else [])
        return (native if encoded else []) + (["bf16", "fp16"] if g["bf16"] else ["fp16", "bf16"])

    order = lanes(takes)
    ranked = []
    for path, p in profiles.items():
        if refusal(p) is not None:
            continue
        group = "nvfp4" if p["family"] in ("mxfp4", "int4") else p["family"]
        place = len(order) if group == "f32" else order.index(group) if group in order else len(order) + 1
        weight_only = p["family"] == "int4" or (p["family"] in ("nvfp4", "mxfp4") and not g["fp4"])
        ranked.append((place, 0 < p["bits"] < width, weight_only, p["bytes"], path))
    if not ranked:
        each = [f"{path} ({p['family'] or '+'.join(p['kinds'])}) {refusal(p)}" for path, p in sorted(profiles.items())]
        raise LookupError(f"precision {width} on {g['name']}: {model} has no variant that works; it publishes {'; '.join(each) or 'none'}")
    chosen = min(ranked)[-1]
    served = profiles[chosen]["family"]
    reasons = []
    if served not in ("bf16", "fp16", "int8", "fp8", "nvfp4", "w4a8"):
        return chosen, reasons
    if not takes and len(lanes(True)) > 2:
        reasons.append(f"precision {width}: {model} declines encoded leaves; served {served}")
    missing = order[:order.index(served)] if served in order else order
    if missing:
        names = ", ".join(missing[:-1]) + " or " + missing[-1] if len(missing) > 1 else missing[0]
        reasons.append(f"precision {width}: {model} has no {names} lane for {g['name']}; served {served}")
    if served == "fp16" and not g["bf16"]:
        reasons.append(f"precision {width}: {g['name']} runs no bf16; served fp16")
    return chosen, reasons


def check(vectors, parse_header, plain):
    spec = json.loads((vectors / "select.json").read_text())
    profiles, failures, passed = {}, [], 0
    if spec["schemes"] != sorted(SCHEMES):
        failures.append("selection: the vectors' schemes are not this reader's")
    for name, want in spec["profiles"].items():
        got = profile(parse_header((vectors / "select" / f"{name}.cbor").read_bytes()), plain)
        profiles[name] = got
        passed += got == want
        if got != want:
            failures.append(f"selection profile: {name}: {got} != {want}")
    for case in spec["cases"]:
        variants = {p: profiles[n] for p, n in case["variants"].items()}
        try:
            got = select(variants, case["constraints"], spec["model"], plain)
            ok = "error" not in case and "refused" not in case and got == (case["variant"], case["reasons"])
        except ValueError:
            ok = case.get("error") == "invalid"
        except LookupError as refused:
            ok = str(refused) == case.get("refused")
        passed += ok
        if not ok:
            failures.append(f"selection: {case['why']}")
    for case in spec["lookups"]:
        asked = case["ask"]
        got = asked if asked in case["paths"] and asked.endswith(".cozytensors") else None
        passed += got == case["variant"]
        if got != case["variant"]:
            failures.append(f"selection lookup: {asked}")
    return passed, failures
