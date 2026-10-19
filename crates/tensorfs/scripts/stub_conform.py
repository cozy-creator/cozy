#!/usr/bin/env python3
"""`_ext.pyi` against the COMPILED EXTENSION, name for name and parameter for parameter.

`py.typed` is a promise this package makes to every consumer's type checker. Until tfs-077
it was false: `_ext.pyi` ended in `def __getattr__(name: str) -> Any`, so every attribute
on the extension typed as `Any`, mypy checked nothing across the seam, and said so in
green. `Store.prepare_model_source` was called with six arguments against a five-parameter
binding and shipped through a consumer's strict-mode CI (cozy-runtime#296).

A hand-written stub that nothing checks is the same defect wearing more files. This is
what checks it. It imports the BUILT extension — the wheel is what a pod runs, not the
Rust source — reads the `.pyi` that ships beside it, and refuses on any disagreement:

  1. NO ESCAPE HATCH. A `__getattr__` anywhere in the stub re-types the whole surface as
     `Any`. It is the first thing checked, because its presence makes every other check
     here vacuous.
  2. COMPLETENESS, both ways. Every public name the extension exports is declared, and
     every declared name exists. An export nobody stubbed is an `Any` hole; a stub entry
     with nothing behind it is a call that fails at runtime having type-checked.
  3. SIGNATURES. Parameter names, order, kind (positional-only / keyword-only / varargs)
     and default values, from `inspect.signature` over pyo3's `__text_signature__`.
  4. KIND. A `getset_descriptor` is declared `@property`, a method as a method, a class as
     a class. A property stubbed as a method type-checks `store.root()` — which raises.

What it does NOT check, stated so the coverage is not mistaken for more than it is: the
TYPES. `hex: str` is this file's word against nobody — pyo3 erases Rust types into
`PyObject` and `__text_signature__` carries no annotation to compare against. Arity,
naming and shape are machine-checked here; meaning is reviewed, and the argument-meaning
half is what `cozy-runtime/tests/test_tensorfs_binding_contract.py` still holds.

    scripts/stub_conform.py [--emit]

`--emit` prints a skeleton stub for the current extension, every parameter typed `object`.
It is a STARTING POINT for a human, never a fix: regenerating over a divergence would
launder a seam change into a green build, which is the failure this exists to end.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import pathlib
import sys
import types
from typing import Any

#: pyo3 renders an unrepresentable default as `...` in `__text_signature__`, and CPython
#: hands it back as this sentinel. The stub spells it `...` too, so the two agree.
ELLIPSIS = "Ellipsis"


#: Dunders that are SURFACE, not plumbing. `with store.acquire_cozytensors(id) as lease:`
#: is the documented way to take a lease, so `__enter__`/`__exit__` are part of the seam and
#: are checked like any other member. The rest (`__repr__`, `__init__`, `__doc__`, pyo3's
#: own machinery) are not, and a stub may declare them freely.
CHECKED_DUNDERS = frozenset({"__enter__", "__exit__", "__len__", "__iter__", "__next__"})


def public(name: str) -> bool:
    """`_fit` is public surface — `__init__.py` imports it. Only dunders are internal."""
    return not name.startswith("__") or name in CHECKED_DUNDERS


# --------------------------------------------------------------------------- the runtime


def runtime_params(obj: object, *, drop_self: bool) -> list[tuple[str, str, str]] | None:
    """(name, kind, default) per parameter, or None when the object carries no signature."""
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    out: list[tuple[str, str, str]] = []
    for i, p in enumerate(sig.parameters.values()):
        if drop_self and i == 0 and p.name == "self":
            continue
        default = "" if p.default is inspect.Parameter.empty else repr(p.default)
        # `self` is positional-only on a method descriptor and ordinary in a stub; every
        # other parameter's kind is compared exactly.
        out.append((p.name, p.kind.name, default))
    return out


def member_kind(obj: object) -> str:
    name = type(obj).__name__
    if name in {"getset_descriptor", "member_descriptor", "property"}:
        return "property"
    if name in {"method_descriptor", "builtin_function_or_method", "function", "wrapper_descriptor"}:
        return "callable"
    if name == "staticmethod":
        return "callable"
    if isinstance(obj, type):
        return "class"
    return "value"


def read_extension(ext: Any) -> dict[str, Any]:
    """Every public export, and for each class every public member, off the built module."""
    surface: dict[str, Any] = {}
    for name in dir(ext):
        if not public(name):
            continue
        obj = getattr(ext, name)
        kind = member_kind(obj)
        entry: dict[str, Any] = {"kind": kind}
        if kind == "callable":
            entry["params"] = runtime_params(obj, drop_self=False)
        if kind == "class":
            members: dict[str, Any] = {}
            for attr in vars(obj):
                if not public(attr):
                    continue
                # Resolve through the class so a `staticmethod` wrapper hands back the
                # function that actually carries `__text_signature__`.
                bound = getattr(obj, attr)
                mkind = member_kind(vars(obj)[attr])
                members[attr] = {
                    "kind": mkind,
                    "params": (
                        runtime_params(bound, drop_self=isinstance(vars(obj)[attr], staticmethod) is False)
                        if mkind == "callable"
                        else None
                    ),
                    "static": isinstance(vars(obj)[attr], staticmethod),
                }
            entry["members"] = members
            entry["bases"] = [b.__name__ for b in obj.__bases__ if b is not object]
        surface[name] = entry
    return surface


# ------------------------------------------------------------------------------ the stub


def stub_params(node: ast.FunctionDef, *, drop_self: bool) -> list[tuple[str, str, str]]:
    a = node.args
    out: list[tuple[str, str, str]] = []
    posonly = list(a.posonlyargs)
    normal = list(a.args)
    ndef = len(a.defaults)
    positional = posonly + normal
    first_default = len(positional) - ndef
    for i, arg in enumerate(positional):
        if drop_self and i == 0 and arg.arg in {"self", "cls"}:
            continue
        kind = "POSITIONAL_ONLY" if i < len(posonly) else "POSITIONAL_OR_KEYWORD"
        default = ""
        if i >= first_default:
            default = ast.unparse(a.defaults[i - first_default])
        out.append((arg.arg, kind, default))
    if a.vararg:
        out.append((a.vararg.arg, "VAR_POSITIONAL", ""))
    for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        out.append((arg.arg, "KEYWORD_ONLY", "" if d is None else ast.unparse(d)))
    if a.kwarg:
        out.append((a.kwarg.arg, "VAR_KEYWORD", ""))
    return out


def decorators(node: ast.FunctionDef) -> set[str]:
    return {ast.unparse(d) for d in node.decorator_list}


def read_stub(path: pathlib.Path) -> tuple[dict[str, Any], list[str]]:
    """The stub's declared surface, plus every escape hatch found in it."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hatches: list[str] = []
    surface: dict[str, Any] = {}

    def scan(body: list[ast.stmt], scope: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for node in body:
            if isinstance(node, ast.FunctionDef):
                if node.name == "__getattr__":
                    hatches.append(f"{scope}__getattr__")
                if not public(node.name):
                    continue
                dec = decorators(node)
                static = "staticmethod" in dec
                entry = {
                    "kind": "property" if "property" in dec else "callable",
                    "static": static,
                    "params": stub_params(node, drop_self=scope != "" and not static),
                }
                if "overload" in dec:
                    previous = out.get(node.name)
                    if previous is not None and "overloads" in previous:
                        previous["overloads"].append(entry["params"])
                    else:
                        entry["overloads"] = [entry["params"]]
                        out[node.name] = entry
                else:
                    out[node.name] = entry
            elif isinstance(node, ast.ClassDef):
                if not public(node.name):
                    continue
                bases = [ast.unparse(b) for b in node.bases]
                # A TypedDict or Protocol in a stub DESCRIBES a value the extension hands
                # back or takes; it is not a claim that a class by that name is exported.
                if {"TypedDict", "Protocol"} & set(bases):
                    continue
                out[node.name] = {
                    "kind": "class",
                    "members": scan(node.body, f"{node.name}."),
                    "bases": bases,
                }
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if public(node.target.id):
                    out[node.target.id] = {"kind": "value"}
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and public(t.id):
                        out[t.id] = {"kind": "value"}
        return out

    surface = scan(tree.body, "")
    return surface, hatches


# ------------------------------------------------------------------------------ the diff


def compare_params(
    where: str,
    runtime: list[tuple[str, str, str]] | None,
    stub: list[tuple[str, str, str]],
    faults: list[str],
) -> None:
    if runtime is None:
        return  # no `__text_signature__` to compare against; nothing is asserted.
    if len(runtime) != len(stub):
        faults.append(
            f"{where}: the extension takes {len(runtime)} parameter(s) "
            f"({', '.join(p[0] for p in runtime) or 'none'}) and the stub declares "
            f"{len(stub)} ({', '.join(p[0] for p in stub) or 'none'})"
        )
        return
    for r, s in zip(runtime, stub, strict=True):
        if r[0] != s[0]:
            faults.append(f"{where}: parameter {r[0]!r} is named {s[1 - 1] and s[0]!r} in the stub")
        elif r[1] != s[1]:
            faults.append(f"{where}: parameter {r[0]!r} is {r[1]} in the extension, {s[1]} in the stub")
        else:
            rd, sd = r[2], s[2]
            if bool(rd) != bool(sd):
                faults.append(
                    f"{where}: parameter {r[0]!r} "
                    + ("has a default in the extension and none in the stub" if rd
                       else "has a default in the stub and none in the extension")
                )
            elif rd and sd != "..." and rd != sd and not (rd == ELLIPSIS and sd == "..."):
                faults.append(f"{where}: parameter {r[0]!r} defaults to {rd}, the stub says {sd}")


def compare_overloads(where: str, runtime: list[tuple[str, str, str]] | None,
                      variants: list[list[tuple[str, str, str]]], faults: list[str]) -> None:
    """Every typed branch describes accepted calls; one carries the full native ABI.

    Mapping.get can require a default of a particular type in a narrower typed branch.
    Its default=None branch still checks every native name, kind and default exactly.
    """
    if runtime is None:
        return
    full = False
    for index, params in enumerate(variants, 1):
        exact: list[str] = []
        compare_params(where, runtime, params, exact)
        full |= not exact
        label = f"{where} overload {index}"
        if len(params) > len(runtime):
            faults.append(f"{label}: declares more parameters than the native callable")
            continue
        for native, declared in zip(runtime, params):
            if native[:2] != declared[:2]:
                faults.append(f"{label}: parameter {declared[0]!r} name/kind differs from the native callable")
            elif declared[2] and (not native[2] or declared[2] not in {native[2], "..."}):
                faults.append(f"{label}: default for {declared[0]!r} differs from the native callable")
        if any(not default and kind not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
               for _, kind, default in runtime[len(params):]):
            faults.append(f"{label}: omits a required native parameter")
    if not full:
        faults.append(f"{where}: no typed overload declares the complete native signature and defaults")


def inherited(surface: dict[str, Any], name: str, seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    """A class's members plus every member it inherits from bases declared in this surface.

    `Error.detail` is set on the INSTANCE, and each subclass carries its own `code`. Both
    spellings have to compare equal to one declaration on the base, or the stub would have
    to repeat two lines for every family and the repetition would be the drift.
    """
    entry = surface.get(name)
    if entry is None or entry.get("kind") != "class" or name in seen:
        return {}
    members = dict(entry.get("members", {}))
    for base in entry.get("bases", []):
        for key, value in inherited(surface, base, seen | {name}).items():
            members.setdefault(key, value)
    return members


def compare(runtime: dict[str, Any], stub: dict[str, Any], scope: str, faults: list[str]) -> None:
    for name in sorted(set(runtime) - set(stub)):
        faults.append(
            f"{scope}{name}: the extension exports it and the stub does not declare it "
            f"— every call through this name type-checks as Any"
        )
    for name in sorted(set(stub) - set(runtime)):
        # A bare annotation may describe an INSTANCE attribute, which no class-level
        # introspection can see; `Error.detail` is one. The direction that matters —
        # an export the stub does not cover — stays absolute above.
        if stub[name].get("kind") == "value":
            continue
        faults.append(
            f"{scope}{name}: the stub declares it and the extension does not export it "
            f"— a call through this name type-checks and then raises"
        )
    for name in sorted(set(runtime) & set(stub)):
        r, s = runtime[name], stub[name]
        where = f"{scope}{name}"
        if r["kind"] != s["kind"]:
            faults.append(f"{where}: the extension has a {r['kind']}, the stub declares a {s['kind']}")
            continue
        if r["kind"] == "callable":
            if r.get("static") != s.get("static") and scope:
                faults.append(
                    f"{where}: "
                    + ("static in the extension, an instance method in the stub"
                       if r.get("static") else "an instance method in the extension, static in the stub")
                )
            if "overloads" in s:
                compare_overloads(where, r.get("params"), s["overloads"], faults)
            else:
                compare_params(where, r.get("params"), s.get("params", []), faults)
        elif r["kind"] == "class":
            compare(
                inherited(runtime, name) if not scope else r["members"],
                inherited(stub, name) if not scope else s["members"],
                f"{name}.",
                faults,
            )


# ----------------------------------------------------------------------------- emit mode


def emit(runtime: dict[str, Any]) -> str:
    lines = ["# SKELETON — every parameter typed `object`. Type it by hand.", "from typing import Any", ""]

    def render(entry: dict[str, Any], name: str, indent: str, static: bool) -> None:
        params = entry.get("params") or []
        rendered = [] if (static or not indent) else ["self"]
        for i, (pname, kind, default) in enumerate(params):
            if kind == "VAR_POSITIONAL":
                rendered.append(f"*{pname}: object")
                continue
            if kind == "VAR_KEYWORD":
                rendered.append(f"**{pname}: object")
                continue
            if kind == "KEYWORD_ONLY" and not any(p.startswith("*") for p in rendered):
                rendered.append("*")
            ann = "" if pname in {"self", "cls"} else ": object"
            rendered.append(f"{pname}{ann}" + (" = ..." if default else ""))
            if kind == "POSITIONAL_ONLY" and (
                i + 1 == len(params) or params[i + 1][1] != "POSITIONAL_ONLY"
            ):
                rendered.append("/")
        if static:
            lines.append(f"{indent}@staticmethod")
        lines.append(f"{indent}def {name}({', '.join(rendered)}) -> object: ...")

    for name, entry in sorted(runtime.items()):
        if entry["kind"] == "class":
            bases = ", ".join(entry["bases"]) if entry["bases"] else ""
            lines.append(f"class {name}({bases}):" if bases else f"class {name}:")
            members = entry["members"]
            if not members:
                lines.append("    ...")
            for mname, m in sorted(members.items()):
                if m["kind"] == "property":
                    lines.append("    @property")
                    lines.append(f"    def {mname}(self) -> object: ...")
                elif m["kind"] == "callable":
                    render(m, mname, "    ", m["static"])
                else:
                    lines.append(f"    {mname}: object")
            lines.append("")
        elif entry["kind"] == "callable":
            render(entry, name, "", False)
        else:
            lines.append(f"{name}: Any")
    return "\n".join(lines) + "\n"


def check_error_families(ext: Any) -> list[str]:
    """Every native seat exposes the same twelve error families on tensorfs.Error."""
    import tensorfs

    faults = []
    for obsolete in ("Refusal", "CODES", "Closure", "ReadPlan", "parse_header", "plan", "regions"):
        if hasattr(ext, obsolete) or hasattr(tensorfs, obsolete):
            faults.append(f"{obsolete}: removed legacy surface remains exported")
    if importlib.util.find_spec("tensorfs.records") is not None:
        faults.append("tensorfs.records: removed legacy header projection remains importable")
    expected = {"invalid", "format", "not_found", "integrity", "unknown_scheme", "decode",
                "io", "full", "network", "denied", "unsupported", "conflict"}
    live = {name: getattr(ext, name) for name in dir(ext)
            if isinstance(getattr(ext, name), type) and issubclass(getattr(ext, name), ext.Error)
            and getattr(ext, name) is not ext.Error}
    if {cls.code for cls in live.values()} != expected:
        faults.append("Error subclasses do not expose exactly the twelve public families")
    for name, cls in live.items():
        if getattr(tensorfs, name, None) is not cls:
            faults.append(f"tensorfs.{name} does not re-export its native error family")
    try:
        ext.Checkpoint.parse(b"not a header")
    except ext.Error as error:
        if error.code != "invalid" or not isinstance(error.detail, str):
            faults.append("Checkpoint.parse did not preserve its common typed failure")
    except Exception as error:
        faults.append(f"Checkpoint.parse escaped the common Error boundary: {type(error).__name__}")
    else:
        faults.append("Checkpoint.parse accepted invalid CBOR")
    return faults


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="print a skeleton stub and exit")
    ap.add_argument("--stub", type=pathlib.Path, default=None, help="the .pyi to check")
    args = ap.parse_args()

    try:
        from tensorfs import _ext
    except ImportError as exc:
        print(f"the extension is not importable, so nothing was checked: {exc}", file=sys.stderr)
        return 2

    runtime = read_extension(_ext)
    if args.emit:
        print(emit(runtime))
        return 0

    # The stub that SHIPS BESIDE the extension being checked. Reading the repo's copy
    # while importing an installed wheel would compare two different versions and pass.
    stub_path = args.stub or pathlib.Path(_ext.__file__).with_name("_ext.pyi")
    if not stub_path.exists():
        print(f"no stub at {stub_path}: the package claims `py.typed` and ships none", file=sys.stderr)
        return 1

    declared, hatches = read_stub(stub_path)
    faults: list[str] = []
    for hatch in hatches:
        faults.append(
            f"{hatch}: an escape hatch in the stub. It re-types everything it covers as "
            f"`Any`, which makes every other check in this file vacuous and every consumer's "
            f"strict-mode run a green light over an unchecked seam (tfs-077)."
        )
    if not hatches:
        compare(runtime, declared, "", faults)
    faults.extend(check_error_families(_ext))
    # A compiled submodule (`_ext.varena`) is typed by the stub of the facade module that
    # re-exports it (a module.pyi or package/__init__.pyi), held to the same full checks.
    for name in sorted(n for n in runtime if isinstance(getattr(_ext, n), types.ModuleType)):
        candidates = [stub_path.with_name(f"{name}.pyi"), stub_path.parent / name / "__init__.pyi"]
        available = [path for path in candidates if path.is_file()]
        if len(available) != 1:
            faults.append(f"{name}: expected one module or package stub, found {len(available)}: {available}")
            continue
        sub_path = available[0]
        sub_declared, sub_hatches = read_stub(sub_path)
        sub_faults = [f"{hatch}: an escape hatch in the stub" for hatch in sub_hatches]
        if not sub_hatches:
            compare(read_extension(getattr(_ext, name)), sub_declared, "", sub_faults)
        faults.extend(f"{name}.{fault}" for fault in sub_faults)

    if faults:
        print(f"_ext.pyi disagrees with the extension it ships beside, in {len(faults)} place(s):\n")
        for fault in faults:
            print(f"  {fault}")
        print(
            "\nThe stub is the seam's only compiler. Fix it to match the binding, or fix the "
            "binding — never regenerate over the difference."
        )
        return 1

    nclass = sum(1 for e in runtime.values() if e["kind"] == "class")
    nfunc = sum(1 for e in runtime.values() if e["kind"] == "callable")
    nmeth = sum(len(e.get("members", {})) for e in runtime.values())
    print(
        f"_ext.pyi matches the extension: {nfunc} functions, {nclass} classes, "
        f"{nmeth} members, no escape hatch. `py.typed` is true."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
