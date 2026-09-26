"""Read metadata only. Never import code or execute the captured environment."""

import json
import sys
from collections import deque
from email.parser import Parser
from typing import TypeGuard

from packaging._parser import MarkerAtom, MarkerList, Variable
from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

def text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("requirement metadata field must be a string")
    return value


def object_mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def object_sequence(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def mapping(value: object) -> dict[str, object]:
    if not object_mapping(value):
        raise ValueError("requirement metadata must be an object")
    return {text(key): item for key, item in value.items()}


def strings(value: object) -> list[str]:
    if not object_sequence(value):
        raise ValueError("requirement metadata field must be a string list")
    return [text(item) for item in value]


request = mapping(json.load(sys.stdin))
python_text = text(request["python"])
python = Version(python_text)
if len(python.release) != 3:
    raise ValueError("requirement evaluation needs the measured Python patch")
environment = {key: text(value) for key, value in default_environment().items()}
environment.update(
    implementation_name="cpython", implementation_version=python_text, os_name="posix",
    platform_machine="x86_64", platform_python_implementation="CPython",
    python_full_version=python_text, python_version=".".join(python_text.split(".")[:2]),
    sys_platform="linux",
)


def variables(node: MarkerList | MarkerAtom) -> set[str]:
    if isinstance(node, tuple):
        return {token.value for token in node if isinstance(token, Variable)}
    found: set[str] = set()
    for item in node:
        if isinstance(item, str):
            if item not in ("and", "or"):
                raise ValueError("unknown packaging marker conjunction")
            continue
        found.update(variables(item))
    return found

if "markers" in request:
    known = {"implementation_name", "implementation_version", "os_name", "platform_machine",
             "platform_python_implementation", "python_full_version", "python_version", "sys_platform"}
    selected = []
    for raw in strings(request["markers"]):
        marker = Marker(raw) if raw else None
        if marker and not variables(marker._markers).issubset(known):
            raise ValueError("registry marker requires facts absent from the publication target")
        selected.append(marker is None or marker.evaluate(environment))
    json.dump({"markers": selected}, sys.stdout)
    sys.exit(0)

metadata = {name: Parser().parsestr(text(raw)) for name, raw in mapping(request["metadata"]).items()}
def bind_extra(node: MarkerList | MarkerAtom, extra: str) -> str | bool:
    """Partially evaluate packaging 26.2's parsed tree, retaining target markers."""
    if isinstance(node, tuple):
        expression = " ".join(token.serialize() for token in node)
        if any(isinstance(token, Variable) and token.value == "extra" for token in node):
            return Marker(expression).evaluate({**environment, "extra": extra})
        return expression
    groups: list[list[str | bool]] = [[]]
    for item in node:
        if isinstance(item, str):
            if item == "or":
                groups.append([])
            elif item != "and":
                raise ValueError("unknown packaging marker conjunction")
        else:
            groups[-1].append(bind_extra(item, extra))
    alternatives = []
    for group in groups:
        if any(item is False for item in group):
            continue
        remaining = [item for item in group if isinstance(item, str)]
        if not remaining:
            return True
        alternatives.append("(" + " and ".join(remaining) + ")")
    return "(" + " or ".join(alternatives) + ")" if alternatives else False


project = canonicalize_name(text(request["project"]))
pending = deque((project, extra) for extra in ["", *strings(request["extras"])])
seen: set[tuple[str, str]] = set()
requirements: set[str] = set()
extras: dict[str, set[str]] = {}
while pending:
    name, extra = pending.popleft()
    if (name, extra) in seen:
        continue
    seen.add((name, extra))
    if len(seen) > 4096 or name not in metadata:
        raise ValueError("selected dependency metadata is absent or exceeds the bound")
    if extra:
        extras.setdefault(name, set()).add(extra)
    for raw in metadata[name].get_all("Requires-Dist", []):
        requirement = Requirement(raw)
        if requirement.marker:
            bound = bind_extra(requirement.marker._markers, extra)
            if bound is False:
                continue
            requirement.marker = None if bound is True else Marker(bound)
        requirements.add(str(requirement))
        target = canonicalize_name(requirement.name)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            pending.extend((target, value) for value in ["", *sorted(requirement.extras)])

json.dump({"requirements": sorted(requirements),
           "requires_python": metadata[project].get("Requires-Python", ""),
           "extras": {name: sorted(values) for name, values in sorted(extras.items())}}, sys.stdout)
