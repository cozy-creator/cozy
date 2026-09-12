"""Read metadata only. Never import code or execute the captured environment."""

import json
import sys
from collections import deque
from email.parser import Parser

from packaging.markers import Marker, Variable, default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

request = json.load(sys.stdin)
python = Version(request["python"])
if len(python.release) != 3:
    raise ValueError("requirement evaluation needs the measured Python patch")
environment = default_environment()
environment.update(
    implementation_name="cpython", implementation_version=request["python"], os_name="posix",
    platform_machine="x86_64", platform_python_implementation="CPython",
    python_full_version=request["python"], python_version=".".join(request["python"].split(".")[:2]),
    sys_platform="linux",
)


def variables(node):
    if isinstance(node, tuple):
        return {token.value for token in node if isinstance(token, Variable)}
    return set().union(*(variables(item) for item in node if item not in ("and", "or")))

if "requirements" in request:
    active = set()
    for raw in request["requirements"]:
        requirement = Requirement(raw)
        known = {"implementation_name", "implementation_version", "os_name", "platform_machine",
                 "platform_python_implementation", "python_full_version", "python_version", "sys_platform"}
        if requirement.marker and not variables(requirement.marker._markers).issubset(known):
            # Inventory carries no kernel release or selected public-package extra.
            # Keep those markers for worker admission rather than inventing a value.
            active.add(str(requirement))
            continue
        if requirement.marker is None or requirement.marker.evaluate(environment):
            requirement.marker = None
            active.add(str(requirement))
    json.dump({"requirements": sorted(active), "extras": {}}, sys.stdout)
    sys.exit(0)

metadata = {name: Parser().parsestr(raw) for name, raw in request["metadata"].items()}
image = request["image"]


def image_owned(name):
    return name in image["distributions"] or any(name.startswith(p) for p in image["prefixes"])


def bind_extra(node, extra):
    """Partially evaluate packaging 26.2's parsed tree, retaining target markers."""
    if isinstance(node, tuple):
        expression = " ".join(token.serialize() for token in node)
        if any(isinstance(token, Variable) and token.value == "extra" for token in node):
            return Marker(expression).evaluate({**environment, "extra": extra})
        return expression
    groups = [[]]
    for item in node:
        if item == "or":
            groups.append([])
        elif item != "and":
            groups[-1].append(bind_extra(item, extra))
    alternatives = []
    for group in groups:
        if any(item is False for item in group):
            continue
        remaining = [item for item in group if item is not True]
        if not remaining:
            return True
        alternatives.append("(" + " and ".join(remaining) + ")")
    return " or ".join(alternatives) if alternatives else False


pending = deque((request["project"], extra) for extra in ["", *request["extras"]])
seen = set()
requirements = set()
extras = {}
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
        if not image_owned(target) and (
            requirement.marker is None or requirement.marker.evaluate(environment)
        ):
            pending.extend((target, value) for value in ["", *sorted(requirement.extras)])

json.dump({"requirements": sorted(requirements),
           "extras": {name: sorted(values) for name, values in sorted(extras.items())}}, sys.stdout)
