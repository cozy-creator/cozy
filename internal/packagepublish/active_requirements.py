"""Read metadata only. Never import code or execute the captured environment."""

import json
import sys
from collections import deque
from email.parser import Parser

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

request = json.load(sys.stdin)
metadata = {name: Parser().parsestr(raw) for name, raw in request["metadata"].items()}
image = request["image"]


def image_owned(name):
    return name in image["distributions"] or any(name.startswith(p) for p in image["prefixes"])


# Private worker captures are CPython 3.12 on Linux amd64. Use the same marker
# profile as worker admission, independently of this tool's own patch version.
environment = default_environment()
environment.update(
    implementation_name="cpython", implementation_version="3.12.0", os_name="posix",
    platform_machine="x86_64", platform_python_implementation="CPython",
    python_full_version="3.12.0", python_version="3.12", sys_platform="linux",
)
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
        if requirement.marker and not requirement.marker.evaluate({**environment, "extra": extra}):
            continue
        requirement.marker = None
        requirements.add(str(requirement))
        target = canonicalize_name(requirement.name)
        if not image_owned(target):
            pending.extend((target, value) for value in ["", *sorted(requirement.extras)])

json.dump({"requirements": sorted(requirements),
           "extras": {name: sorted(values) for name, values in sorted(extras.items())}}, sys.stdout)
