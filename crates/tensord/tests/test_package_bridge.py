"""Real installed-package gates: static description and typed metadata."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cozy_runtime  # noqa: F401  (required: absence must fail, not skip)
import msgspec
import pytest

from tensord_client.packages import read_metadata

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "cpu_classifier"


def test_static_description_does_not_execute_package_top_level(tmp_path):
    """What the machine asks of an environment's client (`environments::describe_project`)."""
    source = tmp_path / "package"
    shutil.copytree(FIXTURE, source)
    module = source / "cpu_classifier" / "__init__.py"
    module.write_text(module.read_text() + '\nraise RuntimeError("import must not occur")\n')
    request = json.dumps({"kind": "describe", "project": str(source), "environment_python": None})
    reply = json.loads(subprocess.check_output([sys.executable, "-m", "tensord_client.runtime_describe"], input=request.encode()))
    assert reply["kind"] == "described", reply
    interface = json.dumps(reply["interface"])
    assert "classify" in interface and "cpu_classifier:app" in interface


def test_consumed_metadata_is_typed_and_unknown_extras_are_harmless(tmp_path):
    source = tmp_path / "metadata"
    shutil.copytree(FIXTURE, source)
    project = source / "pyproject.toml"
    project.write_text(project.read_text() + '\n[future.optional]\nnested = {x = [1, 2]}\n')
    metadata = read_metadata(source)
    assert metadata.name == "tensord-cpu-classifier"
    assert metadata.application == "cpu_classifier:app"
    project.write_text(project.read_text().replace('version = "0.1.0"', "version = 42"))
    with pytest.raises(msgspec.ValidationError):
        read_metadata(source)
