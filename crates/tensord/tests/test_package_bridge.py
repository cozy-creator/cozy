"""Real installed-package gates: static description, generation records and holds."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cozy_runtime  # noqa: F401  (required: absence must fail, not skip)
import msgspec
import pytest

from tensord_client.packages import GenerationHold, collect, describe, install, read_metadata
from tensord_client.package_records import GENERATION_DECODER, Generation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "cpu_classifier"


@pytest.fixture(scope="module")
def generation(tmp_path_factory):
    output = tmp_path_factory.mktemp("immutable-package")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output / "client"), str(ROOT)],
                   check=True)
    client = next((output / "client").glob("*.whl"))
    generation = install(FIXTURE, output / "generations", client)
    with GenerationHold(Path(generation.python).parents[2]) as held:
        yield held


def test_static_description_does_not_execute_package_top_level(tmp_path):
    source = tmp_path / "package"
    shutil.copytree(FIXTURE, source)
    module = source / "cpu_classifier" / "__init__.py"
    module.write_text(module.read_text() + '\nraise RuntimeError("import must not occur")\n')
    interface = describe(source)
    assert isinstance(interface, msgspec.Raw)
    assert b"classify" in bytes(interface)
    assert b"cpu_classifier:app" in bytes(interface)


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


def test_sdk_interface_remains_opaque_through_generation_boundary():
    document = msgspec.Raw(b'{"future":{"unknown":[1,true]},"entrypoints":[]}')
    original = Generation("g", "p", "1", "p:app", "/g/env/bin/python", [], document)
    replay = GENERATION_DECODER.decode(msgspec.json.encode(original))
    assert isinstance(replay.interface, msgspec.Raw)
    assert bytes(replay.interface) == bytes(document)


def test_a_live_generation_cannot_be_collected(generation):
    assert "cozy-runtime" in {entry.name for entry in generation.dependencies}
    assert not collect(Path(generation.python).parents[2])


def test_generation_collected_only_after_last_holder(tmp_path):
    root = tmp_path / "generation"
    root.mkdir()
    (root / ".hold").touch()
    (root / "generation.json").write_bytes(msgspec.json.encode({
        "identity": "generation", "package": "p", "version": "1", "application": "p:app",
        "python": str(root / "env/bin/python"), "dependencies": [], "interface": {},
    }))
    with GenerationHold(root):
        assert not collect(root)
    assert collect(root)
    assert not root.exists()
