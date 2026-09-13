"""Build an exact CPU Runtime peer for CLI product tests without publishing it."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib


def main() -> None:
    repository, output = Path(sys.argv[1]).resolve(), Path(sys.argv[3]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "--verify", sys.argv[2] + "^{commit}"], text=True
    ).strip()
    archive = output / "source.tar"
    subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", "--output=" + str(archive), commit],
        check=True,
    )
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    recipe_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = hashlib.sha256(f"{archive_digest}:{recipe_digest}".encode()).hexdigest()
    source = output / "source"
    source.mkdir()
    with tarfile.open(archive) as package:
        package.extractall(source, filter="data")
    metadata = source / "pyproject.toml"
    original = metadata.read_text()
    released_version = tomllib.loads(original)["project"]["version"]
    version = released_version + "+dev.h" + identity
    metadata.write_text(original.replace(f'version = "{released_version}"', f'version = "{version}"', 1))
    (source / "src/cozy_runtime/_build_provenance.py").write_text(
        '"""CPU test peer from an immutable source archive."""\n'
        f'COMMIT = "{commit[:12]}"\nSOURCE_GIT_COMMIT = "{commit}"\n'
        f'SOURCE_ARCHIVE_SHA256 = "sha256:{archive_digest}"\n'
    )
    environment = dict(os.environ, COZY_RUNTIME_BUILD_KERNELS="0", SOURCE_DATE_EPOCH="946684800")
    subprocess.run(
        ["uv", "--no-config", "build", "--wheel", "--no-sources", "--python", "3.12",
         "--out-dir", str(output / "wheels"), str(source)],
        env=environment,
        check=True,
    )
    wheel, = (output / "wheels").glob("*.whl")
    manifest = {
        "source_commit": commit,
        "source_archive_sha256": archive_digest,
        "recipe_sha256": recipe_digest,
        "distribution": version,
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "scope": "CPU CLI product tests; no publication or native kernel qualification",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(wheel))


if __name__ == "__main__":
    main()
