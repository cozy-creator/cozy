"""The PEP 517 backend: maturin, refusing at once when Rust is missing.

pip reaches it only when no prebuilt wheel fits. Unpatched, maturin would download a Rust
toolchain into a temporary directory and fail later, deep in a compile, on a missing linker.
"""

import platform
import shutil
import sys

import maturin
from maturin import build_sdist, get_requires_for_build_sdist  # noqa: F401


def _rust() -> None:
    if shutil.which("cargo") is None:
        sys.exit(
            f"tensorfs has no prebuilt wheel for CPython {platform.python_version()} on "
            f"{platform.system()} {platform.machine()}, so the installer is building it from source, which needs "
            "Rust (https://rustup.rs) and a C linker. Prebuilt wheels cover CPython 3.10 and later on "
            "Linux x86_64/aarch64 (glibc 2.17+), macOS arm64/x86_64 and Windows x86_64/arm64."
        )


def get_requires_for_build_wheel(config_settings=None):
    _rust()
    return maturin.get_requires_for_build_wheel(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _rust()
    return maturin.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _rust()
    return maturin.build_wheel(wheel_directory, config_settings, metadata_directory)


get_requires_for_build_editable = get_requires_for_build_wheel
prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _rust()
    return maturin.build_editable(wheel_directory, config_settings, metadata_directory)
