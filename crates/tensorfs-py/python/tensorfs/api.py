"""The public component API; storage and admission stay in the native library."""

from __future__ import annotations

import io
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import _ext

_settings: dict[str, Any] = {}


def configure(*, endpoint=None, cache_dir=None, token=None, ca_der=None) -> None:
    global _settings
    _settings = {**_settings, **{k: v for k, v in locals().items() if v is not None}}


# What the caller named; None leaves the store and endpoint to the native rule (TENSORFS_HOME,
# TENSORFS_ENDPOINT and their defaults), the one every TensorFS surface follows.
def _cache(cache_dir=None) -> Path | None:
    root = cache_dir or _settings.get("cache_dir")
    return None if root is None else Path(root).expanduser()


def _endpoint(endpoint=None):
    return endpoint or _settings.get("endpoint")


def _client(cache_dir=None, endpoint=None, token=None, *, configured_token=True):
    credential = token if token is not None else _settings.get("token") if configured_token else None
    return _ext._Client(_cache(cache_dir), endpoint=_endpoint(endpoint), token=credential, ca_der=_settings.get("ca_der"))


def open(name_or_path, *, revision=None, variant=None, endpoint=None, local_files_only=False, cache_dir=None, token=None, lazy=True):
    return _client(cache_dir, endpoint, token).open(
        os.fspath(name_or_path), revision=revision, variant=variant, local_files_only=local_files_only, lazy=lazy
    )


def materialize(source: str | _ext.Checkpoint, *, local_dir=None, format="cozytensors", decode=False, revision=None, variant=None, components=None, cache_dir=None, local_files_only=False, token=None, endpoint=None) -> str:
    if format not in {"cozytensors", "safetensors"}:
        raise ValueError("format must be cozytensors or safetensors")
    if format == "cozytensors" and decode:
        raise ValueError("decode requires format='safetensors'")
    if isinstance(source, _ext.Checkpoint):
        if components is not None:
            raise ValueError("components selection requires a repository source in cozytensors format")
        if local_dir is None:
            raise ValueError("a checkpoint requires an explicit local_dir")
        if local_files_only or any(value is not None for value in
                                   (revision, variant, cache_dir, token, endpoint)):
            raise ValueError("repository and client options do not apply to a checkpoint")
        return str(_ext._materialize(source, local_dir, format=format, decode=decode))
    if not isinstance(source, str):
        raise TypeError("source must be a repository reference or Checkpoint")
    if format == "cozytensors":
        return str(_client(cache_dir, endpoint, token).materialize(
            (source, revision, variant), components=components,
            local_dir=local_dir, local_files_only=local_files_only,
        ))
    if components is not None:
        raise ValueError("components selection is not supported for safetensors materialization")
    if local_dir is None:
        raise ValueError("safetensors materialization requires an explicit local_dir")
    checkpoint = _client(cache_dir, endpoint, token).open(
        source, revision=revision, variant=variant, local_files_only=local_files_only,
    )
    try:
        return str(_ext._materialize(checkpoint, local_dir, format=format, decode=decode))
    finally:
        checkpoint.close()


def download_file(repo_id: str, filename: str, *, revision=None, cache_dir=None, local_dir=None, local_files_only=False, token=None, endpoint=None) -> str:
    return str(_client(cache_dir, endpoint, token).download_file(
        repo_id, filename, revision=revision, local_dir=local_dir, local_files_only=local_files_only,
    ))


def upload_folder(*, repo_id: str, folder_path, revision="main", delete_patterns=None, token=None, endpoint=None, cache_dir=None) -> str:
    patterns = [delete_patterns] if isinstance(delete_patterns, str) else list(delete_patterns or [])
    return _client(cache_dir, endpoint, token).upload_folder(Path(folder_path), repo_id, revision, patterns)


def convert_to_cozytensors(source: str | Sequence[str], *, converter=None, source_variant=None, cache_dir=None, token=None, progress=None):
    return _client(cache_dir, token=token, configured_token=False).convert_to_cozytensors(
        source, converter=converter, source_variant=source_variant, progress=progress,
    )


class CheckpointFile(io.RawIOBase):
    """The checkpoint as one read-only, seekable file: its header, then its chunks and assets in
    header order at 4 KiB-aligned offsets, assembled from the store on demand; nothing is written."""

    def __init__(self, native):
        super().__init__()
        self._native = native

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        return self._native.readinto(buffer)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._native.seek(offset, whence)

    def tell(self) -> int:
        return self._native.tell()

    def close(self) -> None:
        self._native.close()
        super().close()


class Endpoint:
    def __init__(self, url, token=None, *, ca_der=None, cache_dir=None):
        self._native = _ext.Endpoint(url, token, ca_der=ca_der, cache_dir=_cache(cache_dir))

    def repo(self, repo_id):
        return self._native.repo(repo_id)

    def commit(self, repo_id, revision):
        return self._native.commit(repo_id, revision)

    def update_repo(self, repo_id, change):
        return self._native.update_repo(repo_id, change)


class ComponentWriter:
    def __init__(self, native):
        self._native = native

    def put(self, key, data, *, dtype=None, shape=None) -> None:
        self._native.put_array(key, data, dtype=dtype, shape=shape)

    def put_encoded(self, key, *, dtype, shape, scheme, params: Mapping, parts: Mapping) -> None:
        self._native.put_encoded(key, dtype, shape, scheme, parts, dict(params))

    def copy(self, key, source, *, as_key=None) -> None:
        self._native.copy_component(key, source, as_key)

    def asset(self, name, data) -> None:
        self._native.asset(name, data)

    def order(self, keys) -> None:
        self._native.order(keys)

    def has(self, key) -> bool:
        return self._native.has(key)


class Writer:
    def __init__(self, dest=None, *, variant=None, base=None, resume=None, repo_id=None, revision=None, endpoint=None, token=None, cache_dir=None):
        self._native = _ext.Writer(dest, variant=variant, base=base, resume=resume, cache_dir=_cache(cache_dir), repo_id=repo_id, revision=revision, endpoint=_endpoint(endpoint), token=token if token is not None else _settings.get("token"), ca_der=_settings.get("ca_der"))
        self._resume = resume
        self._committed = None
        self.local_checkpoint = None
        self._failed = None
        self.metadata = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.commit()
        elif self._resume is None:
            self.abandon()

    def pipeline(self, cls=None, config: Mapping | None = None) -> None:
        self._native.pipeline(cls, dict(config) if config is not None else {})

    def component(self, name: str, *, cls=None, config: Mapping | None = None, config_file=None) -> ComponentWriter:
        return ComponentWriter(self._native.component(name, cls, dict(config) if config is not None else {}, config_file))

    def commit(self):
        if self._failed is not None:
            raise self._failed
        if self._committed is None:
            for key, value in self.metadata.items():
                self._native.set_metadata(key, value)
            try:
                ck = self._native.commit()
            except _ext.Error as error:
                self.local_checkpoint = getattr(error, "checkpoint", None)
                self._failed = error
                raise
            self._committed = ck
        return self._committed

    def abandon(self) -> None:
        self._native.abandon()

    def has(self, component, key) -> bool:
        return self._native.has(component, key)
