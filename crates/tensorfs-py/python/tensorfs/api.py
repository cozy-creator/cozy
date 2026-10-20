"""The public component API; storage and admission stay in the native library."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import _ext

_settings: dict[str, Any] = {}


def configure(*, endpoint=None, cache_dir=None, token=None) -> None:
    global _settings
    _settings = {**_settings, **{k: v for k, v in locals().items() if v is not None}}


def _cache(cache_dir=None) -> Path:
    return Path(cache_dir or _settings.get("cache_dir") or os.environ.get("TENSORFS_HOME") or Path.home() / ".cache/tensorfs").expanduser()


def _endpoint(endpoint=None):
    return endpoint or _settings.get("endpoint") or os.environ.get("TENSORFS_ENDPOINT", "https://tensorhub.com")


def _client(cache_dir=None, endpoint=None, token=None, ca_der=None, *, configured_token=True):
    credential = token if token is not None else _settings.get("token") if configured_token else None
    return _ext._Client(_cache(cache_dir), endpoint=_endpoint(endpoint), token=credential, ca_der=ca_der)


def open(name_or_path, *, revision=None, variant=None, endpoint=None, local_files_only=False, cache_dir=None, token=None, ca_der=None):
    return _client(cache_dir, endpoint, token, ca_der).open(
        os.fspath(name_or_path), revision=revision, variant=variant, local_files_only=local_files_only
    )


def snapshot_download(repo_id: str, *, revision=None, variant=None, components=None, cache_dir=None, local_dir=None, local_files_only=False, token=None, endpoint=None, ca_der=None) -> str:
    return str(_client(cache_dir, endpoint, token, ca_der).snapshot_download(
        (repo_id, revision, variant), components=components,
        local_dir=local_dir, local_files_only=local_files_only,
    ))


def hub_download(repo_id: str, filename: str, *, revision=None, cache_dir=None, local_dir=None, local_files_only=False, token=None, endpoint=None, ca_der=None) -> str:
    return str(_client(cache_dir, endpoint, token, ca_der).hub_download(
        repo_id, filename, revision=revision, local_dir=local_dir, local_files_only=local_files_only,
    ))


def upload_folder(*, repo_id: str, folder_path, revision=None, token=None, endpoint=None, cache_dir=None, ca_der=None) -> str:
    return _client(cache_dir, endpoint, token, ca_der).upload_folder(Path(folder_path), repo_id, revision)


def convert(source: str | Sequence[str], *, converter=None, variant=None, cache_dir=None, token=None, progress=None):
    return _client(cache_dir, token=token, configured_token=False).convert(
        source, converter=converter, variant=variant, progress=progress,
    )


def resolve_source(source: str, *, token=None) -> str:
    return _client(token=token, configured_token=False).resolve_source(source)


def export(src, dir, *, format="safetensors", dequantize=None) -> list[str]:
    if format not in {"safetensors", "cozytensors"}:
        raise ValueError("format must be safetensors or cozytensors")
    return _ext._export(src, dir, format=format, dequantize=dequantize)


class Endpoint:
    def __init__(self, url, token=None, *, ca_der=None, cache_dir=None):
        self._native = _ext.Endpoint(url, token, ca_der=ca_der, cache_dir=_cache(cache_dir) if url == "local" or cache_dir is not None else None)

    def repo(self, repo_id):
        return self._native.repo(repo_id)

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
    def __init__(self, dest=None, *, variant=None, base=None, resume=None, repo_id=None, endpoint=None, token=None, cache_dir=None, ca_der=None):
        self._native = _ext.Writer(dest, variant=variant, base=base, resume=resume, cache_dir=_cache(cache_dir), repo_id=repo_id, endpoint=_endpoint(endpoint), token=token if token is not None else _settings.get("token"), ca_der=ca_der)
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

    def pipeline(self, cls: str, config: Mapping) -> None:
        self._native.pipeline(cls, dict(config))

    def component(self, name: str, *, cls: str, config: Mapping) -> ComponentWriter:
        return ComponentWriter(self._native.component(name, cls, dict(config)))

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
