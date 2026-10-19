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


def _store(cache_dir=None):
    return _ext.Store(_cache(cache_dir))


def _endpoint(endpoint=None):
    return endpoint or _settings.get("endpoint") or os.environ.get("TENSORFS_ENDPOINT", "https://tensorhub.com")


def open(name_or_path, *, revision=None, variant=None, endpoint=None, local_files_only=False, cache_dir=None, token=None, ca_der=None):
    store = _store(cache_dir)
    path = Path(name_or_path).expanduser()
    if path.exists():
        return store.open_path(path, variant)
    reference = str(name_or_path) + ("@" + revision if revision is not None else "")
    if local_files_only or str(name_or_path).startswith("local/"):
        return store.checkpoint(reference, variant)
    at = _endpoint(endpoint)
    credential = token if token is not None else _settings.get("token")
    return store.open_remote(str(name_or_path), revision, variant, at, credential, ca_der)


def snapshot_download(repo_id: str, *, revision=None, variant=None, components=None, cache_dir=None, local_dir=None, local_files_only=False, token=None, endpoint=None) -> str:
    ck = open(repo_id, revision=revision, variant=variant, endpoint=endpoint, local_files_only=local_files_only, cache_dir=cache_dir, token=token)
    dest = Path(local_dir) if local_dir is not None else _cache(cache_dir) / "snapshots" / ck.id.removeprefix("sha256:")
    return str(_store(cache_dir).snapshot(ck, dest, None if variant is None else [variant]))


def upload_folder(*, repo_id: str, folder_path, revision=None, token=None, endpoint=None, cache_dir=None, ca_der=None) -> str:
    at = _endpoint(endpoint)
    if repo_id.startswith("local/") or at == "local":
        return _store(cache_dir).upload_local(Path(folder_path), repo_id, revision)
    return _store(cache_dir).upload_folder(Path(folder_path), repo_id, revision, at, token if token is not None else _settings.get("token"), ca_der)


def convert(source: str | Sequence[str], *, converter=None, variant=None, cache_dir=None, token=None, progress=None):
    return _store(cache_dir).import_(source, converter=converter, variant=variant)


def export(src, dir, *, format="safetensors", dequantize=None) -> list[str]:
    if format != "safetensors":
        raise ValueError("format must be safetensors")
    return _store().export(src, dir, dequantize=dequantize)


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

    def component(self, name: str, *, cls: str, config: Mapping, config_file: str = "config.json") -> ComponentWriter:
        return ComponentWriter(self._native.component(name, cls, dict(config), None if config_file == "config.json" else config_file))

    def commit(self):
        if self._committed is None:
            for key, value in self.metadata.items():
                self._native.set_metadata(key, value)
            try:
                ck = self._native.commit()
            except _ext.Error as error:
                self._committed = getattr(error, "checkpoint", None)
                raise
            self._committed = ck
        return self._committed

    def abandon(self) -> None:
        self._native.abandon()

    def has(self, component, key) -> bool:
        return self._native.has(component, key)
