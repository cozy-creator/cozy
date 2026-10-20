from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from types import TracebackType
from typing import Protocol
from typing_extensions import Buffer

from ._ext import Checkpoint, Component, TokenProvider

StrPath = str | PathLike[str]
Token = str | TokenProvider | Callable[[str, str, Mapping[str, str], str | None], Mapping[str, str]]

class DLPack(Protocol):
    def __dlpack__(self) -> object: ...

def configure(*, endpoint: str | None = None, cache_dir: StrPath | None = None, token: Token | None = None,
              ca_der: bytes | None = None) -> None: ...
def open(name_or_path: StrPath, *, revision: str | None = None, variant: str | None = None,
         endpoint: str | None = None, local_files_only: bool = False, cache_dir: StrPath | None = None,
         token: Token | None = None, lazy: bool = True) -> Checkpoint: ...
def materialize(source: str | Checkpoint, *, local_dir: StrPath | None = None,
                format: str = "cozytensors", dequantize: str | None = None,
                revision: str | None = None, variant: str | None = None,
                components: Sequence[str] | None = None, cache_dir: StrPath | None = None,
                local_files_only: bool = False, token: Token | None = None,
                endpoint: str | None = None) -> str: ...
def download_file(repo_id: str, filename: str, *, revision: str | None = None,
                 cache_dir: StrPath | None = None, local_dir: StrPath | None = None,
                 local_files_only: bool = False, token: Token | None = None,
                 endpoint: str | None = None) -> str: ...
def upload_folder(*, repo_id: str, folder_path: StrPath, revision: str = "main",
                  token: Token | None = None, endpoint: str | None = None,
                  cache_dir: StrPath | None = None) -> str:
    """Makes the folder a commit of `repo_id` and moves tag `revision` to it (a new tag is private,
    an existing one keeps its flags). The commit's id, `sha256:<tree hex>`."""
def convert_to_cozytensors(source: str | Sequence[str], *, converter: str | None = None, variant: str | None = None,
            cache_dir: StrPath | None = None, token: Token | None = None,
            progress: Callable[[str, int, int], object] | None = None) -> Checkpoint: ...

class Endpoint:
    def __init__(self, url: str, token: Token | None = None, *, ca_der: bytes | None = None, cache_dir: StrPath | None = None) -> None: ...
    def repo(self, repo_id: str) -> tuple[dict[str, object], str] | None: ...
    def commit(self, repo_id: str, revision: str) -> tuple[str, str | None]:
        """`revision` (a tag, a commit or a header id, or a commit no tag names) as (commit id, tag)."""
    def update_repo(self, repo_id: str, change: Callable[[dict[str, object] | None], dict[str, object] | None]) -> dict[str, object]: ...

class ComponentWriter:
    def put(self, key: str, data: Buffer | DLPack, *, dtype: str | None = None, shape: Sequence[int] | None = None) -> None: ...
    def put_encoded(self, key: str, *, dtype: str, shape: Sequence[int], scheme: str,
                    params: Mapping[str, object], parts: Mapping[str, tuple[str, Sequence[int], Buffer]]) -> None: ...
    def copy(self, key: str, source: Component, *, as_key: str | None = None) -> None: ...
    def asset(self, name: str, data: Buffer) -> None: ...
    def order(self, keys: Sequence[str]) -> None: ...
    def has(self, key: str) -> bool: ...

class Writer:
    metadata: dict[str, object]
    local_checkpoint: Checkpoint | None
    def __init__(self, dest: StrPath | None = None, *, variant: str | None = None,
                 base: Checkpoint | None = None, resume: str | None = None,
                 repo_id: str | None = None, revision: str | None = None, endpoint: str | None = None,
                 token: Token | None = None, cache_dir: StrPath | None = None) -> None:
        """`dest` is a header file, a handle into the shared local store that roots its data while
        it holds this header; `materialize` makes the portable copy. With `repo_id`, `commit()`
        makes a commit of that repo and moves tag `revision` (default `main`; a new tag is private)
        to it; `variant` names `model.<variant>.cozytensors` in it. Both need `repo_id`."""
    def __enter__(self) -> Writer: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None: ...
    def pipeline(self, cls: str, config: Mapping[str, object]) -> None: ...
    def component(self, name: str, *, cls: str, config: Mapping[str, object],
                  config_file: str | None = None) -> ComponentWriter: ...
    def commit(self) -> Checkpoint: ...
    def abandon(self) -> None: ...
    def has(self, component: str, key: str) -> bool: ...
