from pathlib import Path
from typing import Any, Literal, TypeAlias

from typing_extensions import Self

COMMANDS: frozenset[str]

class Malformed(ValueError): ...

class _Wire:
    def __init__(self, **fields: Any) -> None: ...
    def json(self) -> str: ...
    def replace(self, **changes: Any) -> Self: ...

class Hello: ...
class Shutdown: ...

class TensorFSOptions:
    endpoint: str
    token: str
    expires_unix_ms: int
    ca_der: bytes

class MemoSettings(_Wire):
    process_bytes: int
    entry_bytes: int

class CallInterface(_Wire):
    module: str
    export: str
    interface_path: Path
    self_call: bool
    interface_document: Any | None
    builtin: str | None

class Construction(_Wire):
    path: str
    manifest: str
    components: list[str]

class Start(_Wire):
    devices: str
    application: str
    package_interface: Path
    sequence_parallel_degree: int
    import_only: bool
    call_interfaces: list[CallInterface]
    constructions: list[Construction]
    leases: int | None
    memory_protocol: int
    memo: MemoSettings | None

class Adapter(_Wire):
    model_id: str
    ref: str
    scale: float
    kind: str
    component: str
    source_component: str
    family: str

class Binding(_Wire):
    application: str
    package_interface: str
    model_class: str
    model_binding_path: str
    model_parameter_name: str
    model_parameter_names: list[str]
    component: str
    components: list[str]
    snapshots: dict[str, str]
    store: str
    snapshot: str
    release: str
    variant: str
    development: bool
    custody: str
    placement: str
    package: str
    model: str
    adapters: list[Adapter]
    homes: dict[str, int]

class ModelLoad(_Wire):
    binding: Binding

class Load(_Wire):
    construction: str
    tensorfs: TensorFSOptions | None
    actor: str
    devices: str
    sequence_parallel_degree: int
    binding: Binding
    models: list[ModelLoad]
    attention_pin: str

class Activate(_Wire):
    construction: str

class InputMetadata(_Wire):
    input_id: str
    media_type: str
    digest: str
    length: int
    order: int

class InputFile(_Wire):
    local: Path
    media_type: str
    digest: str
    length: int
    order: int

class PrepareRequest(_Wire):
    request_id: str
    construction: str
    entrypoint: str
    payload: Any
    attention_kernel: str
    bits: int
    input_metadata: dict[str, InputMetadata]

class Invoke(_Wire):
    tensorfs: TensorFSOptions | None
    max_output_bytes: int
    request_id: str
    construction: str
    entrypoint: str
    spool: Path
    deadline_s: float | None
    attention_kernel: str
    bits: int
    inputs: dict[str, InputFile]
    trees: dict[str, tuple[Path, str]]
    sources: dict[str, str]
    weights: dict[str, int]
    destination: str
    actor: str
    operation: str
    family: str

class Map(_Wire):
    construction: str
    level: Literal["gpu", "ram"]

class Unload(_Wire):
    construction: str

class Probe(_Wire):
    collect: bool

class Fork(_Wire):
    socket: Path
    root: Path
    environment: dict[str, str]

class Bearer(_Wire):
    actor: str
    tensorfs: TensorFSOptions

class ByteGrant(_Wire):
    output_id: str
    kind: str
    digest: str
    local: Path
    length: int
    content_bytes: int
    media_type: str

class CallProgress(_Wire):
    sequence: int
    payload: Any

class Answer(_Wire):
    event: str
    seq: int
    ok: bool
    code: str
    detail: str
    descriptor: bool
    sha256: str
    length: int
    digest: str
    sequence: int
    state: str
    child_request_id: str
    result: str
    byte_grants: list[ByteGrant]
    progress: CallProgress | None
    traceback: str
    local: str
    cost_ms: float
    disputed: bool
    stored: bool
    reason: str

Command: TypeAlias = Hello | Shutdown | Start | Load | Activate | PrepareRequest | Invoke | Map | Unload | Probe | Fork | Bearer

def decode_command(data: bytes) -> Command: ...
def decode_answer(data: bytes) -> Answer: ...

class ChildEvents(_Wire): ...

class ChildCall(_Wire):
    call_index: int
    module: str
    export: str
    payload: str
    progress_label: str

class ChildPoll(_Wire):
    call_index: int

class ChildCancel(_Wire):
    call_index: int

class ChildForget(_Wire):
    call_index: int

class GpuRelease(_Wire): ...

class ModelPrefetch(_Wire):
    module: str
    export: str
    payload: str

class PublishPart(_Wire):
    local: str
    duration_us: int

class Publish(_Wire):
    output: str
    label: str
    asset_ref: str
    asset_kind: str
    media_type: str
    size_bytes: int
    digest: str
    parts: list[PublishPart]

class CheckpointItem(_Wire):
    output: str
    repo_id: str
    revision: str
    tag: str
    variant: str

class StageMemoLookup(_Wire):
    key: str
    stage: str
    numerics: str

class StageMemoStore(_Wire):
    key: str
    stage: str
    numerics: str
    local: str
    sha256: str
    length: int
    cost_ms: float
    reason: str

Request: TypeAlias = (
    ChildEvents | ChildCall | ChildPoll | ChildCancel | ChildForget | GpuRelease | ModelPrefetch
    | Publish | CheckpointItem | StageMemoLookup | StageMemoStore
)
