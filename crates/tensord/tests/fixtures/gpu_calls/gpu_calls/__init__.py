"""One path for every call on a GPU machine: a deviceless root calls a GPU child, a GPU root
calls a child on its own GPU, a raw-GPU call runs beside a warm executor without ending it, and
a deviceless job calls children whose model slot declares one GPU. Each answers its process and
device so a caller sees where it ran."""
from __future__ import annotations

import asyncio
import hashlib
import os

import msgspec

from cozy_runtime.author import App, Config, Context, Loader, Model, invocable, uses_components

app = App()


class Ran(msgspec.Struct):
    device: str
    pid: int
    total: float


class Orchestrated(msgspec.Struct):
    root: Ran
    child: Ran


class Warm(msgspec.Struct):
    pid: int
    #: The calls this process has served, its retained CUDA tensor intact through them.
    served: int


def _torch():
    import torch

    return torch


@invocable
async def gpu_child(ctx: Context, *, hold_s: float = 0.0) -> Ran:
    """Undeclared: the machine's GPU."""
    if ctx.device.type != "cuda":
        raise ValueError("a GPU child got no GPU")
    tensor = _torch().full((4096,), 2.0, device=str(ctx.device))
    for _ in range(max(0, int(hold_s * 20))):
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.05)
    return Ran(ctx.device.type, os.getpid(), float(tensor.sum().item()))


app.entrypoint(internal=True)(gpu_child)


class Request(msgspec.Struct):
    hold_s: float = 0.0


@app.entrypoint(accelerator=False)
async def cpu_root(ctx: Context, payload: Request) -> Orchestrated:
    """Deviceless: it never opens CUDA, and calls a GPU child."""
    if ctx.device.type != "cpu":
        raise ValueError("a deviceless root got a device")
    child = await gpu_child(hold_s=payload.hold_s)
    return Orchestrated(Ran("cpu", os.getpid(), 0.0), child)


@app.entrypoint
async def gpu_root(ctx: Context, payload: Request) -> Orchestrated:
    """On the GPU, holding a tensor while its child runs on the same GPU."""
    if ctx.device.type != "cuda":
        raise ValueError("a GPU root got no GPU")
    retained = _torch().ones(4096, device=str(ctx.device))
    child = await gpu_child(hold_s=payload.hold_s)
    if float(retained.sum().item()) != 4096 or child.total != 8192:
        raise ValueError("the child changed the root's tensor")
    if child.pid == os.getpid():
        raise ValueError("the child ran in its root's process")
    return Orchestrated(Ran(ctx.device.type, os.getpid(), 4096.0), child)


_held: list[object] = []


@app.entrypoint
async def warm(ctx: Context, payload: Request) -> Warm:
    """Keeps 256 MiB of CUDA memory across its calls, as a loaded model would."""
    if not _held:
        _held.append(_torch().ones(64 << 20, device=str(ctx.device)))
    _held.append(None)
    if float(_held[0].sum().item()) != 64 << 20:
        raise ValueError("the retained tensor changed")
    return Warm(os.getpid(), len(_held) - 1)


@app.entrypoint
async def raw(ctx: Context, payload: Request) -> Ran:
    """A raw-GPU call: its own torch memory, no model."""
    tensor = _torch().full((32 << 20,), 3.0, device=str(ctx.device))
    return Ran(ctx.device.type, os.getpid(), float(tensor.sum().item()))


class Graph:
    """The checkpoint's one tensor, `model.weight`, of the config's `n` elements."""

    def __init__(self, n: int) -> None:
        module = _torch().nn.Module()
        module.weight = _torch().nn.Parameter(_torch().empty(n), requires_grad=False)
        self.components = {"model": module}


def _graph(config: Config) -> Graph:
    return Graph(config.as_int("n", 2048))


class Tiny(Model[Graph]):
    """A constructed slot that declares no group: one GPU, as birefnet's and the describer's."""

    graph: Graph
    #: Each model file its load was given (configs, tokenizer, assets): SHA-256 by manifest path.
    assets: dict[str, str]

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=_graph)
        files = loader.assets
        self.assets = {n: hashlib.sha256(files.read(n)).hexdigest() for n in files.names()}

    @uses_components("model")
    def weight(self) -> object:
        return self.graph.components["model"].weight


@invocable
async def modeled(ctx: Context, *, model: Tiny, hold_s: float = 0.0) -> Ran:
    if ctx.device.type != "cuda":
        raise ValueError("a modeled child got no GPU")
    for _ in range(max(0, int(hold_s * 20))):
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.05)
    return Ran(str(ctx.device), os.getpid(), 0.0)


app.entrypoint(internal=True)(modeled)


@invocable
async def resident(ctx: Context, *, model: Tiny) -> Ran:
    """Reads its model's weight on the GPU."""
    weight = model.weight()
    return Ran(str(weight.device), os.getpid(), float(weight.sum().item()))


app.entrypoint(resident)


class Files(msgspec.Struct):
    #: The weight's length: the `n` its `model/config.json` gave the construction.
    n: int
    assets: dict[str, str]


@invocable
async def files(ctx: Context, *, model: Tiny) -> Files:
    """What its model's load read through the machine: the config and every asset."""
    return Files(int(model.weight().numel()), model.assets)


app.entrypoint(files)


class Children(msgspec.Struct):
    children: list[Ran]


@app.entrypoint(accelerator=False)
async def cpu_job(ctx: Context, payload: Request) -> Children:
    """H3 long_form's shape: deviceless, holding its run's GPUs while one-GPU children run."""

    async def child() -> Ran:
        return await modeled(hold_s=payload.hold_s)

    return Children(list(await asyncio.gather(child(), child())))
