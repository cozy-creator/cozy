#!/usr/bin/env python3
"""`tensorfs.varena.cuts` on the CPU, through real torch and diffusers modules (xs-046 B3).

The device is a `Card`: a dispatch mode below the catch's layer under which any op whose new
results exceed `limit` bytes runs out of memory, as a GPU allocator does; its `Room` raises the
limit when rungs 1-2 or rung 4 run. Every cut is checked against the same call run whole
(float64, so a cut's rounding shows), and every warm run against its first:

- FFN token chunks: diffusers FeedForward (SwiGLU, GEGLU, GELU), a fused SwiGLU subclass, a block
  that calls its FFN's parts itself, and a part whose result does not join (runs whole);
- attention by head groups then query rows, with GQA and a mask;
- bands: convolution (stride, dilation, 3D), padding, nearest upsampling and group norm, and a
  whole diffusers VAE decode cut wherever the card says;
- the block catch: a bare block out of memory runs again with its generators and declared state
  put back; a writer runs under the layer; an op asks its FFN to cut finer before rung 4;
- the lattice ends: past every rung, `exhausted`; a package `split` loop.

    CUDA_VISIBLE_DEVICES= python scripts/varena-cuts.py
Runs with any python that has torch and diffusers, and `PYTHONPATH=<worktree>/python`.
"""

import functools
import sys

import torch
from diffusers import AutoencoderKL
from diffusers.models.activations import SwiGLU
from diffusers.models.attention import FeedForward
from tensorfs.varena import cuts
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_default_dtype(torch.float64)
CPU = torch.device("cpu")
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


def same(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-10) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and float((a - b).abs().max()) <= tol


FACTORIES = frozenset(
    {
        "empty",
        "empty_like",
        "empty_strided",
        "new_empty",
        "new_empty_strided",
        "new_zeros",
        "zeros_like",
        "full_like",
        "new_full",
    }
)


class Card(TorchDispatchMode):
    """A device on which an op needs its results and a working copy of its largest operand
    (a workspace, a contiguous copy): past `limit` bytes it runs out of memory, as a GPU
    allocator does. `only`: the op kinds it limits (others always fit)."""

    def __init__(self, limit: int, only: frozenset[str] = frozenset()) -> None:
        super().__init__()
        self.limit, self.only, self.refused, self.largest = limit, only, 0, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        out = func(*args, **(kwargs or {}))
        if self.only and func.overloadpacket.__name__ not in self.only:
            return out
        if any(r.alias_info is not None for r in func._schema.returns):
            return out
        reads = func.overloadpacket.__name__ not in FACTORIES
        nbytes = [
            t.numel() * t.element_size()
            for t in cuts._tensors((args, kwargs))
            if reads and t.device.type != "meta"
        ]
        made = sum(
            t.numel() * t.element_size() for t in cuts._tensors(out) if t.device.type != "meta"
        )
        charge = made + max(nbytes, default=0) if made else 0
        if charge > self.limit:
            self.refused += 1
            raise torch.OutOfMemoryError(f"CUDA out of memory. Tried to allocate {charge} B.")
        self.largest = max(self.largest, charge)
        return out


class Room:
    """The card's memory: rungs 1-2 free `frees` bytes once, rung 4 `lowers` steps of `step`."""

    def __init__(self, card: Card, frees: int = 0, lowers: int = 0, step: int = 0) -> None:
        self.card, self.frees, self.lowers, self.step = card, frees, lowers, step
        self.made = self.lowered = 0

    def make(self, device: torch.device, need: int) -> bool:
        self.made += 1
        self.card.limit += self.frees
        freed, self.frees = self.frees, 0
        return freed > 0

    def lower(self, device: torch.device, need: int) -> bool:
        if not self.lowers:
            return False
        self.lowered += 1
        self.lowers -= 1
        self.card.limit += self.step
        return True

    def available(self, device: torch.device) -> int:
        return self.card.limit

    def exhausted(self, device: torch.device, need: int, op: str) -> Exception:
        return MemoryError(f"exhausted: {op} needs {need} B")


def fresh(
    limit: int, only: frozenset[str] = frozenset(), **room: int
) -> tuple[cuts.Memo, Card, Room]:
    memo, card = cuts.Memo(), Card(limit, only)
    served = Room(card, **room)
    cuts.serve(memo, served)
    return memo, card, served


def site(name: str) -> cuts.Site:
    return cuts.Site((name, 0), CPU, cuts.Track(name))


def measured(call, grad=torch.no_grad):  # type: ignore[no-untyped-def]
    """The call's result and the largest charge any op of it makes (on an unbounded card)."""
    card = Card(1 << 60)
    with card, grad():
        out = call()
    return out, card.largest


# ------------------------------------------------------------------------------ FFN


def ffn_cut() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 300, 64)
    for activation in ("swiglu", "geglu", "gelu"):
        ff = FeedForward(64, mult=4, activation_fn=activation).eval()
        whole, largest = measured(functools.partial(ff, x))
        count = cuts.install(ff)
        memo, card, room = fresh(largest // 3)
        with card, torch.no_grad():
            cut = ff(x)
            retries = memo.retries
            again = ff(x)
        sizes = [s for k, s in memo.sizes.items() if k[0] == "ffn"]
        check(f"ffn {activation}: FeedForward and its projection wrapped", count == 2, count)
        check(
            f"ffn {activation}: token pieces equal whole",
            same(cut, whole),
            (cut - whole).abs().max(),
        )
        check(
            f"ffn {activation}: the piece size is remembered",
            sizes and min(sizes) < 600,
            memo.sizes,
        )
        check(
            f"ffn {activation}: warm run cuts up front, no retry",
            memo.retries == retries and same(again, whole),
            memo.retries - retries,
        )


class FusedSwiGLU(SwiGLU):
    """A package's fused SwiGLU (H3's glue): its own forward, diffusers' class underneath."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.proj(hidden_states).chunk(2, dim=-1)
        return value * nn.functional.silu(gate)


class FusedBlock(nn.Module):
    """H3's fused block calls `ff.net[2](ff.net[0](x))`, around FeedForward.forward."""

    def __init__(self) -> None:
        super().__init__()
        self.ff = FeedForward(64, mult=4, activation_fn="swiglu")
        self.ff.net[0].__class__ = FusedSwiGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff.net[2](self.ff.net[0](x))


class Carrier:
    """A part's result that is no tensor (an fp8 operand with its scales)."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value


class CarrierSwiGLU(SwiGLU):
    def forward(self, hidden_states: torch.Tensor) -> Carrier:  # type: ignore[override]
        return Carrier(SwiGLU.forward(self, hidden_states))


def fused_ffn() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 500, 64)
    block = FusedBlock().eval()
    whole, largest = measured(lambda: block(x))
    cuts.install(block)
    memo, card, room = fresh(
        largest * 3 // 5
    )  # net[2] on the whole [500, 256] fits, the projection not
    with card, torch.no_grad():
        cut = block(x)
    check(
        "fused SwiGLU called by its block cuts at the projection",
        same(cut, whole) and card.refused,
        (cut - whole).abs().max(),
    )
    check(
        "fused SwiGLU's cut is remembered by its kind",
        any(k[:2] == ("ffn", "SwiGLU") for k in memo.sizes),
        memo.sizes,
    )

    act = CarrierSwiGLU(64, 512).eval()
    plain, largest = measured(lambda: act(x).value)
    cuts.install(act)
    memo, card, room = fresh(largest // 3)
    with torch.no_grad():
        try:
            with card:
                act(x)
            refused = False
        except torch.OutOfMemoryError:
            refused = True
        check(
            "a part whose result does not join runs whole (and the whole may refuse)",
            refused and any(k[1] == "SwiGLU" for k in memo.whole),
            memo.whole,
        )
        check("after which it runs whole without trying pieces", same(act(x).value, plain))


# ------------------------------------------------------------------------ attention


def sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """A [batch, sequence, heads, dim] kernel, GQA by repeating K/V heads, its scores whole."""
    group = q.shape[2] // k.shape[2]
    q, k, v = (
        t.transpose(1, 2)
        for t in (q, k.repeat_interleave(group, dim=2), v.repeat_interleave(group, dim=2))
    )
    scores = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores + mask
    return (scores.softmax(-1) @ v).transpose(1, 2)


def attention_cut() -> None:
    torch.manual_seed(2)
    q, k, v = torch.randn(1, 96, 8, 4), torch.randn(1, 400, 2, 4), torch.randn(1, 400, 2, 4)
    mask = torch.randn(1, 8, 96, 400)
    whole, _ = measured(lambda: sdpa(q, k, v, mask))
    row = 2 * 400 * 8  # a query row of one head: its scores and their working copy
    for limit, what in (
        (4 * 96 * row, "head groups"),
        (96 * row, "one head"),
        (16 * row, "query rows"),
    ):
        memo, card, room = fresh(limit)
        with card, torch.no_grad():
            cut = cuts.attend(sdpa, q, k, v, mask, rows=16)
            retries = memo.retries
            again = cuts.attend(sdpa, q, k, v, mask, rows=16)
        size = next(s for key, s in memo.sizes.items() if key[0] == "attention")
        check(f"attention by {what} equals whole", same(cut, whole), (cut - whole).abs().max())
        check(
            f"attention by {what}: warm run cuts up front",
            memo.retries == retries and same(again, whole),
            (memo.retries, retries),
        )
        if what == "query rows":
            check(
                "query rows start on the kernel's block multiple",
                size % 16 == 0 and size < 96,
                size,
            )
    memo, card, room = fresh(16 * row)
    try:
        with card, torch.no_grad():
            cuts.attend(sdpa, q, k, v, mask, rows=0, heads=False)
        check("attention that may not cut runs out of memory", False)
    except torch.OutOfMemoryError:
        check(
            "attention that may not cut runs out of memory, once retried",
            memo.retries == 1,
            memo.retries,
        )


# ---------------------------------------------------------------------------- bands

BANDED = frozenset(
    {
        "convolution",
        "conv2d",
        "conv3d",
        "native_group_norm",
        "group_norm",
        "upsample_nearest2d",
        "constant_pad_nd",
        "pad",
    }
)


def banded(limit: int, only: frozenset[str], seed: int, call):  # type: ignore[no-untyped-def]
    memo, card, room = fresh(limit, only)
    with card:
        cuts.enter()
        try:
            torch.manual_seed(seed)
            out = call()
            retries = memo.retries
            torch.manual_seed(seed)
            again = call()
        finally:
            cuts.leave()
    return out, again, memo, retries


def bands() -> None:
    torch.manual_seed(3)
    x2, x3 = torch.randn(1, 8, 64, 40), torch.randn(1, 4, 6, 32, 24)
    f = nn.functional
    ops = {
        "conv 3x3 pad 1": lambda: f.conv2d(x2, torch.randn(8, 8, 3, 3), torch.randn(8), padding=1),
        "conv stride 2 dilation 2": lambda: f.conv2d(
            x2, torch.randn(6, 8, 3, 3), stride=2, padding=3, dilation=2
        ),
        "conv grouped 5x1": lambda: f.conv2d(x2, torch.randn(8, 2, 5, 1), padding=(2, 0), groups=4),
        "conv3d": lambda: f.conv3d(x3, torch.randn(4, 4, 3, 3, 3), padding=1),
        "pad constant": lambda: f.pad(x2, (1, 2, 3, 5), value=0.5),
        "upsample nearest": lambda: f.interpolate(x2, scale_factor=2, mode="nearest"),
        "upsample nearest 3d": lambda: f.interpolate(x3, scale_factor=(1, 2, 2), mode="nearest"),
        "upsample nearest-exact": lambda: f.interpolate(x2, scale_factor=3, mode="nearest-exact"),
        "group norm": lambda: f.group_norm(x2, 4, torch.randn(8), torch.randn(8), eps=1e-6),
        "group norm 3d": lambda: f.group_norm(x3, 2),
    }
    for name, make in ops.items():
        for grad in (torch.no_grad, torch.inference_mode):  # aten's decomposed or composite ops
            torch.manual_seed(4)
            whole, largest = measured(make, grad)
            with grad():
                out, again, memo, retries = banded(largest - 1, frozenset(), 4, make)
            check(
                f"{name} ({grad.__name__}) in bands equals whole",
                same(out, whole),
                (out - whole).abs().max(),
            )
            check(
                f"{name} ({grad.__name__}): warm run in bands up front",
                memo.retries == retries > 0 and same(again, whole),
                (memo.retries, retries),
            )


def vae_decode() -> None:
    torch.manual_seed(5)
    vae = AutoencoderKL(
        block_out_channels=(16, 32, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
    ).eval()
    z = torch.randn(1, 4, 24, 20)
    card = Card(1 << 60, BANDED)
    with card, torch.no_grad():
        whole = vae.decode(z).sample
    for share in (3, 16):
        with torch.no_grad():
            out, again, memo, retries = banded(
                card.largest // share, BANDED, 0, lambda: vae.decode(z).sample
            )
        kinds = sorted(
            {
                k[0].overloadpacket.__name__
                for k in memo.sizes
                if isinstance(k[0], torch._ops.OpOverload)
            }
        )
        check(
            f"VAE decode in 1/{share} of its largest op equals whole",
            same(out, whole, 1e-9),
            (out - whole).abs().max(),
        )
        check(
            f"VAE decode in 1/{share}: warm decode has no retries",
            memo.retries == retries > 0,
            (memo.retries, retries),
        )
        check(
            f"VAE decode in 1/{share}: cuts {kinds}",
            {"convolution", "native_group_norm", "upsample_nearest2d"} <= set(kinds),
            kinds,
        )


# ----------------------------------------------------------------------- the catch


class Noisy(nn.Module):
    """A block that draws noise and counts its calls (declared state), writing nothing else."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(32, 256)
        self.calls = 0
        self.seen: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.seen.append(self.calls)
        noise = torch.randn(x.shape[0], 256)
        return (self.lin(x) + noise).relu().sum(-1, keepdim=True) * torch.ones(1, 32)


class Writer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(2)


def block_catch() -> None:
    torch.manual_seed(6)
    block = Noisy().eval()
    x = torch.randn(64, 32)
    torch.manual_seed(7)
    whole, _ = measured(lambda: block(x))
    there = cuts.Site(("noisy", 0), CPU, cuts.Track("noisy"), block, ("calls", "seen"))
    memo, card, room = fresh(1 << 60)
    with torch.no_grad():
        torch.manual_seed(7)
        first = cuts.run(there, block.forward, (x,), {})
    facts = memo.blocks[("noisy", 0)]
    check(
        "a block that writes nothing is bare",
        facts.bare and facts.generators == (torch.default_generator,),
        facts,
    )
    memo, card, room = fresh(64 * 256 * 8 + 1000, frees=1 << 40)  # the noise fits, the linear not
    memo.blocks[("noisy", 0)] = facts
    with card, torch.no_grad():
        cuts.enter()
        try:
            torch.manual_seed(7)
            block.calls, block.seen = 0, []
            out = cuts.run(there, block.forward, (x,), {})
        finally:
            cuts.leave()
    check(
        "a bare block out of memory runs again under the layer, exactly",
        same(out, whole) and same(first, whole),
        (out - whole).abs().max(),
    )
    check(
        "its generator and declared state were put back",
        block.calls == 1 and block.seen == [1],
        (block.calls, block.seen),
    )
    check(
        "one retry, room made once",
        memo.retries == 1 and room.made == 1 and card.refused == 1,
        (memo.retries, room.made, card.refused),
    )

    memo, card, room = fresh(1 << 60)
    y = torch.ones(4)
    cuts.run(site("writer"), Writer().forward, (y,), {})
    check(
        "a block that writes into its input is no bare block",
        not memo.blocks[("writer", 0)].bare and float(y[0]) == 2,
    )

    torch.manual_seed(8)
    fused = FusedBlock().eval()
    x = torch.randn(1, 500, 64)
    whole, largest = measured(lambda: fused(x))
    cuts.install(fused)
    memo, card, room = fresh(1 << 60, lowers=8, step=1 << 40)
    there = site("fused")
    with torch.no_grad():
        cuts.run(there, fused.forward, (x,), {})
        memo.blocks[("fused", 0)] = cuts.Facts(False, ())  # as a writer: under the layer
        card.limit = largest * 3 // 5
        with card:
            out = cuts.run(there, fused.forward, (x,), {})
    check(
        "an op out of memory inside an FFN asks it to cut finer, never rung 4",
        same(out, whole) and card.refused and room.lowered == 0,
        (card.refused, room.lowered, (out - whole).abs().max()),
    )


def lattice() -> None:
    memo, card, room = fresh(100, frees=100, lowers=3, step=100)
    with card:
        cuts.enter()
        try:
            torch.ones(4096)
            ended = None
        except MemoryError as exc:
            ended = exc
        finally:
            cuts.leave()
    check(
        "past rungs 1-2 and every rung 4 step, exhausted",
        ended is not None and room.lowered == 3 and memo.retries == 4,
        (ended, room.lowered, memo.retries),
    )

    torch.manual_seed(9)
    x = torch.randn(64, 40)
    whole = (x * 2).sum(-1)

    def doubled(most: int) -> torch.Tensor:
        return torch.cat([(p * 2).sum(-1) for p in x.split(max(most // 40, 1))])

    memo, card, room = fresh(4000)
    with card:
        rows = cuts.split("double", "double:40", CPU, x.numel(), 40, doubled)
    size = memo.sizes[("split", "double:40")]
    check(
        "a split loop runs in windows that fit, joined exactly",
        same(rows, whole) and size * 16 <= 4000,
        size,
    )


def main() -> int:
    for test in (ffn_cut, fused_ffn, attention_cut, bands, vae_decode, block_catch, lattice):
        test()
    print(f"{sum(results)}/{len(results)} passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
