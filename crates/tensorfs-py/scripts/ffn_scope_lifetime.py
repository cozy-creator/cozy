"""Real CPU FFN cuts keep their shrink policy without retaining prior activations.

Load the exact Python source without importing/building the native TensorFS extension.
An optional source argument permits the same regression to demonstrate the old failure.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import weakref
from pathlib import Path

import torch


def main() -> None:
    source = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1] / "python/tensorfs/varena/cuts.py"
    )
    spec = importlib.util.spec_from_file_location("ffn_scope_lifetime_cuts", source)
    assert spec is not None and spec.loader is not None
    cuts = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cuts
    spec.loader.exec_module(cuts)

    class Room:
        available_bytes = 0

        def available(self, device: torch.device) -> int:
            assert device.type == "cpu"
            return self.available_bytes

    memo, room = cuts.Memo(), Room()
    cuts.serve(memo, room)
    shape = memo.shape("ffn", "eight-tokens")
    track = cuts.Track("ffn", shape)
    site = cuts.Site(("ffn", 1), torch.device("cpu"), track)
    previous = cuts._local.site
    sizes: list[int] = []

    def forward(value: torch.Tensor) -> torch.Tensor:
        sizes.append(value.numel() // value.shape[-1])
        return value.square()

    call = cuts._ffn("FeedForward", 16, forward)
    try:
        cuts._local.site = site
        value = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)
        pointer = weakref.ref(value)
        expected = value.square()
        actual = call(value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert sizes == [8] and len(shape.scopes) == 1
        key, (_, _, shrink) = next(iter(shape.scopes.items()))
        del value
        gc.collect()
        assert pointer() is None, "cached FFN shrink scope retained its input activation"
        cells = dict(zip(shrink.__code__.co_freevars, shrink.__closure__ or (), strict=True))
        assert all(not isinstance(cell.cell_contents, torch.Tensor) for cell in cells.values())

        # The retained policy still uses current room and cuts actual CPU work
        # after its original activation is gone: 8 tokens -> 3,3,2 -> singletons.
        room.available_bytes = 3 * 16 * 4
        shape.peak = 999
        assert cuts.finer(shape) and memo.sizes[key] == 3 and shape.peak == -1
        sizes.clear()
        value = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)
        pointer = weakref.ref(value)
        actual = call(value)
        assert sizes == [3, 3, 2]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del value
        gc.collect()
        assert pointer() is None, "a later FFN pass retained its input activation"
        assert cuts.finer(shape) and memo.sizes[key] == 1
        sizes.clear()
        actual = call(torch.arange(64, dtype=torch.float32).reshape(2, 4, 8))
        assert sizes == [1] * 8 and not cuts.finer(shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print("PASS CPU FFN scope frees prior activations; finer cuts remain exact; "
              f"closure={','.join(cells)}")
    finally:
        cuts._local.site = previous


if __name__ == "__main__":
    main()
