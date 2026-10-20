"""A metadata-only import must leave a fresh CPU parent single-threaded."""

from pathlib import Path


def threads():
    return {path.name: path.joinpath("comm").read_text().strip()
            for path in Path("/proc/self/task").iterdir()}


before = threads()
import tensorfs

after = threads()
assert after == before, (before, after)
print(f"TensorFS {tensorfs.__version__} import: no native worker started; CPU parent thread inventory unchanged")
