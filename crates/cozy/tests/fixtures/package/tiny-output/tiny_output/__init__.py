"""A torch-free package for cozy's run tests: one function that writes a small text file."""

import time

import msgspec
from cozy_runtime.author import App, FileAsset, Outputs

app = App()


class WriteInput(msgspec.Struct, forbid_unknown_fields=True):
    text: str = "hello"
    seconds: float = 0.0


class WriteOutput(msgspec.Struct):
    file: FileAsset


@app.entrypoint(accelerator=False)
def write(payload: WriteInput, out: Outputs) -> WriteOutput:
    """Waits `seconds`, then saves `text` as a file."""
    time.sleep(payload.seconds)
    return WriteOutput(out.save_bytes(payload.text.encode(), media_type="text/plain"))
