# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==0.16.10", "offline-steps==1.0.0"]
# [tool.uv.sources]
# offline-steps = {path = "./library", editable = true}
# ///
from offline_steps import after_disconnect, finish, prepare


async def main(ctx) -> int:
    prepared = await prepare(value=7)
    continued = await after_disconnect(value=prepared.value)
    finished = await finish(value=continued.value)
    assert finished.value == 51
    return finished.value
