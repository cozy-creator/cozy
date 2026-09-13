# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.16.10", "offline-steps==1.0.0"]
# [tool.uv.sources]
# offline-steps = {path = "./library", editable = true}
# ///
from offline_steps import finish, prepare


async def main(ctx) -> int:
    print("edited caller after coordinator restart")
    prepared = await prepare(value=7)
    finished = await finish(value=prepared.value + 10)
    assert finished.value == 60
    return finished.value
