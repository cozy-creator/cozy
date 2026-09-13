# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==0.14.1", "tensorfs==0.3.36", "publication-source-tools==0.0.1"]
# [tool.uv.sources]
# publication-source-tools = {path = "./source_tools", editable = true}
# ///
"""Tiny native publication plumbing proof; no trained-model quality claim."""
from cozy_runtime.author.publication import publish_release, upload_checkpoint
from publication_source import produce

PRIVATE_SCRIPT_CANARY = "cl171-private-script-69cdc7a5"
DESTINATION = "proofowner/assessment"
RELEASE = "fixture"
GATE = True
EXPECTED_REVISION = None


async def main(ctx):
    artifact = await produce(value=1)
    ctx.log(f"Native producer result: {artifact.manifest.digest}")
    if not GATE:
        raise ValueError("planned quality gate refusal")
    checkpoint = await upload_checkpoint(artifact, destination=DESTINATION)
    release = await publish_release(destination=DESTINATION, release=RELEASE,
                                    lanes={"bf16": checkpoint},
                                    expected_revision=EXPECTED_REVISION)
    return {"checkpoint": checkpoint.checkpoint,
            "upload": checkpoint.observation,
            "release": release.release,
            "revision": release.revision,
            "observation": release.observation}
