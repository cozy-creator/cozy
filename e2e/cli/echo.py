# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The cli suite's weightless request: from cozy through TensorD, the Runtime and TensorFS to the executor's
reply and back, with no model, so it runs on any computer."""


def main(*, text: str) -> str:
    return text[::-1]
