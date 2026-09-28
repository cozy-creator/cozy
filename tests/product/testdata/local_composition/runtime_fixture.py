"""Privileged local test launcher: real Worker, native provider and durable fault boundary."""
import json
import sys
from pathlib import Path

from cozy_runtime.internal.worker import session

config = json.loads(Path(__file__).with_name("source_fixture.json").read_bytes())


class FixtureWorker(session.Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.source_calls is not None:
            self.source_calls.endpoints = {"huggingface": config["provider"]}
            self.source_calls.native_registry = Path(config["registry"]).read_bytes()


session.Worker = FixtureWorker
from cozy_runtime.cli.main import main
raise SystemExit(main(sys.argv[1:]))
