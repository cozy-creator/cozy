"""A store and endpoint the caller does not name come from the native rule (TENSORFS_HOME,
TENSORFS_ENDPOINT, their defaults), as on every TensorFS surface; no network."""

import os
import sys
from pathlib import Path

import tensorfs

root = Path(sys.argv[1])
os.environ["TENSORFS_HOME"] = ""
try:
    tensorfs.open("acme/none", local_files_only=True)
except tensorfs.InvalidError as error:
    assert "TENSORFS_HOME is empty" in str(error), error
else:
    raise AssertionError("an empty TENSORFS_HOME named a store")

# An empty endpoint is the default one, so an offline open is merely absent.
os.environ["TENSORFS_HOME"] = str(root / "home")
os.environ["TENSORFS_ENDPOINT"] = ""
try:
    tensorfs.open("acme/none", local_files_only=True)
except tensorfs.NotFoundError:
    pass
else:
    raise AssertionError("acme/none opened")
assert (root / "home" / "format").is_file(), "TENSORFS_HOME was not the store"
print("store and endpoint by the native rule")
