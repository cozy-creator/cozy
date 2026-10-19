# TensorFS

TensorFS is a Rust library and `tfs` command-line tool for opening, reading, writing and
sharing cozytensors checkpoints. The Python package is built from the same implementation.

## Developer documentation

- [Rust API and examples](API.md): create a client, call it, and use the returned objects.
- [Python API and examples](../tensorfs-py/API.md): import the library, read tensors, write and publish.

The guides describe the planned 0.13 interface while its implementation is being integrated.
For reference documentation generated from the checked-in Rust code, run
`cargo doc -p tensorfs --no-deps --open`.

## Technical format documents

- [cozytensors specification](spec/cozytensors.md)
- [Distribution protocol](spec/distribution.md)
- [Readable format examples](spec/examples/)

The Rust implementation is in this crate. The Python binding code, Python package and
wheel configuration live together in [tensorfs-py](../tensorfs-py/).
