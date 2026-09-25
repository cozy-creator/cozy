# Derived checkpoint companion files

A native derivation can attach small, immutable ordinary files:

```python
Derivation(
    sources={"source": capability},
    targets=targets,
    configs=configs,
    order=order,
    files={"LICENSE": license_bytes, "Notice": attribution_bytes},
)
```

The optional transport argument is omitted when `files` is empty, preserving existing
ordinary derivation calls. Nonempty files require TensorFS 0.3.51 and Runtime 0.18.21.

These bytes are part of the locked work declaration. Changing them changes the
work identity and cannot adopt progress recorded for another declaration. TensorFS
writes ordinary CAS blobs at commit and returns actual named `Entry::File` entries
beside `model.cozytensors`. The same objects are model assets in the header, so normal
model-closure downloads and retention include required companion documents. No
configuration value is treated as a file and no filesystem path is supplied by the
caller.

The first API permits at most 16 new files, 64 KiB per file, and 512 KiB altogether.
Base64 transport therefore leaves room under Runtime's 1 MiB declaration limit;
the full serialized declaration still has that existing limit. New file bytes count
against the output grant and the native write estimate. Paths use the normal
manifest grammar and cannot collide with the checkpoint header or traverse another
file. Streaming larger assets is outside this bounded metadata API.

TensorFS inherits existing ordinary companions by exact reference from source
checkpoints. Header/tensor backing objects listed redundantly as ordinary source
files are excluded unless that same path is a declared header asset with matching content.
An explicit asset role survives byte deduplication with tensor data. Equal multi-source names and contents deduplicate; differing
contents at one name require an explicitly declared replacement. Tensor parts retain
the same references. The source manifest remains the authority for the inherited
file set; no global cache path or network lookup enters the derivation.

Crash recovery retains the small frozen file bytes in the work/checkpoint record.
A retry can recreate these objects at commit without repeating tensor work. The
normal native receipt, independent adoption, retention, GC and publication closure
cover both the tensors and the companion files.
