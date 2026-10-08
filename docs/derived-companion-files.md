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

A companion may instead name a verified Store object: the native `begin_derived` binding
takes `files={path: (object_id, length)}` beside base64 values. Nothing is copied into the
intent and no byte is new; the object becomes the same manifest entry and header asset,
and a commit refuses when the Store does not hold it. Uploads attach a pipeline's pinned
tokenizers this way, since they exceed the inline bound.

TensorFS inherits both ordinary manifest companions and header assets by reference
from source checkpoints. An asset present only in the header keeps its logical digest,
length, media type and ordered segment references; it does not need a whole-file CAS
object or a manifest entry. Retaining large assets does not consume the inline-file
limit or write budget. Header/tensor backing objects listed redundantly as ordinary
source files are excluded unless that same path is a declared header asset with
matching content. An explicit asset role survives byte deduplication with tensor data.

Equal multi-source companion entries or complete asset records deduplicate. Differing
asset contents, metadata or segmentation at one name require an explicitly declared
replacement. An inherited asset and manifest companion at the same path must agree on
logical digest and length; the header's media type and segments remain authoritative.
Declaring `files` or a verified object at that path replaces both inherited forms,
using the ordinary single-object, `application/octet-stream` asset representation
(an empty ordinary file has no header asset). All inherited names share the normal
companion path rules. Tensor parts retain the same references; no global cache path
or network lookup enters the derivation.

Crash recovery retains the small frozen file bytes in the work/checkpoint record.
A retry can recreate these objects at commit without repeating tensor work. The
normal native receipt, independent adoption, retention, GC and publication closure
cover both the tensors and the companion files.
