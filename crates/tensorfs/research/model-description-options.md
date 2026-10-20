# Model descriptions in cozytensors: design exploration

Status: proposal, 2026-10-19. This does not change the stored format, current
Python/Rust interfaces, or the TensorFS build being qualified.

## Recommendation

Prototype a small, framework-independent, versioned model contract for one
architecture that has two real consumers. Keep one canonical parameter layout
per version and use the existing import/read machinery. Do not add a per-file
tensor transformation language or a universal graph executor to TensorFS.

The prototype must prove more than preserving a class name: an independent
consumer must recover the parameters' mathematical roles, configuration and
axis conventions without importing the producer's Python model class. Only
then should the contract become part of the normative format.

## Three alternatives

| Design | Produced from | Consumed by | Main tradeoff |
|---|---|---|---|
| A. Specified architecture and canonical weights | A tested importer or framework exporter for a known architecture | Independent implementations of that architecture | Small artifact/interface; real specification and adapter work per family |
| B. Standard portable computation graph | A supported graph exporter operating on a trusted model | An existing graph engine through a separate adapter | Supports more novel computations; exporter/operator coverage and performance become requirements |
| C. Validated framework-native snapshot | A live model plus its exact installed definition and configuration | An explicitly supplied matching local factory | Easy for current Python users; still depends on framework code and does not meet independent-semantics goal |

These are different guarantees. A definition hash can make C reproducible but
cannot turn it into A. A can describe a fixed mathematical family without
carrying every operation as a graph. B describes the operations themselves.

## A: architecture contract

An illustrative header extension would identify a model schema and carry its
fully resolved semantic configuration. The names are provisional:

```text
model.schema = "cozy.model.<architecture>/1"
model.config = {all variable choices affecting the trained computation}
```

The versioned specification defines the meaning of canonical parameter keys,
their logical axes, repeated layers, Q/K/V order, normalization, positional
encoding, and component connections. The artifact supplies dimensions and
other variable choices. Python class names can remain export/loading hints;
they do not define the contract.

This is self-description relative to a published specification, just as an
encoding name plus parameters relies on the encoding's specified mathematics.
A new identifier without such a definition would accomplish little.

**Production:** a supported importer reads the source's explicit conventions,
normalizes names and layouts once, resolves configuration defaults, validates
required roles and writes ordinary TensorFS tensors. A model author can use a
framework exporter that knows the same contract. Unknown custom models remain
readable tensor collections until a contract/exporter is supplied; we must not
claim they have independently specified model semantics.

**Consumption:** `open` still reads the checkpoint. A consumer first verifies
the declared schema/version and configuration, constructs its own supported
implementation, and loads canonical roles using existing TensorFS reads. It
may select different kernels, fuse operations or split work across devices.
No model execution, second I/O engine or automatic Python import belongs in
TensorFS itself.

Physical quantization remains a separate contract. Some source layout changes
can rearrange encoded arrays; others require scheme-aware conversion or
re-encoding. A supported model schema must not imply that every transformation
is zero-copy or lossless.

## B: portable graph

An optional graph could use an existing standard such as ONNX, with TensorFS
supplying its weights. It would describe mathematical operators and their
connections, rather than device-specific kernel choices. An external engine
would compile or execute it. [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html).

Production can use an existing exporter, but a trace is not automatically a
complete portable model: dynamic shapes, branches, caches, custom operators,
randomness and unsupported Python behavior need explicit treatment. Exporting
a wrongly configured source faithfully preserves its mistake.
[PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx_export.html).

ONNX external-data locations are filesystem paths, not TensorFS object refs.
Any integration needs a precise parameter-binding interface rather than placing
invented content-addressed URLs in that field. A graph engine may also require
different packed weights or decoded floating-point tensors; current Runtime
speed and memory use do not carry over automatically.
[External data](https://onnx.ai/onnx/repo-docs/ExternalData.html).

This is worth a separate bounded experiment if executing unfamiliar model
architectures without their original Python code is a product requirement.
It should not be a prerequisite for finishing ordinary TensorFS storage APIs.

## C: validated native snapshot

An optional framework adapter could save a live model's configuration, exact
definition identity, parameter/buffer roles and explicit tied parameters. A
local factory would reconstruct it and require an exact state binding, rather
than accepting missing keys or silently selecting a similar implementation.

This improves reproducibility and catches missing weights, incompatible
definitions, shapes and lost ties. It cannot determine from equal-shaped
matrices alone that a producer mislabeled scale rows as shift rows. It is a
useful adapter feature, not the stronger model-semantics format guarantee.

## Concrete errors and realistic benefits

| Upstream problem | What a model contract changes | What remains necessary |
|---|---|---|
| FLUX LoRA conversion permuted both factors instead of only the output factor | Specifies which axes and parameter roles the adapter modifies | Correct importer and numerical conversion tests |
| H3 mixed-rank LoRA used key order to infer one global alpha | Requires an explicit effective scale for every adapter target | Source importer must establish the intended coefficient |
| MiniMax-M2 legacy config caused full-head instead of partial rotary encoding | Requires explicit rotary dimension, pairing and frequency semantics | Each execution backend must implement the declared contract correctly |
| A base-model prefix mismatch left weights randomly initialized | Requires complete canonical-role binding; missing required weights refuse construction | No implicit partial-load fallback in the consuming adapter |

Sources and exact dates are in [recent checkpoint mismatches](recent-checkpoint-mismatches.md).
These benefits prevent consumer guesswork and make violations testable. They
do not prove that an exporter correctly labeled every value, nor recover data
lost before the checkpoint was written.

## First proof and adoption gate

Use one real model family and two independent consumers. Show that two producer
layouts normalize to the same semantic tensors; both consumers produce matching
outputs within an explicit numerical tolerance. Include same-shaped row swaps,
Q/K/V ordering, a changed semantic config field, missing roles and a scaled
adapter. Preserve separate storage, quantization and GPU-performance tests.

Keep the existing TensorSchema view limited to key/shape/logical dtype; do not
rename it into a claim about computation. Do not resurrect compatibility
services, model fingerprints or proof caches as substitutes for semantics.

Before format adoption, decide how readers must acknowledge the model contract.
Current minor-version rules allow unknown trailing fields to be ignored. A new
field alone therefore cannot guarantee that older model constructors honor it.
Raw tensor access and semantic model construction need explicitly different
capability claims, or a deliberate incompatible format revision.
