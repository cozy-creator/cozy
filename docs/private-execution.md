# Private execution admission

Decision: proto-051 runs the existing Creator coordinator on the private worker
from initial submission. The laptop subsequently observes and controls that owner.
An already claimed legacy worker does not silently change ownership.

`cozy.creator.private-execution/1` is Creator's private admission document. Creator
authors and validates its canonical JSON and SHA-256 identity. PodHost binds that
identity in its execution-owner grant, retains the accepted bytes, and passes them
to the fixed image-owned Creator executable. Tensorhub never receives the capsule.

The capsule contains the root job and typed input, exact existing
`LocalPackageRevision` documents, their corresponding `PackageInterface`
documents, and captured callable bindings. It contains no program ordering or
replay instructions: ordinary Python decides which functions to call. Runtime's
existing operation memoization remains independent of this root admission.

Wheel bytes use the existing signed `LocalPackageUpload` path before execution
ownership is accepted. Upload does not require a laptop WorkerControl claim.
Metadata validation is separate from checking verified file custody and recording
the root. Only the latter produces an execution-admitted receipt. A Host ownership
receipt alone cannot be presented as a successfully submitted root job.

The fixed bootstrap receives capsule and authority through Host-owned file
descriptors. Host resolves the coordinator home, package staging root, TLS identity,
and TensorFS root. Captured Python cannot select those paths or launch commands.
The pod generates and retains its own execution key; the laptop key is never sent.

Initial implementation is bounded to an unclaimed worker and captured computation.
Renewable external source/publication grants, observer replay, cancel/revoke,
restart recovery, and full disconnected A-to-B-to-C qualification remain required
before claiming complete disconnected execution. The metadata decoder alone does
not provide a runnable offline mode.
