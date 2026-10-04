# Application JSON precision

Confirmed source audit: the Go machine application path normalizes root payloads, captured
input replays and several native values with PackageInterface JCS. That protocol profile
bounds every number to ±(2^53−1) and merges integer and float spelling. Valid typed uint64
seeds/results therefore fail before execution or lose their intended type.

Planned change: add a separate precision-preserving application entry point to the existing
bounded JSON parser. It retains integer digits and the integer/float distinction, sorts
object keys, ignores formatting, preserves arrays/input order, rejects duplicate keys and
invalid JSON, and shares the current document/depth bounds. The JCS entry point and wire
profile retain their current behavior. Change only authored application values after tracing
consumers; protocol profiles, model artifacts and string-only captured module identities
stay on their existing rules.

Validation: red/green typed ordinary CLI and API job/result checks on the coherent Rust
machine with isolated CPU homes and exact executor-plane SDK wheels, plus targeted native
result/input-snapshot identity tests. Preserve different large seeds and integer versus
float as true intent, while formatting/key order reattaches. Do not substitute smaller
seeds. Root/rental inference qualification remains independent.

Confirmed consumer proof: the unchanged typed parent/internal-job child runs through ordinary
CLI and authenticated API on Rust candidate72c5c88, Runtime source0012896a and executor-plane
TensorFS source3fc31b3. Four CLI requests retain zero, uint64 maximum and beyond-53-bit seeds,
integer/float observations and ordered values; direct API result keeps uint64 maximum,
formatting/key order/1e0 replay the same request, and seed/type/order changes conflict. The
product check passed32.104s. This is CPU application/transport proof, not GPU qualification.

The full path exposed two additional SDK boundaries (managed arguments/results and typed
result spool), a captured child incorrectly classified as serving, and CLI JSON rendering
through TOON's float64 reader. Each was corrected and the same authored request rerun. JSON
now serializes the original typed document directly; TOON's default rendering and protocol
JCS remain unchanged. Application-only changes do not redefine wire metadata profiles.

The ordinary default CLI rendering was also checked with uint64 maximum and float 1.0.
TOON values that cannot round-trip application numeric semantics now use the existing
JSON fallback. Representable TOON output keeps its format, and human scalar results retain
precise JSON numbers. The extended ordinary CLI/API consumer check passed47.628s.
