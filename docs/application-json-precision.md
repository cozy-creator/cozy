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
