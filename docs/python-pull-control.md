# Controlling a Python model pull

`PullCancellation` cooperatively stops one native pull. Pass the token with
`pull(..., cancellation=token)` and call `token.cancel()` when its last consumer
no longer needs it. `token.cancelled` records that intent. A resumed pull uses a
new token; already verified objects and resumable partials remain in the Store.

An optional `progress(available_bytes, total_bytes)` callback observes the native
plan and completed object admissions. Available bytes include objects already
held and verified cache reads, not retransmitted network bytes. Transfer runs
without the Python GIL; callbacks acquire it only while calling Python.
