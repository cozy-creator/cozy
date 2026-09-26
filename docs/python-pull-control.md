# Controlling a Python model pull

Available from TensorFS 0.3.55.

`PullCancellation` cooperatively stops one native pull. Pass the token with
`pull(..., cancellation=token)` and call `token.cancel()` when its last consumer
no longer needs it. `token.cancelled` records that intent. A resumed pull uses a
new token; already verified objects remain in the Store. Existing model-pull recovery is
per object: an unfinished object is fetched again, without re-fetching admitted
objects. Cancellation introduces no timeout or extra cleanup.

An optional `progress(available_bytes, total_bytes)` callback observes the native
plan and completed object admissions. Available bytes include objects already
held and verified cache reads, not retransmitted network bytes. Transfer runs
without the Python GIL; callbacks acquire it only while calling Python.
