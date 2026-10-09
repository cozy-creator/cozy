# Owned Comfy service client

`cozy dev comfy` is an explicit development client for a ComfyUI server already running on an owned development rental. It runs outside the Python package executor; package networking fences remain unchanged. It never rents a machine or starts/stops ComfyUI.

```sh
cozy dev comfy --rental=NAME --input=comfy-request.json \
  --idempotency-key=benchmark-warmup-1 \
  --ssh-known-hosts=./verified-hosts --ssh-key=~/.ssh/rental \
  --out=./results --json
```

The input contains `graph_json` (the Comfy API graph encoded as a string) and the absolute remote `output_root`. Optional fields are `port` (8188), `timeout_s` (3600, at most7200), `trace` (false), `trace_root`, and `expected_steps` (8 or30). Tracing expects the installed H3 CUDA-trace hook and validates every expected completed step. `--python` selects remote Python3.11+; WebSocket events are captured when `websocket-client` is available, and HTTP history polling works without it.

The command verifies the attached rental and boot, uses the supplied strict SSH host-key file, and retains a local operation receipt under Cozy's `dev/comfy` directory. Remote receipts and the captured observer are held under `--remote-state` (default `/root/.cozy/dev/comfy`). The CLI acknowledges a durable remote preparation receipt before starting the observer; losing that acknowledged receipt refuses another submission. A detached remote observer writes its prompt UUID and submission-attempt marker durably before POST. A lost acknowledgement, interrupted CLI or restarted observer observes that original prompt; it never posts it twice. Repeat the exact inputs and idempotency key to recover. Changed inputs or worker boot under the same key are refused.

Completed outputs and available failed/partial trace snapshots are retained with SHA256 and byte lengths. Downloads resume from a local partial prefix and verify the complete digest. Publication refuses to replace another file, including one created during transfer; the verified partial remains available on collision. CLI interruption does not cancel the Comfy request or end the rental.

While observing or retrieving results, the CLI sends bounded, retrying keepalives through the rental's existing pinned machine connection. This is not worker-registered inference: losing all controller keepalives for more than the fixed15-minute idle window can release the rental. The command does not extend that policy or promise a worker-side hold. The remote observation has its original bounded deadline; a result marked timed out does not mean Comfy was canceled.

Validation includes real CPU OpenSSH authentication and strict host-pin refusal, a loopback Comfy protocol fixture, lost acknowledgement and dead-observer recovery with exactly one POST, partial download/hash recovery, incomplete trace retention, and publication collision refusal. Real rented-GPU qualification is recorded separately; these CPU checks do not prove GPU performance.
