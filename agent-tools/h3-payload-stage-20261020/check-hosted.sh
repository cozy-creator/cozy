#!/usr/bin/env bash
# Source/metadata contract only. Fresh hosted Product25 checkout; no real source Hub.
set -euo pipefail
test "$(git rev-parse HEAD)" = 25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2
prefix=agent-tools/h3-payload-stage-20261020
target=crates/tensorfs/tests/h3_payload_transport_contract.rs
test ! -e "$target"
cp "$prefix/transport_contract.rs" "$target"
CARGO_BUILD_JOBS=2 cargo test --locked -p tensorfs --test h3_payload_transport_contract -- --nocapture
