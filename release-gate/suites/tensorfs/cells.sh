# Suite `tensorfs` (gate.sh --suite tensorfs): TensorFS moves weights as fast as before and dedups.
#   straggler        on this host: `tfs download` of BiRefNet clean and through straggler.py, which drips its first
#                    objects as a slow storage GET does (10-01): TensorFS must re-ask a straggler, not wait out its tail
# On the rented machine $M, through `cozy`:
#   probe            the machine's disk, written and read cold; an executor never dials, so no network figure
#   download-cold    a large many-object model ($TFS_MODEL), its time held against the last pass
#   download-again   the same model again: nothing comes from the Hub
#   download-cut/-resume  $TFS_RESUME cancelled part-way, then downloaded again: only the rest comes
#   upload           new random weights (a CozyTensors folder) to a throwaway <account>/ci-tfs-<R>
#   upload-again     the same folder to a second tag: no chunk is sent again (dedup)
#   upload-cut/-resume    new weights cancelled part-way, then uploaded again: only the rest is sent
# check.py holds each against the SKU's last passed suite and records a PASS. The throwaway model is deleted at the end.
TFS_MODEL=${TFS_MODEL:-qwen3.5-9b} TFS_RESUME=${TFS_RESUME:-flux-3-klein} TFS_UPLOAD_MB=${TFS_UPLOAD_MB:-256}
T=$OUT/suite-tensorfs TFS_DEST=
mkdir -p "$T"
# tfs_step <name> <cozy args...>: one command followed to its end: its events in <name>.ndjson, its record
# (`cozy run show`) in <name>.json, its wall time in steps.tsv.
tfs_step() {
  local name=$1 t0 n; shift
  t0=$(date +%s%3N)
  "$CLI" --json "$@" >"$T/$name.ndjson" 2>"$T/$name.err"
  n=$(grep -o '"number":[0-9]*' "$T/$name.ndjson" | head -1 | cut -d: -f2)
  [ -z "$n" ] || timeout 120 "$CLI" --json run show "$n" >"$T/$name.json" 2>/dev/null
  printf '%s\t%s\t%s\n' "$name" "${n:-none}" "$(( $(date +%s%3N) - t0 ))" >> "$T/steps.tsv"
  log "tensorfs $name: run ${n:-none} $(grep -o '"type":"run\.[a-z_]*"' "$T/$name.ndjson" | tail -1)"
}
# tfs_cut <name> <stage> <fraction> <cozy args...>: the command, cancelled once its <stage> progress passes the
# fraction; the bytes it had moved by then go in <name>.cut.
tfs_cut() {
  local name=$1 stage=$2 frac=$3 pid n= moved=; shift 3
  "$CLI" --json "$@" >"$T/$name.ndjson" 2>"$T/$name.err" & pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    [ -n "$n" ] || n=$(grep -o '"number":[0-9]*' "$T/$name.ndjson" | head -1 | cut -d: -f2)
    moved=$(python3 -I "$HERE/suites/tensorfs/check.py" --past "$T/$name.ndjson" "$stage" "$frac")
    if [ -n "$n" ] && [ -n "$moved" ]; then "$CLI" run cancel "$n" -y >/dev/null 2>&1; echo "$moved" > "$T/$name.cut"; break; fi
    sleep 2
  done
  wait "$pid"
  printf '%s\t%s\t\n' "$name" "${n:-none}" >> "$T/steps.tsv"
  log "tensorfs $name: run ${n:-none} cancelled at $(cat "$T/$name.cut" 2>/dev/null || echo '?') bytes"
}
# tfs_weights <dir> <MiB>: new random weights as a CozyTensors folder (tfs import + export, the candidate TensorFS).
tfs_weights() {
  rm -rf "$1" "$1.safetensors"
  uv run -q --no-project --with numpy --with safetensors python -I - "$1.safetensors" "$2" <<'PY' || return 1
import sys
import numpy as np
from safetensors.numpy import save_file
rng = np.random.default_rng()
save_file({f"blocks.{i}.weight": rng.standard_normal((1024, 1024)).astype(np.float16) for i in range(int(sys.argv[2]) // 2)}, sys.argv[1])
PY
  TENSORFS_HOME=$T/tfs-store uv tool run -q --from "${TWF:-tensorfs}" tfs import "$1.safetensors" --name "local/$(basename "$1")" >/dev/null &&
    TENSORFS_HOME=$T/tfs-store uv tool run -q --from "${TWF:-tensorfs}" tfs export "local/$(basename "$1")" "$1" >/dev/null
}
# tfs_straggler <name> <stragglers>: `tfs download` of BiRefNet into a store of its own, through straggler.py.
tfs_straggler() {
  local port pid t0
  port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  python3 -I "$HERE/suites/tensorfs/straggler.py" "$port" "${HUB_URL:-https://tensorhub.com}" "$2" 65536 "$T/$1.proxy" &
  pid=$!
  for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
  t0=$(date +%s%3N)
  TENSORFS_HOME=$T/$1-store uv tool run -q --from "${TWF:-tensorfs}" tfs --endpoint "http://127.0.0.1:$port" --json \
    download birefnet --dir "$T/$1-out" >"$T/$1.out" 2>"$T/$1.err"
  printf '%s\t%s\t%s\n' "$1" "$?" "$(( $(date +%s%3N) - t0 ))" >> "$T/straggler.tsv"
  kill "$pid"; wait "$pid" 2>/dev/null || true  # the proxy ends on TERM
  rm -rf "$T/$1-store" "$T/$1-out"
  log "tensorfs $1: $(tail -n 1 "$T/straggler.tsv")"
}
tfs_cleanup() {
  [ -n "$TFS_DEST" ] || return 0
  "$CLI" model delete "$TFS_DEST" -y >"$T/delete.log" 2>&1 && log "tensorfs: deleted $TFS_DEST" ||
    grep -q model_not_found "$T/delete.log" ||  # no upload landed: nothing to delete
    expect tensorfs-cleanup FAIL "cozy model delete $TFS_DEST: $(tail -c 300 "$T/delete.log")"
  TFS_DEST=
}
trap 'tfs_cleanup; declare -F hub_cleanup >/dev/null && hub_cleanup; end_it' EXIT

ln -sfn "$HERE/suites/tensorfs/probe" "$OUT/ci-tfs-probe"
echo '{}' > "$T/probe.yaml"
tfs_step probe run ./ci-tfs-probe/measure --on "$M" --input "$T/probe.yaml"
tfs_step download-cold model download "$TFS_MODEL" --on "$M" -y
tfs_step download-again model download "$TFS_MODEL" --on "$M" -y
tfs_cut download-cut download 0.3 model download "$TFS_RESUME" --on "$M" -y
tfs_step download-resume model download "$TFS_RESUME" --on "$M" -y
WHO=; for _ in 1 2 3; do  # a dropped connection is asked again
  WHO=$("$CLI" --json whoami 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["account"])' 2>/dev/null) && [ -n "$WHO" ] && break
  sleep 5
done
# Local paths go absolute: `cozy model upload ./folder` resolves them against Calcifer, not here (2026-10-24).
if [ -n "$WHO" ] && tfs_weights "$T/up-a" "$TFS_UPLOAD_MB" && tfs_weights "$T/up-c" "$TFS_UPLOAD_MB"; then
  TFS_DEST=$WHO/ci-tfs-$R
  tfs_step upload model upload "$T/up-a" "$TFS_DEST@a" --on "$M" -y
  tfs_step upload-again model upload "$T/up-a" "$TFS_DEST@b" --on "$M" -y
  tfs_cut upload-cut "uploading to" 0.3 model upload "$T/up-c" "$TFS_DEST@c" --on "$M" -y
  tfs_step upload-resume model upload "$T/up-c" "$TFS_DEST@c" --on "$M" -y
  tfs_cleanup
else
  expect tensorfs-upload FAIL "no account (${WHO:-not signed in}) or no weights to upload"
fi
# On this host, after the machine's cases: the machine is not left idle while it runs.
tfs_straggler straggler-clean 0
tfs_straggler straggler-slow 3
