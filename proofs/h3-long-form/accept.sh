#!/usr/bin/env bash
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
COZY_BIN=${COZY_BIN:-cozy}
PYTHON_BIN=${PYTHON_BIN:-python3}

die() { echo "REFUSED: $*" >&2; exit 2; }
need() { local name=$1; [[ -n ${!name:-} ]] || die "$name is required"; }
fresh() { [[ ! -e $1 ]] || die "$1 already exists; acceptance never reuses output roots"; }

usage() {
  cat <<'EOF'
Usage: accept.sh preflight <out-dir>
       accept.sh single <out-dir>
       accept.sh two-recover <source.yaml> <workflow-key> <out-dir>
       accept.sh cancel-two <source.yaml> <workflow-key> <out-dir>
       accept.sh eight <source.yaml> <workflow-key> <manual-review.json> <out-dir>
       accept.sh download <workflow-id> <out-dir>

Required selection facts (never lane/provider/location inputs):
  RENTAL_ID H3_ENDPOINT GPU_SKU EXPECTED_ENDPOINT_EXECUTION EXPECTED_MODEL_ROOTS
  ASSEMBLER_ENDPOINT

single additionally requires REF_IMAGE REF_VIDEO REF_AUDIO SINGLE_KEY.
EXPECTED_MODEL_ROOTS is a comma-separated exact set of sha256 snapshot roots.
The script never calls `cozy rent`; that separately authorized paid act is explicit.
EOF
}

base_facts() {
  need RENTAL_ID; need H3_ENDPOINT; need GPU_SKU
  need EXPECTED_ENDPOINT_EXECUTION; need EXPECTED_MODEL_ROOTS; need ASSEMBLER_ENDPOINT
}

preflight() {
  local out=$1 control
  base_facts
  fresh "$out"
  mkdir -p "$out"
  control=$($COZY_BIN rent show "$RENTAL_ID" --json)
  printf '%s\n' "$control" > "$out/rental-control.json"
  "$PYTHON_BIN" - "$out/rental-control.json" <<'PY'
import json, os, sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
want_execution = os.environ["EXPECTED_ENDPOINT_EXECUTION"]
want_roots = sorted(x.strip() for x in os.environ["EXPECTED_MODEL_ROOTS"].split(",") if x.strip())
got_roots = sorted(row.get("model_root_digests") or [])
checks = {
    "endpoint_execution_digest": (row.get("endpoint_execution_digest"), want_execution),
    "accelerator": (row.get("accelerator"), os.environ["GPU_SKU"]),
    "model_root_digests": (got_roots, want_roots),
}
for name, (got, want) in checks.items():
    if got != want:
        raise SystemExit(f"REFUSED: {name} is {got!r}, expected {want!r}")
endpoint = str(row.get("endpoint", ""))
repo = os.environ["H3_ENDPOINT"].strip("/")
if not endpoint.startswith(repo + "/v"):
    raise SystemExit(f"REFUSED: rental endpoint {endpoint!r} is outside {repo!r}")
print("preflight: exact endpoint execution, GPU, and model-root set match")
PY
}

workflow_id() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["workflow"])'
}

submit_video() {
  local source=$1 key=$2
  $COZY_BIN video submit "$source" --h3 "$H3_ENDPOINT" \
    --assembler "$ASSEMBLER_ENDPOINT" --rental "$RENTAL_ID" \
    --idempotency-key "$key" --json
}

wait_step() {
  local workflow=$1 ordinal=$2 wanted=$3 state status
  while :; do
    state=$($COZY_BIN workflow status "$workflow" --json)
    status=$(printf '%s' "$state" | "$PYTHON_BIN" -c \
      'import json,sys; d=json.load(sys.stdin); n=int(sys.argv[1]); print(d["steps"][n-1]["status"]); print(d["status"], file=sys.stderr)' "$ordinal")
    [[ $status == "$wanted" ]] && return 0
    case $status in completed|failed|canceled) die "step $ordinal reached $status before $wanted could be observed";; esac
    sleep 2
  done
}

case ${1:-} in
  preflight)
    [[ $# == 2 ]] || { usage; exit 2; }
    preflight "$2"
    ;;
  single)
    [[ $# == 2 ]] || { usage; exit 2; }
    need REF_IMAGE; need REF_VIDEO; need REF_AUDIO; need SINGLE_KEY
    fresh "$2"; mkdir -p "$2/media"
    preflight "$2/control"
    $COZY_BIN run "$H3_ENDPOINT/v1/reference_media_to_video" \
      --worker "$RENTAL_ID" --in "$HERE/fixtures/single-ref2va.json" \
      --asset "references.0.image=$REF_IMAGE" \
      --asset "references.1.video=$REF_VIDEO" \
      --asset "references.2.audio=$REF_AUDIO" \
      --idempotency-key "$SINGLE_KEY" --out "$2/media" --json > "$2/request.json"
    attempt=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("attempt_key", ""))' "$2/request.json")
    if [[ -n $attempt ]]; then
      $COZY_BIN logs "$attempt" --json > "$2/triage.json"
    fi
    "$PYTHON_BIN" "$HERE/verify.py" single "$2"
    ;;
  two-recover)
    [[ $# == 4 ]] || { usage; exit 2; }
    fresh "$4"; mkdir -p "$4"
    preflight "$4/control"
    first=$(submit_video "$2" "$3")
    workflow=$(printf '%s' "$first" | workflow_id)
    printf '%s\n' "$first" > "$4/submit-before-kill.json"
    wait_step "$workflow" 2 in_progress
    service=$($COZY_BIN status --json)
    pid=$(printf '%s' "$service" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["pid"])')
    kill -9 "$pid"
    while kill -0 "$pid" 2>/dev/null; do sleep 1; done
    $COZY_BIN up -d
    second=$(submit_video "$2" "$3")
    replay=$(printf '%s' "$second" | workflow_id)
    [[ $replay == "$workflow" ]] || die "restart minted workflow $replay instead of replaying $workflow"
    printf '%s\n' "$second" > "$4/submit-after-restart.json"
    $COZY_BIN workflow follow "$workflow"
    $COZY_BIN workflow download "$workflow" --out "$4/download"
    "$PYTHON_BIN" "$HERE/verify.py" bundle two "$4/download"
    ;;
  cancel-two)
    [[ $# == 4 ]] || { usage; exit 2; }
    fresh "$4"; mkdir -p "$4"
    preflight "$4/control"
    submitted=$(submit_video "$2" "$3")
    workflow=$(printf '%s' "$submitted" | workflow_id)
    printf '%s\n' "$submitted" > "$4/submit.json"
    wait_step "$workflow" 2 in_progress
    $COZY_BIN workflow cancel "$workflow" --json > "$4/cancel.json"
    if $COZY_BIN workflow follow "$workflow" --json > "$4/follow.ndjson"; then
      die "canceled workflow followed to success"
    fi
    state=$($COZY_BIN workflow status "$workflow" --json)
    printf '%s\n' "$state" > "$4/terminal.json"
    [[ $(printf '%s' "$state" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])') == canceled ]] || \
      die "workflow did not settle canceled"
    $COZY_BIN workflow download "$workflow" --out "$4/download"
    "$PYTHON_BIN" "$HERE/verify.py" bundle cancel "$4/download"
    ;;
  eight)
    [[ $# == 5 ]] || { usage; exit 2; }
    need SINGLE_RECEIPT_DIR; need TWO_RECEIPT_DIR; need CANCEL_RECEIPT_DIR
    "$PYTHON_BIN" "$HERE/verify.py" gate-eight \
      "$SINGLE_RECEIPT_DIR" "$TWO_RECEIPT_DIR" "$CANCEL_RECEIPT_DIR" "$4"
    fresh "$5"; mkdir -p "$5"
    preflight "$5/control"
    submitted=$(submit_video "$2" "$3")
    workflow=$(printf '%s' "$submitted" | workflow_id)
    printf '%s\n' "$submitted" > "$5/submit.json"
    $COZY_BIN workflow follow "$workflow"
    $COZY_BIN workflow download "$workflow" --out "$5/download"
    "$PYTHON_BIN" "$HERE/verify.py" bundle eight "$5/download"
    ;;
  download)
    [[ $# == 3 ]] || { usage; exit 2; }
    $COZY_BIN workflow download "$2" --out "$3"
    ;;
  *) usage; exit 2;;
esac
