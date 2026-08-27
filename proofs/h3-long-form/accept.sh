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
       accept.sh finalize-eight <eight-run-dir> <manual-review.json>
       accept.sh download <workflow-id> <out-dir>

Required selection facts (never lane/provider/location inputs):
  RENTAL_ID H3_ENDPOINT GPU_SKU EXPECTED_ENDPOINT_EXECUTION EXPECTED_MODEL_ROOTS

single additionally requires REF_IMAGE REF_VIDEO REF_AUDIO SINGLE_KEY.
two-recover requires SINGLE_RECEIPT_DIR; cancel-two requires TWO_RECEIPT_DIR.
EXPECTED_MODEL_ROOTS is a comma-separated exact set of sha256 snapshot roots.
The script never calls `cozy rent`; that separately authorized paid act is explicit.
EOF
}

base_facts() {
  need RENTAL_ID; need H3_ENDPOINT; need GPU_SKU
  need EXPECTED_ENDPOINT_EXECUTION; need EXPECTED_MODEL_ROOTS
}

preflight() {
  local out=$1 probe control
  base_facts
  fresh "$out"
  mkdir -p "$out"
  probe=$($COZY_BIN rent probe "$RENTAL_ID" --json)
  printf '%s\n' "$probe" > "$out/rental-probe.json"
  control=$($COZY_BIN rent show "$RENTAL_ID" --json)
  printf '%s\n' "$control" > "$out/rental-control.json"
  "$PYTHON_BIN" "$HERE/verify.py" preflight \
    "$out/rental-probe.json" "$out/rental-control.json"
}

workflow_id() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["workflow"])'
}

compose_video() {
  local source=$1 out=$2
  $COZY_BIN video compose "$source" --rental "$RENTAL_ID" \
    --out "$out/composition.json" --json > "$out/compose.json"
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["creative_plan_digest"])' \
    "$out/compose.json"
}

submit_video() {
  local creative_plan=$1 key=$2
  $COZY_BIN video submit "$creative_plan" --rental "$RENTAL_ID" \
    --idempotency-key "$key" --json
}

wait_step() {
  local workflow=$1 ordinal=$2 wanted=$3 state observed status workflow_status
  while :; do
    state=$($COZY_BIN workflow status "$workflow" --json)
    observed=$(printf '%s' "$state" | "$PYTHON_BIN" -c \
      'import json,sys; d=json.load(sys.stdin); n=int(sys.argv[1]); print(d["steps"][n-1]["status"], d["status"], sep="\t")' "$ordinal")
    IFS=$'\t' read -r status workflow_status <<< "$observed"
    [[ $status == "$wanted" ]] && return 0
    case $workflow_status in
      succeeded|failed|canceled) die "workflow reached $workflow_status while step $ordinal was $status, before $wanted could be observed";;
    esac
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
    need SINGLE_RECEIPT_DIR
    "$PYTHON_BIN" "$HERE/verify.py" gate-stage single "$SINGLE_RECEIPT_DIR"
    fresh "$4"; mkdir -p "$4"
    preflight "$4/control"
    creative=$(compose_video "$2" "$4")
    first=$(submit_video "$creative" "$3")
    workflow=$(printf '%s' "$first" | workflow_id)
    printf '%s\n' "$first" > "$4/submit-before-kill.json"
    wait_step "$workflow" 2 in_progress
    service=$($COZY_BIN status --json)
    printf '%s\n' "$service" > "$4/service-before-kill.json"
    pid=$(printf '%s' "$service" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["pid"])')
    kill -9 "$pid"
    while kill -0 "$pid" 2>/dev/null; do sleep 1; done
    $COZY_BIN status --json > "$4/service-after-kill.json"
    $COZY_BIN up -d
    $COZY_BIN status --json > "$4/service-after-restart.json"
    second=$(submit_video "$creative" "$3")
    replay=$(printf '%s' "$second" | workflow_id)
    [[ $replay == "$workflow" ]] || die "restart minted workflow $replay instead of replaying $workflow"
    printf '%s\n' "$second" > "$4/submit-after-restart.json"
    $COZY_BIN workflow follow "$workflow"
    $COZY_BIN workflow download "$workflow" --out "$4/download"
    "$PYTHON_BIN" "$HERE/verify.py" bundle two "$4"
    ;;
  cancel-two)
    [[ $# == 4 ]] || { usage; exit 2; }
    need TWO_RECEIPT_DIR
    "$PYTHON_BIN" "$HERE/verify.py" gate-stage two "$TWO_RECEIPT_DIR"
    fresh "$4"; mkdir -p "$4"
    preflight "$4/control"
    creative=$(compose_video "$2" "$4")
    submitted=$(submit_video "$creative" "$3")
    workflow=$(printf '%s' "$submitted" | workflow_id)
    printf '%s\n' "$submitted" > "$4/submit.json"
    wait_step "$workflow" 1 in_progress
    $COZY_BIN workflow cancel "$workflow" --json > "$4/cancel.json"
    set +e
    $COZY_BIN workflow follow "$workflow" --json > "$4/follow.ndjson"
    follow_status=$?
    set -e
    if [[ $follow_status == 0 ]]; then
      die "canceled workflow followed to success"
    fi
    [[ $follow_status == 12 ]] || die "workflow follow exited $follow_status, not the canceled exit 12"
    state=$($COZY_BIN workflow status "$workflow" --json)
    printf '%s\n' "$state" > "$4/terminal.json"
    [[ $(printf '%s' "$state" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])') == canceled ]] || \
      die "workflow did not settle canceled"
    $COZY_BIN workflow download "$workflow" --out "$4/download"
    "$PYTHON_BIN" "$HERE/verify.py" bundle cancel "$4"
    ;;
  eight)
    [[ $# == 5 ]] || { usage; exit 2; }
    need SINGLE_RECEIPT_DIR; need TWO_RECEIPT_DIR; need CANCEL_RECEIPT_DIR
    gate=$("$PYTHON_BIN" "$HERE/verify.py" gate-eight \
      "$SINGLE_RECEIPT_DIR" "$TWO_RECEIPT_DIR" "$CANCEL_RECEIPT_DIR" "$4")
    fresh "$5"; mkdir -p "$5"
    printf '%s\n' "$gate" > "$5/prerequisite-gate.json"
    preflight "$5/control"
    creative=$(compose_video "$2" "$5")
    submitted=$(submit_video "$creative" "$3")
    workflow=$(printf '%s' "$submitted" | workflow_id)
    printf '%s\n' "$submitted" > "$5/submit.json"
    $COZY_BIN workflow follow "$workflow"
    $COZY_BIN workflow download "$workflow" --out "$5/download"
    "$PYTHON_BIN" "$HERE/verify.py" bundle eight "$5"
    echo "eight-shot automated verification passed; watch/listen it, bind its verification digest in the manual review, then run finalize-eight" >&2
    ;;
  finalize-eight)
    [[ $# == 3 ]] || { usage; exit 2; }
    "$PYTHON_BIN" "$HERE/verify.py" finalize-eight "$2" "$3"
    ;;
  download)
    [[ $# == 3 ]] || { usage; exit 2; }
    $COZY_BIN workflow download "$2" --out "$3"
    ;;
  *) usage; exit 2;;
esac
