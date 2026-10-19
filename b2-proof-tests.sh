set +e
declare -A r
run() { local name=$1; shift; echo "=== $name: $*"; "$@"; r[$name]=$?; echo "=== $name exit ${r[$name]}"; }
run smoke "$TENSORFS_TEST_PYTHON" -c "import tensorfs, tensorfs.varena as v; print(tensorfs.__version__, tensorfs.Checkpoint.parse, tensorfs.regions, v.seal, v.Ledger.freed_gen, v.Ledger.dispatched, v.Row.running)"
run torch_cpu "$TENSORFS_TEST_PYTHON" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
run glue_limit "$TENSORFS_TEST_PYTHON" crates/tensorfs/scripts/varena-limit.py
run glue_host "$TENSORFS_TEST_PYTHON" b2-host-smoke.py
run varena_ck cargo test -p varena --test checkpoint
run varena_ledger cargo test -p varena --test ledger
run tensord_ledger cargo test -p tensord --test gpu_ledger
run tensord_executor cargo test -p tensord --test device_executor
run tensord_surface cargo test -p tensord --test machine_surface
run tensord_local cargo test -p tensord --lib local_source::
run tensord_roots cargo test -p tensord --test journal_roots
run tensord_bin cargo build -p tensord
export COZY_TEST_TENSORD="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/debug/tensord"
run cozy_run_local cargo test -p cozy --test run_local
run cozy_machine cargo test -p cozy --test machine
fail=0; for k in "${!r[@]}"; do echo "RESULT $k ${r[$k]}"; [ "${r[$k]}" = 0 ] || fail=1; done
exit $fail
