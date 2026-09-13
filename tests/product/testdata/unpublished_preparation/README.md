These are exact artifacts from the real Runtime451 tiny modeled-wheel proof.

- `code_only-1.0.0-py3-none-any.whl` is the actual built project wheel.
- `prepared-code.json` and `observed.pb` came from installed development wheel
  e657d43e59d86e00528461363afa04eeb3dc1f95db504d5d2f759e540d8608cd:
  Worker.prepare_local_package, _apply_desired_state, converge_placement,
  prepare_executor and observed_state. No handler or executor ran.
- `prepared-model.json` is the real native TensorFS checkpoint binding result
  from the same code/environment revision in Runtime451's native CI proof.
  Operation IDs differ; their local revision and environment identities agree.

The Creator product test rebinds only the report's control-stream envelope. It
retains actual STAGED/OFFLINE, zero convergence, closed admission and no slots.
Negative arms alter one refusal condition. Root-owned receipt and reproduction:
outputs/h3-turbo-20260912/revision-observation/runtime-observe.py.
