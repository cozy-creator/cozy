These are exact artifacts from the real Runtime451 tiny modeled-wheel proof.

- `code_only-1.0.0-py3-none-any.whl` is the actual built project wheel.
- `prepared-code.json` and `observed.pb` came from installed development wheel
  e657d43e59d86e00528461363afa04eeb3dc1f95db504d5d2f759e540d8608cd:
  Worker.prepare_local_package, _apply_desired_state, converge_placement,
  prepare_executor and observed_state. No handler or executor ran.
- `prepared-model.json` is the real native TensorFS checkpoint binding result
  from the same code/environment revision in Runtime451's native CI proof.
  Operation IDs differ; their local revision and environment identities agree.

The current Creator protocol test explicitly adapts these legacy placement
envelopes to installation handles and embedded interface bytes. It also rebinds
the report's installation, placement-set identity and control stream while retaining
STAGED/OFFLINE, zero convergence, closed admission and no slots. This adaptation
is a protocol fixture, not evidence that a current Runtime produced these bytes.
Negative arms alter one refusal condition. Root-owned receipt and reproduction:
outputs/h3-turbo-20260912/revision-observation/runtime-observe.py.

`model.cozytensors`, `manifest.json`, and `vocab.txt` are the real native model
files used by Runtime's `test_real_modeled_package_prepare_can_be_acquired_before_model_selection`.
The cold child-default test serves them unchanged through the catalog protocol
and verifies them with the installed native `tfs` CLI. They contain one inline
f32 scalar, inline config, and a six-byte tokenizer asset. No numerical inference
is simulated by this HTTP fixture.
