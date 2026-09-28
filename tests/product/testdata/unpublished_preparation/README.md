`model.cozytensors`, `manifest.json` and `vocab.txt` are the real native model files used by
Runtime's `test_real_modeled_package_prepare_can_be_acquired_before_model_selection`. The
model tests serve them unchanged through the catalog protocol and verify them with the
installed native `tfs` CLI. They hold one inline f32 scalar, inline config and a six-byte
tokenizer asset. No numerical inference is simulated by these fixtures.
