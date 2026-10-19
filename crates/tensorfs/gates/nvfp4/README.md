# nvfp4 lane check

`check.py` reads ~24 nvfp4 tensors spread over a lane through Tensorhub by byte range (a 128-row band each), decodes them with TensorFS's reference decoder, and compares them with the bf16 lane: relative L2 per tensor and the median. PASS: median ≤ 15% (correct layouts measure 9–12%) and no tensor ≥ 25% (a wrong layout, flagged). Exit 1 on any FAIL. Public lanes are read anonymously; `TENSORHUB_TOKEN` reads private ones.

```
W=<a tensorfs dev wheel of this branch>
uv run --no-project --python 3.12 --with numpy --with $W python -I check.py --tensorhub local paul/flux-3-klein paul/flux-3-klein-base paul/minimax-h3@1.0.0-rc.3
uv run --no-project --python 3.12 --with numpy --with $W python -I selftest.py   # a tiny local Hub: ModelOpt and ComfyUI layouts pass, a wrong one fails
```
