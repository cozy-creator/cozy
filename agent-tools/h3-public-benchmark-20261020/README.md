# Public H3 benchmark wrapper — preparation only

This owned prefix will wrap exact `h3_load.py` source `5ea7092ce12d51630c4114539c4137b8ae467826` with the existing public PEP723 author-main/Outputs Tree pattern. It does not change the benchmark, run a model, acquire a machine, or authorize a transfer.

The fixed staging client is `c7accfc6af5e6775fa12d2d7f6a8977db3c64cd4`, portable binary SHA256 `b0712ca8c21ce695e52e412a135778c648f0dac8b8f4c9e991788a685a31ebe9`. Corrected metadata archive SHA256 is `ceafd89e2e5c61b73f2153663b3ea9de9c45d5b7c0121554758715eaba9da25a`; its unchanged D10 header is `0f03804af2280bac6c81c633cd058f4dfbb5fd71f5db0a3c5829907e4ef21f2e`.

Before any client `--download`, require one sealed/assigned H100 UUID, at least79,305,338,149 free GPU bytes,156,316,919,880 available host/cgroup bytes, and228,760,140,584 disk bytes plus a fixed8GiB headroom on the work filesystem. The parent supplies a hard deadline. Every child uses the existing bounded owned-child helper; any failure stops the sequence. The fixed benchmark keeps all60 measured rows and all5 rotated repetitions; slower complete results remain valid data.

The returned Tree contains receipts, source, and logs only. Model chunks, normalized fixtures, wheels and environments stay outside it. Runtime/native/dependency pins remain unresolved until the parent supplies a coherent verified candidate; generation must refuse incomplete pins. No runnable payload or purchase is created by this preparation.
