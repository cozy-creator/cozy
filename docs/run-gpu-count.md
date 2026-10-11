# GPU count for a run

`cozy run` uses the largest parallel configuration supported by the package code
and the target machine. A package without parallel execution uses one GPU. Model
repositories, releases and weight lanes do not choose execution width.

Use `--gpus=2` to require exactly two GPUs for GPU work in that run. The machine
rejects an unsupported count rather than silently narrowing or widening it. This
also applies to GPU child calls: a two-GPU request cannot invoke a GPU operation
that only supports one GPU. CPU orchestration itself remains on the CPU.

An explicit count reserves a disjoint group on the target machine. On a four-GPU
machine, two two-GPU runs can use separate groups. An orchestration run retains
its group while children borrow it, so concurrent children cannot consume GPUs
outside that reservation. A busy group queues; it does not reduce the requested
count. `--gpus` does not change a rental's physical size.

The count is durable request intent: retries retain it, changing it under an
existing idempotency key is refused, and child calls inherit it. New clients
require the controller's GPU-count capability and the machine's `run-gpus/1`
capability only when an explicit count is requested. Older peers must not silently
ignore a requested count. The machine validates supported degrees from package
code before preparing weights.
