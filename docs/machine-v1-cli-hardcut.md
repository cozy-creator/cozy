# CLI machine.v1 hard cut

Owner: Codex root; tracker #331. Base d5ac442a.

The machine serves cozy.machine.v1 and explicitly tells old clients to upgrade.
The CLI must use that API for run, observation, control, byte custody, machine
status and package/software operations. The owner authorized removing worker.v1
and its protocol repository; no legacy fallback or old-peer test requirement remains.

The original APIM C1 checkpoint258572e5 and C2afadc633 remain untouched. C1 is
adopted into this fresh-master task tree. The newer master warm-run test/helpers
are retained where APIM's deletion conflicted. Recheck the stopped unused-declaration
sweep before claiming a build or product pass. The earlier118-test run had20
failures;19 failing functions were removed by C1. Review their behavior coverage
separately from obsolete worker.v1 fixture mechanics.

Integration sequence:
1. Restore the C1 build and current native run/cancel/reattach/output proof.
2. Merge C1 as a working increment.
3. Carry only C2's incremental machine-client deletion onto current master,
   correcting test/CI references and update instructions; prove and merge.
4. Retype remaining records/orchestrator consumers into their own domain or
   machine.v1 types, remove the old package and build/vendoring dependencies.
5. Audit repository consumers and unique/dirty history before authorized retirement.

A watcher disconnect only detaches observation. Cancel intent is durable before
remote contact; unknown acceptance remains a reconciliation obligation, and
reconnect never re-executes already accepted work. Reboot and current release-root
semantics require native proof. Do not preserve retired generation/CloseSubmission/
collection-ack mechanisms merely to keep old fixtures green.

Worker images remain simple OS/CUDA/PyTorch bases. The controller updates software
at runtime, including rollback by target version. No image qualification/admission,
inventory, promotion hold or image accounting is introduced here.

Proof receipts and a per-failure census are retained under
outputs/codex-apim-takeover-20261006/. This initial checkpoint is under review.
