Concurrent pulls of different lanes in one repository still failed after the identical-pull repair. The cache commit now merges compatible additions under SQLite coordination while refusing a changed requested lane, yanked release, or deleted repository. Authored repository CAS, local aliases, ownership receipts, and full checkpoint validation retain their existing rules.

A pinned warm pull could also return success after native cache GC removed its checkpoint between the presence check and return. One shared native writer guard now covers presence, transfer, validation, and recording. Concurrent pulls still proceed; GC returns STORE_BUSY during a pull and succeeds after completion. Duplicate authored checkpoint-validation code was consolidated without changing its checks.

Validation: real native regressions failed on 4146fe9 for both defects and pass on this branch. Focused cache, authored-CAS, cache-root/GC, cold/warm transport, and pinned/moved-lane tests pass with one build/test CPU. A separate safety test preserves refusal before retaining an incomplete snapshot; runtime-only closure versus whole-snapshot GC semantics remains a tracked follow-up. No wheel, worker image, pod, or global CLI was changed by this PR.

Tracker: https://github.com/cozy-creator/tracker/issues/230
