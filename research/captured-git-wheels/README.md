Pinned Git dependencies in local captures
========================================

Actual MiniMax H3 1.26.1 source captures exposed two independent failures. Selected callable wheel capture skips declared Git dependencies under captureOnly, then treats their lock rows as registry entries without wheels. Source snapshots also retain Git dependencies and ask workers to fetch/build them; a worker without Git fails uv sync before execution. Finally StagePrepared's private pack.Build runs publication registry-origin policy with an empty account and rejects a valid retained Hub dependency.

The fix must retain verified pinned source wheels in the invocation snapshot, keep other locked identities unchanged, preserve exact private callable closure custody, and avoid applying publication policy to private runs. Tests must use a real pinned Git repository and prove the captured dependency installs without Git/repository access; an unknown or drifting source must fail visibly. Existing publication restrictions remain intact.

The immediate H3 quality-matrix snapshot uses verified local copies of the same Diffusers and Qwen wheels as a temporary supported packaging path. No app code, GPU default, Hub configuration or remote machine is changed by this task.
