Two concurrent pulls of the same checkpoint can both finish transferring bytes, then the second fails REPOSITORY_CONFLICT while recording the local repository. Checkpoint and release-lane publication currently use separate CAS transactions.

This draft will publish both as one cached-repository transaction and accept a stale identical request only when the committed repository already satisfies its exact checkpoint and active lane. General repository CAS and incompatible updates remain strict.

Validation pending: focused remote native concurrency tests. No local builds/tests/CI, live SDK deployment, or Store/NFS changes. Separate from PR #195's NFS credentials fix.
