Owner: /root/host_wire62_qualify
Purpose: Make optional NFS cache writes use the trusted local Store owner's filesystem identity, preserving root Runtime execution and existing cache data/permissions.
Branch: fix/nfs-cache-store-owner-20260926
Base: 9d02fc3 (origin/master)

No local builds, tests, CI, lint, or vet. Native build and credential/thread-isolation qualification must run remotely or in GitHub. No live pod mutation without root coordination.
