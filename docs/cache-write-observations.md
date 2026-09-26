# Observed cache writes

A model fetch reports bytes newly published to its mounted immutable cache from
verified local objects. The receipt belongs to this pull and deduplicates object
identities. It is a lower bound, not a complete-cache claim: already-held local
objects and objects that were not successfully published contribute no bytes.

The final `pull.cache` JSON event names the exact model, manifest and closure
scope. Combined with the existing verified cache-read count, it lets placement
record evidence from the first download without calling volume existence warm.
Optional cache failures still do not fail a completed local fetch.
