package producttest

// CO-TENANCY ON ONE RENTAL (run 146's defect, h3a-010). Routing may place several
// packages on one rented machine; the Runtime prepares exactly one package per
// PreparePackageSet. The daemon therefore issues ONE prepare per co-resident package —
// each under a download set naming exactly its own package and its own models — and sends
// the united placements as the one full-replace set. These proofs run the real
// orchestrator against the fakePod's second implementation of the pod side, which
// refuses a multi-package download set with the Runtime's own verdict.
