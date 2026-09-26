Owner: /root/qwen_memory_fix_astra
Purpose: retain exact-workload total allocator peaks for staged memory placement
Branch: fix/staged-memory-astra-20260926
Base: c067d500 (origin/master fetched 2026-09-26)

Validation: source review and git diff --check only; no tests, CI, GPU runs, or rentals.

Implementation: schema 45 preserves every old measurement and retains the same attested outcome's total allocator peak, canonical request payload/assets, sealed package closure delivery (or already-known published environment), exact model/adapter manifests, SKU, and GPU width. Placement uses max(declared staged residency, exact observed total); missing or mismatched total evidence uses declared staged residency and explicitly reports total memory as unmeasured. Raw legacy working peaks remain stored but cannot establish co-residency across stages. Non-staged selections retain the legacy estimate.

Limitations: historical rows have no inferred total/identity; cold published requests without environment identity cannot reuse total evidence. T2I evidence does not qualify a different edit request. A new staged edit is admitted against declared residency with total memory explicitly unmeasured; actual execution remains unqualified until it succeeds and supplies its own outcome. Runtime admission remains authoritative.

Source review: checked terminal transaction ownership, candidate SKU/width propagation for both purchases and standing rentals, schema-44 preservation and exact DDL reconstruction, nil/unknown fallback, and fit explanation. gofmt and git diff --check completed; no build, tests, CI, GPU, or rental execution.
