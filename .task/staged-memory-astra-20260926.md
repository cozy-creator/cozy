Owner: /root/qwen_memory_fix_astra
Purpose: retain exact-workload total allocator peaks for staged memory placement
Branch: fix/staged-memory-astra-20260926
Base: c067d500 (origin/master fetched 2026-09-26)

Validation: source review and git diff --check only; no tests, CI, GPU runs, or rentals.

Implementation: schema 45 preserves every old measurement and retains the same attested outcome's total allocator peak, canonical request payload/assets, sealed package closure delivery (or already-known published environment), exact model/adapter manifests, SKU, and GPU width. Placement uses max(declared staged residency, exact observed total); missing or mismatched evidence keeps the legacy estimate.

Limitations: historical rows have no inferred total/identity; cold published requests without environment identity retain fallback. T2I evidence does not authorize a different edit request. This change alone does not qualify or unblock run 1067 using run 1066. Fresh matching successful outcome evidence is required.

Source review: checked terminal transaction ownership, candidate SKU/width propagation for both purchases and standing rentals, schema-44 preservation and exact DDL reconstruction, nil/unknown fallback, and fit explanation. gofmt and git diff --check completed; no build, tests, CI, GPU, or rental execution.
