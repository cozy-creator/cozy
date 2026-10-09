# Provider and price components per rental width

Owner: Paul Fidika; implementation agent: linear_resume.
Branch: fix/rental-provider-price-display-20261009.
Base: origin/master8abcceee894fdb79c6d824c08430807586141e6e.

Scope: retain and show each GPU count's selected provider, correctly distinguish compute from bundled storage, and carry the accepted total price ceiling into the existing quote/create path. Normal CLI checks must expose changing provider offers without silently accepting a more expensive fallback. No running rental is changed; no machines are rented for validation.

Current CLI discards the Hub's per-width provider and labels Vast's storage-inclusive total as GPU-only. Thus the RunPod two-H100 quote and the more expensive Vast four-H100 quote appear to be one provider's price ladder.
