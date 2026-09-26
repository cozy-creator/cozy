# Start a run on a new rental

`cozy run org/package/function --rent-new` buys a fresh managed rental for this
run, using the same model preferences, provider selection and spend limits as
ordinary automatic rental placement. Existing rentals are not reused or ended.
The selected machine, GPU count and hourly price are recorded as usual.

An idempotent retry keeps its original paid acquisition. A daemon restart does
not buy an extra machine for the same request. Replacement after a proved lost
rental continues through the existing durable acquisition lifecycle.

`--rental NAME` selects an existing rental; `--rental-only` allows any suitable
remote rental, including one already running. Neither combines with `--rent-new`.
