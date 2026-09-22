# Runtime source peer for on-demand Python provisioning

CI builds the exact Runtime commit named by COZY_RUNTIME_SOURCE with the existing
read-only checkout and source-addressed CPU wheel recipe. The peer provides
`python-ensure` and the separate `provisionable_minors` inventory capability used
by Creator's package preparation and rental admission.

The wheel retains the source commit in its build identity. No changed development
bytes are presented as a public release, and no PyPI publication is needed for this
qualification. The coordinated Runtime, Hub, and Creator changes must qualify before
delivery. This pin does not update the shared workstation CLI or rented workers.
