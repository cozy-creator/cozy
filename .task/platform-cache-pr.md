Remove `cozy volume`, its warm/drop handlers, and the now-unused Hub client wrappers. Cache-volume creation and placement belong to Tensorhub; consumers rent machines and submit workloads. No provider volume or cached data is changed, and no datacenter flag is introduced.

Validation: isolated build/help readback; no full CI or provider operations.
