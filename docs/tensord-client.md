# Tensord controller consumers

Local roots install and launch `usr/local/bin/tensord`; Runtime wheel intake reads
`.data/scripts/tensord`, and executable identity names `tensord` with the existing
`cozy.machine.v1` API. Process/unit fixtures follow that name. Captured packages,
current journal/event fields, actor authority and stored engine paths are unchanged.

An older-named native root is explicitly refused before install/start/stop can
reinterpret it as a legacy worker. The refusal keeps its journal, identity and launch
record; it does not start, stop or alias the older process. Unreadable identity is
also refused. Initial cutover uses a fresh declared baked candidate or separate
machine home. Standard tensord-to-tensord update/rollback remains the native path.

Use actual owned native fixtures to prove new authorized-key launch/ordinary CLI
execution and preserved old-root refusal. No personal daemon, provider rental,
GPU, deployment or publication is part of this increment.

Tracker: https://github.com/cozy-creator/tracker/issues/336.
