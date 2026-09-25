Owner: /root/h3_predefined
Purpose: Capture published invocable dependency closures and generated caller interfaces for Runtime-owned execution.
Branch: fix/cl-297-published-child-closure
Base: b21ae398d36f44c7171acc1772f6dec2ec7857e3
Tracker: cl-297

Qualification: real Creator CLI published CPU parent calling a separately published child, then H3 using reference-image on one worker. Preserve exact dependency identity, callee defaults, internal-call restrictions and offline execution.
