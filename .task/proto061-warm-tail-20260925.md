# Warm queue outcome handling

- Owner: Codex /root/final_review, 2026-09-25
- Purpose: keep control observations flowing while an outcome's media is mirrored and durably settled; preserve bounded work and durable-before-ack custody.
- Branch: fix/proto061-warm-tail-20260925
- Base: origin/master 9df7ba90a967b75e3d1a4542c595960fd9b044e6
- Evidence: proto061-sdxl-anima-20260925/remote-mixed-headroom, requests 1021-1023. The control receive loop blocked during output mirroring and independently owed export; request 1023 routed against a 4.43-second-old capacity report and was refused while outcomes awaited acknowledgment.
- Validation: targeted product coverage with blocked output mirroring and explicit observation/ack gates; live GPU qualification belongs to the parent session.
