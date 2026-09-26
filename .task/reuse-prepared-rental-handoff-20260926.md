Owner: /root/h3_cozy_astra
Purpose: reuse an already observed exact private serving placement without claiming package preparation while the prior request's output custody is settling.
Branch: fix/reuse-prepared-rental-handoff-20260926
Base: a5e6c8b6c175dad37a069b46081398ff2962c146 (origin/master fetched 2026-09-26)

Evidence: consecutive H3 runs1085/1086 on Titus retain executor PID37048/epoch3, but both H100s sit at0% utilization for14.056s from worker release to next invoke. Creator repeatedly parks the successor behind the first open attempt while mirroring outputs, then re-prepares the same installation/models. Retain exact installation/model/adapter identity, worker observation, mode/parent, and admission checks; do not move terminal custody acceptance or acknowledgements earlier.

No deployment during the unchanged baseline cohort. Source review only; no tests or CI under user instruction. Root owns subsequent merge/build/CLI refresh and qualification.
