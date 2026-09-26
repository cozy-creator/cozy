Owner: /root/scheduler_finish
Purpose: reuse retained source-child environments during remote capture instead of rebuilding Qwen for each H3 submission.
Branch: fix/immediate-remote-run-capture-20260926
Base: 3fe25f2d (updated origin/master)

Reuse stays within the same child's package/version and verified frozen dependency inputs. Each capture reads fresh source/interface and retains independent requirements and bindings. Root owns global deployment and daemon restart.
