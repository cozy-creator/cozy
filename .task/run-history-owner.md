Owner: /root
Purpose: let cozy run list show every retained run by default, with bounded API pages.
Branch: fix/run-history-pagination-20260926
Base: c899707a (origin/master)

User reported history stopped after the newest50 runs. No history reset or deletion.
Keep explicit --limit for smaller windows and full terminal viewport navigation.
