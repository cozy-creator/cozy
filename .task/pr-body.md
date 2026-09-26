`cozy run list` stopped at the newest50 records, and snapshot rendering could hide another30. Older runs remained in SQLite but were unreachable from the default history view.

List all retained runs by default. Fetch history in bounded500-row API pages using the oldest run number as the next cursor; preserve state/package filters and terminal scrolling. Explicit `--limit N` still selects a smaller window, and `--limit 0` means all history. Snapshot output no longer silently truncates this paginated list.

Validation: pending one low-priority CLI build and ordinary CLI readback against the existing history. No history reset or deletion; no full local CI.
