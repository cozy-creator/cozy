`cozy run list` stopped at the newest50 records, and snapshot rendering could hide another30. Older runs remained in SQLite but were unreachable from the default history view.

List all retained runs by default. Fetch history in bounded500-row API pages using the oldest run number as the next cursor; preserve state/package filters and terminal scrolling. Explicit `--limit N` still selects a smaller window, and `--limit 0` means all history. Snapshot output no longer silently truncates this paginated list.

Validation: low-priority single-CPU Go build passed. Installed the normal global CLI and verified all1,092 retained runs, numbers1–1,092 exactly once with zero omissions, against the read-only SQLite count. Explicit limits and combined package/state filters passed. In a real terminal session, End reached runs15–1 and the viewport reported all1,092 rows. Both remotely executing films continued across the daemon restart. No history reset/deletion and no full local CI.
