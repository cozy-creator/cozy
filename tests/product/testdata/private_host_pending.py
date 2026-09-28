import json
import sqlite3
import sys
import time

database = sqlite3.connect("file:/var/lib/cozy/pod-supervisor.sqlite?mode=ro", uri=True)
deadline = time.monotonic() + 10
while True:
    rows = list(database.execute("SELECT request_id,retain_work FROM attempts"))
    if not rows:
        print(json.dumps({"pending_host_attempts": 0}))
        break
    if time.monotonic() >= deadline:
        raise AssertionError(rows)
    time.sleep(0.02)
