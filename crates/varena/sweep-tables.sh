#!/usr/bin/env bash
# One-shot: removes this user's varena tables in /dev/shm that nobody has open, and prints how
# many went. Every open of a table holds a lock on it (and each row its own), so a table with
# no lock in /proc/locks has no live row. Run it while no varena process is starting.
set -euo pipefail
dev=$(mountpoint -d /dev/shm | awk -F: '{ printf "%02x:%02x", $1, $2 }')
locked=$(grep -oE '[0-9a-f]+:[0-9a-f]+:[0-9]+' /proc/locks | sort -u)
n=0
for f in /dev/shm/cozy-varena-*; do
  [ -O "$f" ] || continue
  grep -qx "$dev:$(stat -c %i "$f")" <<<"$locked" && continue
  rm -f -- "$f"
  n=$((n + 1))
done
echo "$n"
