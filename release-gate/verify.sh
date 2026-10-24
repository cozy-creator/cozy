#!/bin/bash
# verify.sh <gate run id | gate dir>...: prints the candidate set that passed (candidate.json) and exits 0 only when
# every SKU's gate passed (or the lead accepted its FAIL) and all of them gated the same set. A run id is a
# cozy-creator Actions run that ran release-gate.yml (its gate-<SKU> artifacts); a dir is a hand gate's --out.
# Promotions (promote.sh, release-promote.yml, publish-tensorfs.yaml, cozy-runtime publish.yaml) check what they
# publish against this output.
set -euo pipefail
[ $# -gt 0 ] || { sed -n '2,6s/^# \{0,1\}//p' "$0" >&2; exit 2; }
dirs=() tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for e in "$@"; do
  if [[ $e =~ ^[0-9]+$ ]]; then
    gh run download "$e" -R cozy-creator/cozy-creator -p 'gate-*' -D "$tmp/$e" >&2
    for d in "$tmp/$e"/gate-*; do [ -d "$d" ] && dirs+=("$d"); done
  else
    dirs+=("$e")
  fi
done
[ ${#dirs[@]} -gt 0 ] || { echo "no gate in $*" >&2; exit 1; }
first=
for d in "${dirs[@]}"; do
  grep -qE '^VERDICT (PASS|FAIL .*accepted by the lead)' "$d/verdict.txt" 2>/dev/null \
    || { echo "$d: $(grep -s '^VERDICT' "$d/verdict.txt" || echo 'no verdict')" >&2; exit 1; }
  [ -f "$d/candidate.json" ] || { echo "$d: no candidate.json" >&2; exit 1; }
  c=$(jq -S 'del(.label)' "$d/candidate.json")
  [ -z "$first" ] || [ "$c" = "$first" ] || { echo "$d gated another set than ${dirs[0]}" >&2; exit 1; }
  first=$c
  echo "$d: $(grep '^VERDICT' "$d/verdict.txt")" >&2
done
echo "$first"
