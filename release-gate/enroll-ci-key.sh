#!/bin/bash
# enroll-ci-key.sh <email> [cozy binary]: the owner's one step for CI sign-in. Enrolls a new device key for the
# account (the Hub emails a code: type it here), stores it as COZY_CI_DEVICE_KEY in the release-gate environment of
# each repo whose workflows run the gate, and erases the local copy. Run it again to rotate the key.
set -euo pipefail
email=$1 cozy=${2:-cozy}
f=$(mktemp); trap 'shred -u "$f" 2>/dev/null || rm -f "$f"' EXIT
"$cozy" login "$email" --export "$f"
for r in cozy-creator/cozy-creator cozy-creator/cozy-runtime; do
  gh api -X PUT "repos/$r/environments/release-gate" --input - >/dev/null <<'JSON'
{"deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
  # master and release/* branches (release.yml, publish.yaml, dispatches) and TensorFS's release tags (publish-tensorfs.yaml).
  for policy in master:branch release/*:branch tensorfs-v*:tag; do
    gh api -X POST "repos/$r/environments/release-gate/deployment-branch-policies" -f name="${policy%:*}" -f type="${policy#*:}" >/dev/null 2>&1 || true
  done
  gh secret set COZY_CI_DEVICE_KEY -R "$r" -e release-gate < "$f"
  echo "$r: release-gate holds the CI key"
done
