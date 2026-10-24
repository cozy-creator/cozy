#!/bin/sh
# Vendors Tensorhub's API document byte for byte (hub-client generates from it): scripts/hub-api.sh [ref]
set -eu
sha=$(gh api "repos/cozy-creator/tensorhub/commits/${1:-master}" --jq .sha)
gh api "repos/cozy-creator/tensorhub/contents/api/openapi.yaml?ref=$sha" -H 'Accept: application/vnd.github.raw' >"$(dirname "$0")/../crates/hub-client/openapi.yaml"
echo "cozy-creator/tensorhub api/openapi.yaml @ $sha" >"$(dirname "$0")/../crates/hub-client/SOURCE"
