#!/usr/bin/env bash
# windows-gpu-hosted-runner.sh prove [gh workflow run flags: --ref <ref> -f version=<release> -f runs=...] | status | delete
# The org's GitHub-hosted Windows T4 runner (xs-061 option C): size gpu-t4-4-core, partner image 2349
# "Microsoft Data Science Windows Server 2022", at most one machine, under a random name that is its only label.
# prove: create it, run windows-gpu.yaml on it (fidika/sdxl/generate unless -f runs= says otherwise), follow the
# run, download its bundle, and delete the runner on every exit. delete removes every windows-gpu-t4-* runner.
set -euo pipefail
org=cozy-creator repo=cozy-creator/cozy-creator prefix=windows-gpu-t4- per_hour=6.12

# ids: the org's windows-gpu-t4-* hosted runners (a failed listing is retried).
lookup() {
  until ids=$(gh api "orgs/$org/actions/hosted-runners" --paginate --jq ".runners[] | select(.name | startswith(\"$prefix\")) | .id"); do
    echo "listing the org's hosted runners failed; again" >&2; sleep 10
  done
}
# Deletes every windows-gpu-t4-* runner, sending DELETE again until the org's list holds none.
delete() {
  lookup
  while [ -n "$ids" ]; do
    for id in $ids; do
      gh api -X DELETE "orgs/$org/actions/hosted-runners/$id" --jq '"\(.name) (id \(.id)): \(.status)"' || echo "DELETE $id failed; again" >&2
    done
    sleep 10
    lookup
  done
  echo "no ${prefix}* hosted runner remains"
}
prove() {
  echo "a T4 job bills about \$$per_hour/h, even on this public repo, and only while the org's Actions budget is above \$0" \
    "(it has run out before): fidika/sdxl/generate alone takes ~0.5 h (~\$3); the job's 6 h hang bound is ~\$37." >&2
  # A runner group restricted to windows-gpu.yaml when the org has one (windows-gpu), else one admitting public repositories.
  group=$(gh api "orgs/$org/actions/runner-groups" --jq '[.runner_groups[] | select(.name == "windows-gpu")][0].id // empty')
  [ -n "$group" ] || { group=$(gh api "orgs/$org/actions/runner-groups" --jq '[.runner_groups[] | select(.allows_public_repositories)][0].id')
    echo "note: no windows-gpu runner group; using group $group, which any workflow of a public repo can reach" >&2; }
  name=$prefix$(openssl rand -hex 6) tag=$(openssl rand -hex 6)
  trap delete EXIT
  printf '{"name": "%s", "size": "gpu-t4-4-core", "image": {"id": "2349", "source": "partner"}, "runner_group_id": %s, "maximum_runners": 1}' \
    "$name" "$group" | gh api -X POST "orgs/$org/actions/hosted-runners" --input - --jq '"\(.name) (id \(.id)): \(.status)"'
  gh workflow run windows-gpu.yaml -R "$repo" -f runner="$name" -f tag="$tag" -f runs=fidika/sdxl/generate "$@"
  run=
  until [ -n "$run" ]; do
    sleep 5
    run=$(gh run list -R "$repo" -w windows-gpu.yaml -e workflow_dispatch -L 20 --json databaseId,displayTitle \
      --jq "[.[] | select(.displayTitle == \"windows-gpu $tag\")][0].databaseId // empty")
  done
  echo "https://github.com/$repo/actions/runs/$run"
  status=0
  gh run watch "$run" -R "$repo" --exit-status || status=$?
  gh run download "$run" -R "$repo" -D "windows-gpu-proof-$run"
  echo "bundle: windows-gpu-proof-$run"
  return "$status"
}

case ${1:-} in
  prove) shift; prove "$@" ;;
  status) gh api "orgs/$org/actions/hosted-runners" --paginate --jq ".runners[] | select(.name | startswith(\"$prefix\"))" ;;
  delete) delete ;;
  *) sed -n 2,6p "$0"; exit 2 ;;
esac
