#!/usr/bin/env bash
# windows-gpu-hosted-runner.sh create|delete|status|prove [gh workflow run flags: --ref <ref> -f key=value ...]
# The org's GitHub-hosted Windows T4 runner (xs-061 option C): size gpu-t4-4-core, partner image 2349
# "Microsoft Data Science Windows Server 2022", at most one machine, named and labelled windows-gpu-t4.
# prove: create it, run windows-gpu.yaml on it, follow the run, download its bundle, delete it on every exit.
set -euo pipefail
org=cozy-creator repo=cozy-creator/cozy-creator name=windows-gpu-t4

caveat() {
  echo "note: $name bills per job minute (about \$6.12/h) even on this public repo, and only while the org's" \
    "Actions budget is above \$0; cozy-creator's has run out before." >&2
}
# id: the runner's id, empty when the org has none (a failed listing exits).
lookup() { id=$(gh api "orgs/$org/actions/hosted-runners" --paginate --jq ".runners[] | select(.name == \"$name\") | .id"); }
create() {
  caveat
  lookup
  [ -z "$id" ] || { echo "$name exists already (id $id)" >&2; exit 1; }
  # This public repo's jobs reach only a group that admits public repositories.
  group=$(gh api "orgs/$org/actions/runner-groups" --jq '[.runner_groups[] | select(.allows_public_repositories)][0].id')
  printf '{"name": "%s", "size": "gpu-t4-4-core", "image": {"id": "2349", "source": "partner"}, "runner_group_id": %s, "maximum_runners": 1}' \
    "$name" "$group" | gh api -X POST "orgs/$org/actions/hosted-runners" --input - --jq '"\(.name) (id \(.id)): \(.status)"'
}
delete() {
  lookup
  [ -n "$id" ] || { echo "$name: none"; return; }
  gh api -X DELETE "orgs/$org/actions/hosted-runners/$id" --jq '"\(.name) (id \(.id)): \(.status)"'
  # Gone once the org's list no longer has it.
  while [ -n "$id" ]; do sleep 10; lookup; done
  echo "$name deleted"
}
prove() {
  create
  trap delete EXIT
  since=$(date -u +%FT%TZ) run=
  gh workflow run windows-gpu.yaml -R "$repo" -f runner="$name" "$@"
  until [ -n "$run" ]; do
    sleep 5
    run=$(gh run list -R "$repo" -w windows-gpu.yaml -e workflow_dispatch -L 20 --json databaseId,displayTitle,createdAt \
      --jq "[.[] | select(.displayTitle == \"windows-gpu on $name\" and .createdAt >= \"$since\")][0].databaseId // empty")
  done
  echo "https://github.com/$repo/actions/runs/$run"
  status=0
  gh run watch "$run" -R "$repo" --exit-status || status=$?
  gh run download "$run" -R "$repo" -D "windows-gpu-proof-$run"
  echo "bundle: windows-gpu-proof-$run"
  return "$status"
}

case ${1:-} in
  create) create ;;
  delete) delete ;;
  status) gh api "orgs/$org/actions/hosted-runners" --paginate --jq ".runners[] | select(.name == \"$name\")" ;;
  prove) shift; prove "$@" ;;
  *) sed -n 2,5p "$0"; exit 2 ;;
esac
