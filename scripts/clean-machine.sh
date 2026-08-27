#!/usr/bin/env bash
# Run cl-013's acceptance fixture on a CLEAN MACHINE — a throwaway container that has none
# of this workspace's state.
#
#   scripts/clean-machine.sh --dist <release dir> --asset <name>
#                            [--upgrade <path>] [--endpoint <weightless .tar.gz>]
#
# What "clean" means here, exactly: a fresh user with an empty home, no Go toolchain, no
# Python, no `uv`, no `~/.cozy`, no build tree, and no mount of this repository other than
# `scripts/` (the fixture itself) and the release directory, both READ-ONLY. The one host
# facility it receives is a disposable child in the container's own cgroup: Runtime's
# executor containment is a launch requirement, not something this harness may bypass.
#
# The container has network because a real first install does: `uv` comes off the internet
# and the endpoint's own venv resolves its lock from an index. Nothing else is shared, and
# `--rm` tears the container down whether the fixture passed or not.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ubuntu:24.04}"
DIST=""; ASSET=""; UPGRADE=""; ENDPOINT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dist) DIST="$(cd "$2" && pwd)"; shift 2 ;;
    --asset) ASSET="$2"; shift 2 ;;
    --upgrade) UPGRADE="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    --endpoint) ENDPOINT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    *) echo "usage: $0 --dist <dir> --asset <name> [--upgrade <path>] [--endpoint <path>]" >&2; exit 2 ;;
  esac
done
[ -n "$DIST" ] && [ -n "$ASSET" ] || { echo "refusing: --dist and --asset are required" >&2; exit 2; }

MOUNTS=(-v "$ROOT/scripts:/fixture:ro" -v "$DIST:/dist:ro")
ARGS=(--dist /dist --asset "$ASSET" --prefix /home/tester/.local --home /home/tester/.cozy)
[ -z "$UPGRADE" ]  || { MOUNTS+=(-v "$(dirname "$UPGRADE"):/upgrade:ro");   ARGS+=(--upgrade "/upgrade/$(basename "$UPGRADE")"); }
[ -z "$ENDPOINT" ] || { MOUNTS+=(-v "$(dirname "$ENDPOINT"):/endpoint:ro"); ARGS+=(--endpoint "/endpoint/$(basename "$ENDPOINT")"); }

exec docker run --rm --network bridge --cpus 2 --cgroupns=host \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw "${MOUNTS[@]}" \
  -e DEBIAN_FRONTEND=noninteractive "$IMAGE" bash -c '
set -euo pipefail
apt-get -qq update && apt-get -qq install -y --no-install-recommends ca-certificates curl >/dev/null
useradd -m tester
# Docker normally presents cgroup v2 read-only. Delegate one child of this disposable
# container to the unprivileged fixture user, then enter it before dropping privileges.
# Runtime can create and reclaim only descendants of that child; the child itself dies
# with --rm. No production containment check is disabled for acceptance.
CGROUP_REL="$(cut -d: -f3 /proc/self/cgroup)"
CGROUP_PARENT="/sys/fs/cgroup/${CGROUP_REL#/}"
CGROUP_FIXTURE="$CGROUP_PARENT/cozy-fixture"
mkdir "$CGROUP_FIXTURE"
chown tester:tester "$CGROUP_FIXTURE" "$CGROUP_FIXTURE/cgroup.procs" \
  "$CGROUP_FIXTURE/cgroup.freeze" "$CGROUP_FIXTURE/cgroup.kill"
echo $$ > "$CGROUP_FIXTURE/cgroup.procs"
# uv is the ONE prerequisite a first install has, and it arrives the way a user gets it.
# It brings its own CPython, so this machine never had a Python either.
su tester -c "curl -LsSf https://astral.sh/uv/install.sh | sh" >/dev/null 2>&1
exec su tester -c "PATH=\$HOME/.local/bin:\$PATH nice -n 19 bash /fixture/accept.sh $*"
' -- "${ARGS[@]}"
