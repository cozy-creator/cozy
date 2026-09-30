#!/usr/bin/env bash
# Resolve one public Runtime/TensorFS/evaluator input set for all CI partitions.
set -euo pipefail
runtime_api=https://pypi.org/pypi/cozy-runtime
if [ -n "$RUNTIME_VERSION" ]; then
  [[ $RUNTIME_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'runtime_version must be a public release version' >&2; exit 2; }
  runtime_api="$runtime_api/$RUNTIME_VERSION"
fi
metadata="$RUNNER_TEMP/runtime-release.json"
curl -fsS "$runtime_api/json" -o "$metadata"
if [ -n "$RUNTIME_VERSION" ]; then
  jq -e --arg version "$RUNTIME_VERSION" '.info.version == $version' "$metadata"
fi
runtime_version=$(jq -r .info.version "$metadata")
read -r runtime_url runtime_sha < <(jq -er '[.urls[] | select(.yanked == false) | select(.filename | endswith("cp312-abi3-manylinux_2_28_x86_64.whl"))] |
  if length == 1 then .[0] | "\(.url) \(.digests.sha256)" else error("expected one native Runtime wheel") end' "$metadata")
test -n "$runtime_url" && test -n "$runtime_sha"
wheel="$RUNNER_TEMP/${runtime_url##*/}"
curl -fsSLo "$wheel" "$runtime_url"
echo "$runtime_sha  $wheel" | sha256sum -c
uv python install 3.12
uv tool install --python 3.12 --with-executables-from tensorfs "$wheel[media,model-execution]"
uv tool install --python 3.12 cozy-eval
tensorfs_version=$("$(uv tool dir)/cozy-runtime/bin/python" -c 'from importlib.metadata import version; print(version("tensorfs"))')
eval_version=$("$(uv tool dir)/cozy-eval/bin/python" -c 'from importlib.metadata import version; print(version("cozy-eval"))')
read -r tensorfs_url tensorfs_sha < <(curl -fsS "https://pypi.org/pypi/tensorfs/$tensorfs_version/json" |
  jq -er '[.urls[] | select(.yanked == false) | select(.filename | contains("cp312-abi3-manylinux") and endswith("x86_64.whl"))] |
    if length == 1 then .[0] | "\(.url) \(.digests.sha256)" else error("expected one native TensorFS wheel") end')
test -n "$tensorfs_url" && test -n "$tensorfs_sha"
jq -n --arg runtime_version "$runtime_version" --arg runtime_url "$runtime_url" --arg runtime_sha "$runtime_sha" \
  --arg tensorfs_version "$tensorfs_version" --arg tensorfs_url "$tensorfs_url" --arg tensorfs_sha "$tensorfs_sha" --arg eval_version "$eval_version" \
  '{runtime:{version:$runtime_version,url:$runtime_url,sha256:$runtime_sha},tensorfs:{version:$tensorfs_version,url:$tensorfs_url,sha256:$tensorfs_sha},eval_version:$eval_version}' \
  > "$RUNNER_TEMP/cozy-ci-toolchain.json"
cat "$RUNNER_TEMP/cozy-ci-toolchain.json"
