#!/usr/bin/env bash
# One authenticated Ubuntu media dependency bundle, installed offline by each runner.
set -euo pipefail
mode=${1:?usage: ci-media.sh prepare|install BUNDLE}
media_bundle=$(realpath -m "${2:?media bundle directory required}")
# This helper is CI infrastructure for the same distro/architecture, never a local installer.
source /etc/os-release
media_arch=$(dpkg --print-architecture)
if [ "$ID/$VERSION_ID/$media_arch" != ubuntu/24.04/amd64 ]; then
  echo 'CI media bundle requires Ubuntu 24.04 amd64' >&2
  exit 2
fi
apt_config=(-o "Dir::Etc::sourcelist=/dev/null" -o "Dir::Etc::sourceparts=$media_bundle/sources"
  -o "Dir::State::lists=$media_bundle/lists" -o "Dir::Cache::archives=$media_bundle/archives"
  -o 'Dir::Cache::pkgcache=' -o 'Dir::Cache::srcpkgcache=')

case "$mode" in
  prepare)
    if [ -e "$media_bundle" ]; then
      echo 'prepare requires a new media bundle directory' >&2
      exit 2
    fi
    mkdir -p "$media_bundle/sources" "$media_bundle/lists/partial" "$media_bundle/archives/partial"
    cp /etc/apt/sources.list.d/ubuntu.sources "$media_bundle/sources/ubuntu.sources"
    # Standard APT checks signed distro indexes and package hashes. Empty status makes
    # acquisition include transitive dependencies already installed on the prepare runner.
    sudo apt-get "${apt_config[@]}" update
    sudo apt-get "${apt_config[@]}" -o Dir::State::status=/dev/null \
      --download-only --no-install-recommends --assume-yes install ffmpeg
    # APT creates private lock/partial paths. Only this completed CI bundle is
    # transferred to the invoking owner so the manifest and artifact can read it.
    sudo chown -R "$(id -u):$(id -g)" "$media_bundle"
    printf '%s\t%s\t%s\n' "$ID" "$VERSION_ID" "$media_arch" > "$media_bundle/platform.tsv"
    : > "$media_bundle/packages.tsv"
    shopt -s nullglob
    media_packages=("$media_bundle"/archives/*.deb)
    [ "${#media_packages[@]}" -gt 0 ] || { echo 'APT acquired no media packages' >&2; exit 1; }
    for deb in "${media_packages[@]}"; do
      package=$(dpkg-deb --field "$deb" Package)
      version=$(dpkg-deb --field "$deb" Version)
      architecture=$(dpkg-deb --field "$deb" Architecture)
      if [ "$architecture" != all ] && [ "$architecture" != "$media_arch" ]; then
        echo "foreign package architecture: $package $architecture" >&2
        exit 1
      fi
      printf '%s\t%s\t%s\t%s\n' "${deb##*/}" "$package" "$version" "$architecture" >> "$media_bundle/packages.tsv"
    done
    (
      cd "$media_bundle"
      { printf '%s\0' platform.tsv packages.tsv; find sources lists archives -type f -print0; } |
        sort -z | xargs -0 sha256sum > SHA256SUMS
    )
    echo "authenticated media bundle: ${#media_packages[@]} packages"
    cat "$media_bundle/packages.tsv"
    ;;
  install)
    read -r bundle_os bundle_release bundle_arch < <(tr '\t' ' ' < "$media_bundle/platform.tsv")
    if [ "$bundle_os/$bundle_release/$bundle_arch" != "$ID/$VERSION_ID/$media_arch" ]; then
      echo 'media bundle distro or architecture differs from this runner' >&2
      exit 2
    fi
    (cd "$media_bundle" && sha256sum --check SHA256SUMS)
    # Preserve the runner's real installed state. Request only ffmpeg: offline APT
    # resolves from the authenticated bundled indexes and full package closure,
    # retaining already-installed satisfying versions instead of requesting every .deb.
    sudo apt-get "${apt_config[@]}" --no-download --no-install-recommends --assume-yes install ffmpeg
    ffmpeg -hide_banner -version
    ffprobe -hide_banner -version
    ffmpeg -hide_banner -encoders 2>/dev/null |
      awk '$2=="libx264" {video=1} $2=="aac" {audio=1} END {exit !(video && audio)}'
    ffmpeg -hide_banner -decoders 2>/dev/null |
      awk '$2=="h264" {video=1} $2=="aac" {audio=1} END {exit !(video && audio)}'
    echo 'media tools verified: ffmpeg/ffprobe, H.264/AAC encoding and decoding'
    ;;
  *) echo 'usage: ci-media.sh prepare|install BUNDLE' >&2; exit 2 ;;
esac
