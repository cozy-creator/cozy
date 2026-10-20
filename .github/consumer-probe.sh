#!/usr/bin/env bash
# xs-060 consumer probe: the public install path, a first user's walkthrough, upgrade and uninstall.
# Every command's exit code and output go to transcript.md. Never fails early: the transcript is the result.
set -u
OS=$1 PS=${2:-}
# The install line: the public one, or a release candidate's mirror (COZY_DOWNLOAD_URL).
LINE=https://tensorhub.com/install; [ -n "${COZY_DOWNLOAD_URL:-}" ] && LINE=$COZY_DOWNLOAD_URL/cozy-installer
T=$PWD/transcript.md
echo "# $OS" > "$T"
win=0; [ "${RUNNER_OS:-}" = Windows ] && win=1
cozy=$HOME/.cozy/bin/cozy; [ $win = 1 ] && cozy=$HOME/.cozy/bin/cozy.exe

# rec <label> <bound-seconds> <command…>: run it, bounded, recording exit code, wall time and output.
rec() {
  local label=$1 bound=$2; shift 2
  local start=$SECONDS out code
  out=$(perl -e 'alarm shift; exec @ARGV' "$bound" "$@" 2>&1 < /dev/null); code=$?
  [ $code = 142 ] && out="$out
[probe: killed after ${bound}s]"
  { echo; echo "## $label"; echo "\`$*\` → exit $code, $((SECONDS - start)) s"; echo '```'; printf '%s\n' "$out" | head -c 6000; echo '```'; } >> "$T"
  printf '\n=== %s (exit %s)\n%s\n' "$label" "$code" "$out"
}

section() { echo; echo "# $1" >> "$T"; echo "##### $1"; }

if [ -n "${DEV_BINARY:-}" ]; then
  section "Dev build $("$DEV_BINARY" --version)"
  mkdir -p "$(dirname "$cozy")" && cp "$DEV_BINARY" "$cozy"
elif true; then
section "Fresh install, public path"
if [ $win = 1 ]; then
  rec install 600 $PS -NoProfile -ExecutionPolicy Bypass -c "irm $LINE.ps1 | iex; 'this session finds: ' + (Get-Command cozy -ErrorAction SilentlyContinue).Source"
  rec "user PATH" 30 powershell -NoProfile -c "[Environment]::GetEnvironmentVariable('Path', 'User')"
  rec "new PowerShell finds cozy" 60 powershell -NoProfile -c "\$env:Path = [Environment]::GetEnvironmentVariable('Path','User') + ';' + [Environment]::GetEnvironmentVariable('Path','Machine'); (Get-Command cozy).Source; cozy --version"
else
  rec install 600 sh -c "curl -LsSf $LINE.sh | sh"
  rec "login sh finds cozy" 30 env -i HOME="$HOME" PATH=/usr/bin:/bin sh -lc 'command -v cozy'
  rec "login bash finds cozy" 30 env -i HOME="$HOME" PATH=/usr/bin:/bin bash -lc 'command -v cozy'
  command -v zsh > /dev/null && rec "interactive zsh finds cozy" 30 env -i HOME="$HOME" PATH=/usr/bin:/bin zsh -ic 'command -v cozy'
fi
fi
if [ ! -x "$cozy" ]; then echo "no cozy installed" >> "$T"; exit 0; fi

section "First user, not signed in"
rec "no arguments" 60 "$cozy"
rec version 30 "$cozy" --version
rec help 30 "$cozy" --help
rec status 120 "$cozy" status
rec whoami 60 "$cozy" whoami
rec "hub list" 60 "$cozy" hub list
rec "package search (no query)" 60 "$cozy" package search
rec "package search sdxl" 60 "$cozy" package search sdxl
rec "model search sdxl" 60 "$cozy" model search sdxl
rec "package info fidika/sdxl" 60 "$cozy" package info fidika/sdxl
rec "run fidika/sdxl (its functions)" 60 "$cozy" run fidika/sdxl
rec "run locally, machine not enabled" 120 "$cozy" run fidika/sdxl/generate prompt='a lighthouse at dusk' seed=7
rec "rent catalog" 60 "$cozy" machine rent
rec "credits" 60 "$cozy" credits
rec "run --on new, signed out" 120 "$cozy" run fidika/sdxl/generate --on new prompt='a lighthouse at dusk' seed=7
rec "doctor" 120 "$cozy" doctor
rec "tools list" 60 "$cozy" tools list
rec "run list" 60 "$cozy" run list
rec "machine list" 60 "$cozy" machine list

section "Local machine"
if [ "${RUNNER_OS:-}" = Linux ]; then
  # A desktop login has a user service manager; a runner's user gets one by lingering.
  sudo loginctl enable-linger "$(id -un)"
  export XDG_RUNTIME_DIR=/run/user/$(id -u)
  until [ -S "$XDG_RUNTIME_DIR/bus" ]; do sleep 1; done
  export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
fi
rec "machine enable" 600 "$cozy" machine enable
rec "machine show local" 60 "$cozy" machine show local
rec "doctor, enabled" 300 "$cozy" doctor
bound=600; [ "${RUNNER_OS:-}" = macOS ] && bound=2400
rec "run locally: sdxl" $bound "$cozy" run fidika/sdxl/generate prompt='a lighthouse at dusk' seed=7 steps=2
rec "birefnet functions" 120 "$cozy" run fidika/birefnet
# Your own code: a script with the Runtime, local on every OS (cohort wheels for macOS and Windows).
printf '%s\n' '# /// script' '# requires-python = ">=3.12"' '# dependencies = ["cozy-runtime"]' '# [tool.cozy]' \
  '# accelerator = false' '# ///' 'def main(name: str = "you") -> str:' '    return f"hello, {name}"' > greet.py
rec "run a local script" 900 "$cozy" run ./greet.py name=consumer
rec "run list after" 60 "$cozy" run list
rec "bad input name" 120 "$cozy" run fidika/sdxl/generate nosuch=1 prompt=x
rec "bad input value" 120 "$cozy" run fidika/sdxl/generate prompt=x seed=abc
rec "unknown package" 60 "$cozy" run fidika/nosuchpackage/generate prompt=x
rec "unknown function" 60 "$cozy" run fidika/sdxl/nosuch prompt=x
rec "--gpus 0" 300 "$cozy" run fidika/sdxl/generate --gpus 0 prompt=x seed=1
ls -la "$HOME/.cozy/outputs" >> "$T" 2>&1

section "Offline (Calcifer behind a dead proxy)"
rec "down" 120 "$cozy" down
export HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost
rec "offline model search" 120 "$cozy" model search sdxl
rec "offline run" 180 "$cozy" run fidika/sdxl/generate prompt=x seed=1
rec "offline update" 120 "$cozy" update
rec "down (offline Calcifer)" 120 "$cozy" down
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY

section "Uninstall"
rec "machine disable" 300 "$cozy" machine disable
rec "uninstall --yes" 300 "$cozy" uninstall --yes
rec "left behind" 30 sh -c 'ls -la ~/.cozy ~/.config/cozy 2>&1; grep -n cozy ~/.profile ~/.bashrc ~/.bash_profile ~/.zshrc ~/.zshenv 2>/dev/null; true'
if [ $win = 1 ]; then
  rec "user PATH after" 30 powershell -NoProfile -c "[Environment]::GetEnvironmentVariable('Path', 'User')"
else
  rec "login shell after" 30 env -i HOME="$HOME" PATH=/usr/bin:/bin sh -lc 'command -v cozy || echo gone'
fi

[ -n "${DEV_BINARY:-}" ] && exit 0
section "Upgrade from v0.10.1"
old=https://github.com/cozy-creator/cozy-creator/releases/download/v0.10.1
if [ $win = 1 ]; then
  rec "install v0.10.1" 600 env -u COZY_DOWNLOAD_URL $PS -NoProfile -ExecutionPolicy Bypass -c "irm $old/cozy-installer.ps1 | iex"
else
  rec "install v0.10.1" 600 env -u COZY_DOWNLOAD_URL sh -c "curl -LsSf $old/cozy-installer.sh | sh"
fi
rec "old version" 30 "$cozy" --version
rec "update" 600 "$cozy" update
rec "new version" 30 "$cozy" --version
rec "status after update" 120 "$cozy" status
rec "uninstall after upgrade" 300 "$cozy" uninstall --yes
exit 0
