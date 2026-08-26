#!/usr/bin/env bash
# th-039's LIVE RUN for the deterministic wheel packer. Not a test suite (README: no
# automated tests; verification is running the real thing) — every line below is the real
# `cozy pack` binary against real endpoint trees, and it PRINTS what it observed.
#
#   CGO_ENABLED=0 go build -o cozy ./cmd/cozy       # the binary the product ships
#   scripts/wheel-live.sh [--cozy <bin>] [--endpoints <dir>] [--runtime <dir>] [<section> …]
#
# Sections: determinism | seat | arms | install   (default: all four)
#
#   determinism  the same tree packed under different umask, TZ, locale, working
#                directory, HOME, file mtimes, file modes and absolute path produces
#                BYTE-IDENTICAL wheels. This is the gate, not a hope.
#   seat         the same tree packed inside a container — another rootfs, another uid,
#                another libc, no $HOME — agrees byte for byte with this host.
#   arms         planted trees that must refuse, each by its own name, observed red.
#   install      the real SDXL endpoint tree packed, `pip install`ed into a throwaway
#                venv, and then `cozy-runtime describe`d THROUGH THE INSTALLED WHEEL —
#                no source tree on sys.path anywhere.
#
# CPU only. No pod, no model download, no inference.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COZY="$HERE/cozy"
ENDPOINTS="${COZY_ENDPOINTS:-$HOME/cozy_v2/serverless-endpoints}"
RUNTIME="${COZY_RUNTIME_SRC:-$HOME/cozy_v2/cozy-runtime}"
SECTIONS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --cozy) COZY="$2"; shift 2 ;;
    --endpoints) ENDPOINTS="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    determinism|seat|arms|install) SECTIONS+=("$1"); shift ;;
    *) echo "usage: $0 [--cozy <bin>] [--endpoints <dir>] [--runtime <dir>] [determinism|seat|arms|install]" >&2; exit 2 ;;
  esac
done
[ ${#SECTIONS[@]} -eq 0 ] && SECTIONS=(determinism seat arms install)
[ -x "$COZY" ] || { echo "refusing: no cozy binary at $COZY (go build -o cozy ./cmd/cozy)" >&2; exit 2; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export COZY_HOME="$WORK/home"

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n' "$1"; }
head_() { printf '\n=== %s\n' "$1"; }
say()  { printf '       %s\n' "$1"; }

# digest_of <wheel-dir> — the sha256 of the one wheel written there, computed by a tool
# that is not the packer, so the recorded digest is corroborated rather than asserted.
digest_of() { sha256sum "$1"/*.whl | cut -d' ' -f1; }

# pack_in <env-assignments...> -- <outdir> <tree> <name> <version>
# Runs the REAL binary in a subshell carrying a hostile environment.
pack_env() {
  local label="$1" outdir="$2" tree="$3" name="$4" version="$5" umask_val="$6" tz="$7" loc="$8" cwd="$9" home="${10}"
  mkdir -p "$outdir" "$home"
  ( cd "$cwd" && umask "$umask_val" && \
    TZ="$tz" LC_ALL="$loc" LANG="$loc" HOME="$home" COZY_HOME="$home/.cozy" \
    SOURCE_DATE_EPOCH=$RANDOM$RANDOM \
    nice -n 19 "$COZY" pack "$tree" --name "$name" --version "$version" --out "$outdir" ) >"$outdir/.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then bad "$label: pack exited $rc"; sed 's/^/       /' "$outdir/.log"; return 1; fi
  return 0
}

# ---------------------------------------------------------------- determinism

section_determinism() {
  head_ "determinism — one tree, four environments, one digest"
  for pair in "h3 h3 0.1.0" "sdxl sdxl 0.1.0"; do
    set -- $pair
    local tree="$ENDPOINTS/$1" name="$2" version="$3"
    [ -d "$tree" ] || { say "skipping $1 (no tree at $tree)"; continue; }

    # A — this box, plainly.
    pack_env "$name/A" "$WORK/$name/a" "$tree" "$name" "$version" 022 UTC C "$HERE" "$WORK/$name/homeA" || continue
    # B — the same box again, to catch anything that varies run to run within one host.
    pack_env "$name/B" "$WORK/$name/b" "$tree" "$name" "$version" 022 UTC C "$HERE" "$WORK/$name/homeA" || continue
    # C — a deliberately hostile environment: restrictive umask, a timezone across the
    # date line, a UTF-8 locale with a different collation, a different working
    # directory, a different HOME, and a SOURCE_DATE_EPOCH the packer must ignore.
    pack_env "$name/C" "$WORK/$name/c" "$tree" "$name" "$version" 077 Pacific/Kiritimati en_US.UTF-8 / "$WORK/$name/homeC" || continue

    # D — a COPY of the tree at a different absolute path, with every mtime moved and
    # every mode changed. Path, clock and permission bits must reach the wheel nowhere.
    local copy="$WORK/$name/copy-at-another-path/$name"
    mkdir -p "$(dirname "$copy")" && cp -r "$tree" "$copy"
    find "$copy" -type f -exec touch -t 200001011111.11 {} +
    find "$copy" -type d -exec chmod 0777 {} +
    find "$copy" -type f -exec chmod 0600 {} +
    pack_env "$name/D" "$WORK/$name/d" "$copy" "$name" "$version" 002 America/New_York C.UTF-8 /tmp "$WORK/$name/homeD" || continue

    local da db dc dd
    da=$(digest_of "$WORK/$name/a"); db=$(digest_of "$WORK/$name/b")
    dc=$(digest_of "$WORK/$name/c"); dd=$(digest_of "$WORK/$name/d")
    say "$name A umask 022 TZ=UTC        LC=C            sha256:$da"
    say "$name B same env, second run    (rerun)         sha256:$db"
    say "$name C umask 077 TZ=Kiritimati LC=en_US.UTF-8  sha256:$dc"
    say "$name D copied tree, mtimes+modes rewritten     sha256:$dd"
    if [ "$da" = "$db" ] && [ "$da" = "$dc" ] && [ "$da" = "$dd" ]; then
      ok "$name: four environments, one project_wheel_digest"
    else
      bad "$name: the wheel is NOT a function of the tree alone"
    fi

    # The digest the packer REPORTS is the digest of the bytes on disk.
    local reported
    reported=$(grep -o 'sha256:[0-9a-f]\{64\}' "$WORK/$name/a/.log" | head -1 | cut -d: -f2)
    [ "$reported" = "$da" ] && ok "$name: reported project_wheel_digest equals sha256sum of the file" \
                            || bad "$name: reported $reported, file is $da"

    # No absolute path, no host path, no timestamp drift inside the container.
    local whl; whl=$(ls "$WORK/$name"/a/*.whl)
    if python3 - "$whl" <<'PY'
import sys, zipfile
bad = []
with zipfile.ZipFile(sys.argv[1]) as z:
    for i in z.infolist():
        if i.filename.startswith('/') or '..' in i.filename.split('/') or '\\' in i.filename:
            bad.append(('path', i.filename))
        if i.date_time != (1980, 1, 1, 0, 0, 0):
            bad.append(('mtime', i.filename, i.date_time))
        if (i.external_attr >> 16) & 0o7777 != 0o644:
            bad.append(('mode', i.filename, oct((i.external_attr >> 16) & 0o7777)))
        if i.compress_type != zipfile.ZIP_STORED:
            bad.append(('method', i.filename))
print('\n'.join(map(str, bad)))
sys.exit(1 if bad else 0)
PY
    then ok "$name: every entry is relative, 1980-01-01, mode 0644, stored"
    else bad "$name: a container field is not fixed"
    fi
  done
}

# ------------------------------------------------------- a second, isolated seat

# The issue asks for two SEATS (build sandbox and dev box). The build sandbox does not
# exist yet, so the strongest honest stand-in available without a pod is a container:
# another root filesystem, another uid, another libc, another /tmp, another clock
# reading, no $HOME and no state of this box in reach. The binary is the same static
# CGO_ENABLED=0 build — what is being tested is the packer, not the compiler.
section_seat() {
  head_ "a second seat — the same tree packed inside a container, as another uid"
  command -v docker >/dev/null || { say "skipping: docker is not on PATH"; return; }

  for pair in "h3 h3 0.1.0" "sdxl sdxl 0.1.0"; do
    set -- $pair
    local tree="$ENDPOINTS/$1" name="$2" version="$3"
    [ -d "$tree" ] || { say "skipping $1 (no tree at $tree)"; continue; }
    pack_env "$name/host" "$WORK/seat/$name/host" "$tree" "$name" "$version" 022 UTC C "$HERE" "$WORK/seat/$name/home" || continue
    local host; host=$(digest_of "$WORK/seat/$name/host")

    for image in ubuntu:24.04 redis:7-alpine; do
      docker image inspect "$image" >/dev/null 2>&1 || { say "skipping $image (not present locally; this run pulls nothing)"; continue; }
      local dest="$WORK/seat/$name/$(echo "$image" | tr ':/' '__')"; mkdir -p "$dest"; chmod 0777 "$dest"
      if ! nice -n 19 docker run --rm --network none --user 1234:1234 --entrypoint /packer \
            -e TZ=Asia/Kathmandu -e LC_ALL=C.UTF-8 -e HOME=/nonexistent -w /srv \
            -v "$COZY":/packer:ro -v "$tree":/srv/tree:ro -v "$dest":/out \
            "$image" pack /srv/tree --name "$name" --version "$version" --out /out \
            >"$dest/.log" 2>&1; then
        bad "$name on $image: the packer did not run"; sed 's/^/       /' "$dest/.log"; continue
      fi
      local got; got=$(digest_of "$dest")
      say "$name on $image (uid 1234, TZ=Kathmandu, no HOME)  sha256:$got"
      [ "$got" = "$host" ] && ok "$name: container seat and host seat agree byte for byte" \
                           || bad "$name: the container seat produced a DIFFERENT wheel"
    done
    say "$name on this host                                sha256:$host"
  done
}

# ---------------------------------------------------------------- refusal arms

# arm <label> <expected code> <expected phrase in the refusal> <tree>
arm() {
  local label="$1" code="$2" phrase="$3" tree="$4"
  local out; out=$(nice -n 19 "$COZY" pack "$tree" --name planted --version 0.1.0 --out "$WORK/armout" 2>&1)
  local rc=$?
  if [ $rc -eq 0 ]; then bad "$label: PACKED — the door is open"; return; fi
  if ! printf '%s' "$out" | grep -q "error($code)"; then
    bad "$label: refused, but not as $code"; printf '%s\n' "$out" | sed 's/^/       /'; return
  fi
  if [ -n "$phrase" ] && ! printf '%s' "$out" | grep -qi "$phrase"; then
    bad "$label: refused as $code but never names: $phrase"; printf '%s\n' "$out" | sed 's/^/       /'; return
  fi
  ok "$label -> exit $rc, error($code)"
  printf '%s\n' "$out" | head -3 | sed 's/^/       | /'
}

section_arms() {
  head_ "typed refusals — planted trees, observed red"
  local P="$WORK/planted"

  # 1. a repo declaring a build backend that would have to be EXECUTED.
  mkdir -p "$P/backend"
  cat >"$P/backend/pyproject.toml" <<'TOML'
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[project]
name = "planted"
version = "0.1.0"
TOML
  printf 'app = None\n' >"$P/backend/planted.py"
  arm "custom [build-system] backend" build_backend_unsupported "future sandboxed class" "$P/backend"

  # 2. a project whose OWN wheel would need compiled extensions.
  mkdir -p "$P/compiled/planted"
  printf '__all__ = []\n' >"$P/compiled/planted/__init__.py"
  printf '#include <Python.h>\n' >"$P/compiled/planted/_fast.c"
  arm "compiled-extension source in a package dir" compiled_extension "future sandboxed class" "$P/compiled"

  # 2b. the same door from the other side: a prebuilt binary extension.
  mkdir -p "$P/binary/planted"
  printf '__all__ = []\n' >"$P/binary/planted/__init__.py"
  head -c 64 /dev/urandom >"$P/binary/planted/_fast.cpython-313-x86_64-linux-gnu.so"
  arm "prebuilt .so in a package dir" compiled_extension "py3-none-any" "$P/binary"

  # 3. a packaging step that wants to RUN project code.
  mkdir -p "$P/setuppy"
  printf 'from setuptools import setup\nsetup(name="planted")\n' >"$P/setuppy/setup.py"
  printf 'app = None\n' >"$P/setuppy/planted.py"
  arm "setup.py (packaging that executes project code)" project_code_execution "future sandboxed class" "$P/setuppy"

  # 4. metadata that is not metadata.
  mkdir -p "$P/malformed"
  cat >"$P/malformed/pyproject.toml" <<'TOML'
[project]
name = "planted"
version = 3
TOML
  printf 'app = None\n' >"$P/malformed/planted.py"
  arm "malformed [project] metadata" metadata_malformed "" "$P/malformed"

  # 4b. metadata a BACKEND would have had to compute.
  mkdir -p "$P/dynamic"
  cat >"$P/dynamic/pyproject.toml" <<'TOML'
[project]
name = "planted"
dynamic = ["version"]
TOML
  printf 'app = None\n' >"$P/dynamic/planted.py"
  arm "[project] dynamic = [\"version\"]" metadata_dynamic "" "$P/dynamic"

  # 5. a link is a second name for bytes that may not be in the tree at all.
  mkdir -p "$P/symlink"
  printf 'app = None\n' >"$P/symlink/planted.py"
  ln -s /etc/passwd "$P/symlink/secrets.txt"
  arm "symlink out of the tree" unsafe_entry "regular files only" "$P/symlink"

  # 6. an endpoint whose declared application module is not in the wheel: unservable by
  #    construction, because everything at serve time imports the INSTALLED wheel.
  mkdir -p "$P/absent"
  printf '[application]\nobject = "nowhere:app"\n' >"$P/absent/endpoint.toml"
  printf 'app = None\n' >"$P/absent/planted.py"
  arm "application module absent from the wheel" application_module_absent "installed wheel" "$P/absent"

  # THE GREEN that pairs with the red: with the planted files removed, the same trees pack.
  rm "$P/backend/pyproject.toml" "$P/compiled/planted/_fast.c" "$P/setuppy/setup.py"
  local green=0
  for t in backend compiled setuppy; do
    nice -n 19 "$COZY" pack "$P/$t" --name planted --version 0.1.0 --out "$WORK/armout/$t" >/dev/null 2>&1 && green=$((green+1))
  done
  [ $green -eq 3 ] && ok "the same three trees pack once the planted door is removed (3/3)" \
                   || bad "only $green/3 trees pack after removing the planted files"
}

# ---------------------------------------------------------------- real-tree install

section_install() {
  head_ "real trees — pack, pip install, and describe THROUGH the installed wheel"
  command -v uv >/dev/null || { bad "uv is not on PATH"; return; }
  for name in sdxl h3; do install_one "$name"; done
}

# One endpoint per venv, deliberately: the wheel carries the endpoint's own
# `endpoint.toml` and descriptor at the purelib root, so a venv answers for exactly one
# deployment — which is the serving shape (one executor per deployment, its own venv).
install_one() {
  local name="$1"
  local tree="$ENDPOINTS/$name"
  [ -d "$tree" ] || { bad "no $name tree at $tree"; return; }

  local out="$WORK/real/$name"; mkdir -p "$out"
  nice -n 19 "$COZY" pack "$tree" --name "$name" --version 0.1.0 --out "$out" >"$out/.log" 2>&1 \
    || { bad "packing the real $name tree"; sed 's/^/       /' "$out/.log"; return; }
  local whl; whl=$(ls "$out"/*.whl)
  ok "packed the real $name endpoint tree: $(basename "$whl")"
  say "$(grep project_wheel_digest "$out/.log")"

  local venv="$WORK/venv-$name"
  nice -n 19 uv venv --python 3.13 "$venv" >/dev/null 2>&1 || { bad "uv venv"; return; }
  local py="$venv/bin/python"

  # The endpoint's environment, exactly as the builder's checking container holds it:
  # the author surface plus msgspec, and NOTHING heavy. No torch, no diffusers, no
  # weights — describe derives the surface without loading a model.
  nice -n 19 uv pip install --python "$py" -q msgspec 'protobuf>=6.31' 'grpcio>=1.76' >/dev/null 2>&1 \
    || { bad "installing the light dependency set"; return; }
  if [ -d "$RUNTIME" ]; then
    nice -n 19 uv pip install --python "$py" -q --no-deps "$RUNTIME" >/dev/null 2>&1 \
      || { bad "installing the cozy-runtime checkout"; return; }
  else
    bad "no cozy-runtime checkout at $RUNTIME"; return
  fi

  # THE BAR: pip installs the packer's wheel.
  nice -n 19 uv pip install --python "$py" -q "$whl" >"$WORK/pipinstall-$name.log" 2>&1 \
    || { bad "pip install of the $name wheel"; sed 's/^/       /' "$WORK/pipinstall-$name.log"; return; }
  ok "$name: pip installed the wheel into a throwaway venv"

  local site; site=$("$py" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
  # The source tree is nowhere: this run imports what pip unpacked, and nothing else.
  if COZY_DIST="$name" COZY_SITE="$site" "$py" - <<'PY'
import importlib.metadata as md, importlib.util, os
dist, site = os.environ["COZY_DIST"], os.environ["COZY_SITE"]
d = md.distribution(dist)
print("       installed:", d.metadata["Name"], d.version, "| RECORD files:", len(list(d.files)))
spec = importlib.util.find_spec(dist)
print("       %s resolves to: %s" % (dist, spec.origin))
assert spec.origin.startswith(site), spec.origin
mod = importlib.import_module(dist)
print("       imported %s from the wheel; app = %s" % (dist, type(mod.app).__name__))
PY
  then ok "$name: the installed wheel imports, from site-packages, with no source tree on sys.path"
  else bad "$name: the installed wheel does not import"; fi

  # describe AGAINST THE INSTALLED WHEEL: `--dir` is site-packages, not the repo.
  if ( cd / && nice -n 19 "$venv/bin/cozy-runtime" describe --dir "$site" >"$WORK/describe-$name.log" 2>&1 ); then
    ok "$name: cozy-runtime describe ran against the installed wheel (--dir = site-packages)"
    head -12 "$WORK/describe-$name.log" | sed 's/^/       | /'
  else
    bad "$name: describe against the installed wheel"; sed 's/^/       /' "$WORK/describe-$name.log"
  fi

  # --check compares the derived surface against the descriptor the WHEEL carries.
  if ( cd / && nice -n 19 "$venv/bin/cozy-runtime" describe --dir "$site" --check >"$WORK/check-$name.log" 2>&1 ); then
    ok "$name: the wheel's own endpoint.descriptor.json matches the surface derived inside it"
  else
    bad "$name: describe --check against the installed wheel"; tail -6 "$WORK/check-$name.log" | sed 's/^/       /'
  fi
}

for s in "${SECTIONS[@]}"; do "section_$s"; done
printf '\n%d checks, %d failed\n' $((PASS+FAIL)) "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
