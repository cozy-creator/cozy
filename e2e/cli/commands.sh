# e2e/cli/commands.sh: the rest of cozy's PR-tier commands and flags (e2e/coverage.toml), sourced by cli/run.sh
# after its runs, on the same computer: signed out, CPU, the local machine up. Each call's answer is checked,
# not its exit code alone. Signed-in and renting commands are the release suite's (e2e/release/run.sh).
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
J() { jq -e "$@" > /dev/null; }  # J <filter> <file>: the answer holds what the filter asks

step status
cozy --json status > "$OUT/status.json" || check "cozy status failed"
J '.calcifer.running == true and (.calcifer.version | length > 0)' "$OUT/status.json" || check "status: Calcifer is not running"
J '.machines | any(.name == "local")' "$OUT/status.json" || check "status: the local machine is not listed"

step hubs
cozy --json hub > "$OUT/hub.json" && J '.hubs | any(.name == "tensorhub" and .current)' "$OUT/hub.json" || check "hub: tensorhub is not current"
cozy --json hub add e2e-alt https://tensorhub.com > /dev/null || check "hub add failed"
cozy --json hub list > "$OUT/hubs.json" && J '.hubs | any(.name == "e2e-alt")' "$OUT/hubs.json" || check "hub list: e2e-alt not added"
cozy --json hub use e2e-alt > /dev/null && cozy --json hub list | J '.hubs | any(.name == "e2e-alt" and .current)' || check "hub use: e2e-alt is not current"
# A command sent to a Hub by name, whichever is current.
cozy --tensorhub tensorhub --json package search birefnet --limit 1 > "$OUT/by-name.json" \
  && J '.packages[0].package == "birefnet"' "$OUT/by-name.json" || check "--tensorhub tensorhub: no birefnet"
cozy --json hub use tensorhub > /dev/null || check "hub use tensorhub failed"
cozy --json hub remove e2e-alt > /dev/null && ! cozy --json hub list | J '.hubs | any(.name == "e2e-alt")' || check "hub remove: e2e-alt is still known"

step "signed out"
cozy --json whoami > "$OUT/whoami.json" && J '.logged_in == false' "$OUT/whoami.json" || check "whoami: not signed out"
cozy --json logout > "$OUT/logout.json" || check "logout while signed out failed"

step packages
cozy --json package search birefnet --limit 1 > "$OUT/search.json" && J '(.packages | length) == 1 and .packages[0].package == "birefnet"' "$OUT/search.json" \
  || check "package search --limit 1: not exactly birefnet"
cozy --json package info birefnet > "$OUT/package-info.json" && J '(.latest_release | length > 0) and (.releases | length > 0)' "$OUT/package-info.json" \
  || check "package info birefnet: no releases"
cozy --json package install "$(birefnet_package)" --on local > "$OUT/install.ndjson" || check "package install $(birefnet_package) --on local failed: $(grep -oE '"message": *"[^"]*"' "$OUT/install.ndjson" | tail -1)"
cozy --json package list --on local > "$OUT/packages.json" && grep -q '"[^"]*birefnet"' "$OUT/packages.json" || check "package list --on local: no birefnet"
cozy --json package remove birefnet --on local > "$OUT/remove.ndjson" || check "package remove birefnet --on local failed"
cozy --json package list --on local > "$OUT/packages-after.json" && ! grep -q '"[^"]*birefnet"' "$OUT/packages-after.json" \
  || check "package remove: birefnet is still installed"

step models
cozy --json model search birefnet --limit 1 > "$OUT/models-search.json" && J '(.models | length) == 1' "$OUT/models-search.json" \
  || check "model search --limit 1: not one model"
family=$(cozy --json model search --limit 0 | jq -r '[.models[].family | select(. != null)][0] // empty') || true
if [ -n "$family" ]; then
  cozy --json model search --family "$family" > "$OUT/family.json" && J --arg f "$family" '(.models | length > 0) and all(.models[]; .family == $f)' "$OUT/family.json" \
    || check "model search --family $family: other families listed"
else
  cozy --json model search --family e2e-no-such-family > "$OUT/family.json" && J '(.models | length) == 0' "$OUT/family.json" \
    || check "model search --family: an unknown family listed models"
fi
cozy --json model info birefnet > "$OUT/model-info.json" && J '.releases[0].lanes | length > 0' "$OUT/model-info.json" || check "model info birefnet: no lanes"
variant=$(jq -r '.releases[0].lanes[0].lane' "$OUT/model-info.json") || true
cozy --json model download birefnet --lane 16 --on local > "$OUT/download.ndjson" || check "model download --lane 16 failed"
cozy --json model download birefnet --variant "$variant" local/e2e-birefnet > "$OUT/download-variant.ndjson" || check "model download --variant $variant failed"
# A local .safetensors file, made into a model the machine keeps by name: the file is written to the machine
# and made from its object (object://sha256:…/<name>).
uv run -q --no-project python -I - "$OUT/tiny.safetensors" <<'PY'
import json, struct, sys
header = json.dumps({"w": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]}}).encode()
header += b" " * (-len(header) % 8)
open(sys.argv[1], "wb").write(struct.pack("<Q", len(header)) + header + bytes(32))
PY
cozy --json model download "$OUT/tiny.safetensors" local/e2e-tiny > "$OUT/download-file.ndjson" || check "model download ./tiny.safetensors local/e2e-tiny failed: $(grep -oE '"message": *"[^"]*"' "$OUT/download-file.ndjson" | tail -1)"
cozy --json model list --on local > "$OUT/models.json" || check "model list --on local failed"
grep -q e2e-tiny "$OUT/models.json" || check "model list: the local file's local/e2e-tiny is not listed"
J 'any(.models[]; .model == "_/birefnet" and .precision == "fp16")' "$OUT/models.json" || check "model list: no fp16 _/birefnet after --lane 16"
grep -q e2e-birefnet "$OUT/models.json" || check "model list: the local/e2e-birefnet alias is not listed"

step "a YAML request, then inline inputs over it"
text=yaml-$RANDOM back=''
for ((i = ${#text} - 1; i >= 0; i--)); do back+=${text:i:1}; done
printf 'text: %s\n' "$text" > "$OUT/reverse.yaml"
if (cd "$here" && cell yaml ./weightless/reverse --input "$OUT/reverse.yaml"); then
  grep -q "$back" "$OUT/yaml.ndjson" || check "yaml: the reply is not $back"
fi
if (cd "$here" && cell yaml-over ./weightless/reverse --input "$OUT/reverse.yaml" text=inline); then
  grep -q enilni "$OUT/yaml-over.ndjson" || check "yaml-over: the inline text did not override the file's"
fi
dog=$(picture dog1.jpg)
printf 'image: %s\n' "$dog" > "$OUT/birefnet.yaml"
if cell birefnet-yaml "$(birefnet)" --input "$OUT/birefnet.yaml" --lane 16; then
  white_background "$(output birefnet-yaml webp)" "$dog" || check "birefnet-yaml: not a background removal"
  J '[.. | .precision? // empty] | any(. == "fp16" or . == "bf16")' "$OUT/birefnet-yaml.json" || check "birefnet-yaml --lane 16: not a 16-bit lane"
fi

step "run flags"
before=$(cozy --json run list --limit 0 | jq '.runs | length') || true
(cd "$here" && cozy --json run ./weightless/reverse --describe) > "$OUT/describe.json" || check "run --describe failed"
J '.fields | any(.name == "text")' "$OUT/describe.json" || check "run --describe: no text field"
[ "$(cozy --json run list --limit 0 | jq '.runs | length')" = "$before" ] || check "run --describe started a run"
note=$OUT/inputs/note.txt; echo "an asset" > "$note"
if (cd "$here" && cell fidelity ./weightless/fidelities --asset "notes=$note" --asset-fidelity notes=high); then
  grep -q 'notes=high' "$OUT/fidelity.ndjson" || check "--asset-fidelity notes=high: the executor saw $(grep -o '"assets":\[[^]]*\]' "$OUT/fidelity.ndjson")"
fi
for level in installed downloaded imported host; do
  cozy --json run "$(birefnet)" --warm "$level" --on local > "$OUT/warm-$level.ndjson" || check "run --warm $level failed"
  cozy --json machine show local > "$OUT/warm-$level.json" || true
  # The warm set holds the level asked, or says why it holds less (a CPU binds no model on a GPU).
  J --arg l "$level" '.status.warm | any((.source | test("birefnet")) and .level == $l and (.holds == $l or (.held_back | length > 0)))' "$OUT/warm-$level.json" \
    || check "--warm $level: the warm set does not hold birefnet at $level: $(jq -c '.status.warm' "$OUT/warm-$level.json")"
done
cozy --json run "$(birefnet)" --warm off --on local > "$OUT/warm-off.ndjson" || check "run --warm off failed"
cozy --json machine show local | J '.status.warm | all(.source | test("birefnet") | not)' || check "--warm off: birefnet is still in the warm set"

step "a run followed, a snapshot, a cancel"
(cd "$here" && cozy --json run ./weightless/wait seconds:=120 --on local > "$OUT/wait.ndjson" 2>&1) &
waiting=$!
n=''
for _ in $(seq 60); do n=$(grep -oE '"number": *[0-9]+' "$OUT/wait.ndjson" | head -1 | grep -oE '[0-9]+$') && [ -n "$n" ] && break; sleep 1; done
if [ -z "$n" ]; then check "the waiting run never got a number"; else
  for _ in $(seq 60); do cozy --json run show "$n" --once > "$OUT/snapshot.json" || true; J '.status == "in_progress"' "$OUT/snapshot.json" && break; sleep 2; done
  J '.status == "in_progress"' "$OUT/snapshot.json" || check "run show --once: run $n is $(jq -r .status "$OUT/snapshot.json"), not in_progress"
  cozy --json run show "$n" > "$OUT/followed.ndjson" &
  following=$!
  sleep 3
  kill -0 "$following" 2> /dev/null || check "run show $n stopped before the run settled (it should follow it)"
  cozy --json run cancel "$n" > "$OUT/cancel.json" || check "run cancel $n failed"
  wait "$following" || true
  wait "$waiting" || true
  grep -qE '"status": *"canceled"' "$OUT/followed.ndjson" || check "run show $n did not follow it to canceled"
  cozy --json run show "$n" --once | J '.status == "canceled"' || check "run cancel: run $n is not canceled"
fi

step "run list filters"
cozy --json run list --state canceled > "$OUT/canceled.json" && J --argjson n "${n:-0}" '(.runs | length > 0) and all(.runs[]; .status == "canceled") and any(.runs[]; .number == $n)' "$OUT/canceled.json" \
  || check "run list --state canceled: wrong runs"
cozy --json run list --package birefnet > "$OUT/by-package.json" && J '(.runs | length > 0) and all(.runs[]; .target | test("birefnet"))' "$OUT/by-package.json" \
  || check "run list --package birefnet: other packages listed"
cozy --json run list --on local > "$OUT/by-machine.json" && J '(.runs | length > 0) and all(.runs[]; .machine == "local")' "$OUT/by-machine.json" \
  || check "run list --on local: other machines listed"
cozy --json run list --limit 1 > "$OUT/limit.json" && J '(.runs | length) == 1' "$OUT/limit.json" || check "run list --limit 1: not one run"

step "run play"
# The link opens the browser player (GitHub Pages) with this machine, the output and a token in its fragment; the
# token's expiry is --expires from now.
first=$(cat "$OUT/first.number")
name=$(jq -r '.output[0].name // empty' "$OUT/first.json") || true
expiry() {  # the token's expiry (unix s), from the link's fragment (unpadded base64url JSON before its `.`)
  local t; t=$(sed -n 's/.*[#&]c=\([^.&]*\).*/\1/p' <<< "$1" | tr '_-' '/+')
  while (( ${#t} % 4 )); do t+==; done
  base64 -d <<< "$t" 2> /dev/null | jq -r '.e // empty' 2> /dev/null || true
}
if [ "$first" = none ] || [ -z "$name" ]; then check "run play: no BiRefNet output to play"; else
  for expires in 30s 2h; do
    url=$(cozy --json run play "$first" --output "$name" --expires "$expires" | tee "$OUT/play-$expires.json" | jq -r '.link // empty') || true
    curl -fsS -o /dev/null "${url%%#*}" || check "run play: its player page ${url%%#*} does not answer"
    [[ $url == *"o=$name"* && $url == *"a="* ]] || check "run play: the link names no machine or not output $name: $url"
    want=$(( $(date +%s) + ${expires%[sh]} * $([ "${expires: -1}" = h ] && echo 3600 || echo 1) )) e=$(expiry "$url")
    [ -n "$e" ] && [ $(( e - want )) -le 10 ] && [ $(( want - e )) -le 10 ] || check "run play --expires $expires: the token expires at ${e:-?}, not about $want"
  done
fi

step machines
# The candidate CLI's own spelling of an update (candidates() spoke the installer's): the same wheels again,
# and the machine reports their versions.
if [ -n "$CAND_RT" ] && [ -n "$CAND_TFS" ]; then
  cozy --json machine update local --runtime "$CAND_RT" --tensorfs "$CAND_TFS" -y > "$OUT/update-again.json" || check "machine update local --runtime --tensorfs failed"
  rt=$(basename "$CAND_RT" | cut -d- -f2) tfs=$(basename "$CAND_TFS" | cut -d- -f2)
  cozy --json machine show local | J --arg rt "$rt" --arg tfs "$tfs" '(.status.runtime | startswith($rt)) and (.status.tensorfs | startswith($tfs))' \
    || check "machine update --runtime $rt --tensorfs $tfs: the machine reports other versions"
fi
cozy --json machine list > "$OUT/machines.json" && J '.machines | any(.name == "local")' "$OUT/machines.json" || check "machine list: no local"
cozy --json machine show local > "$OUT/local.json" && J '.state == "ready" and (.address | length > 0)' "$OUT/local.json" || check "machine show local: not ready"

step "machine logs"
cozy machine logs local --lines 5 > "$OUT/log-5.txt" || check "machine logs failed"
[ "$(wc -l < "$OUT/log-5.txt")" -le 5 ] && [ -s "$OUT/log-5.txt" ] || check "machine logs --lines 5: $(wc -l < "$OUT/log-5.txt") lines"
cozy machine logs local --since 1h > "$OUT/log-since.txt" && [ -s "$OUT/log-since.txt" ] || check "machine logs --since 1h: nothing from the last hour"
cozy machine logs local calcifer --lines 5 > "$OUT/log-calcifer.txt" && [ -s "$OUT/log-calcifer.txt" ] || check "machine logs local calcifer: no Calcifer log"
code=0; timeout 8 "$COZY" machine logs local --follow > "$OUT/log-follow.txt" 2>&1 || code=$?
printf 'machine\x1flogs\x1flocal\x1f--follow\x1f\x1e' >> "$OUT/argv.log"  # timeout runs the binary itself
[ "$code" = 124 ] && [ -s "$OUT/log-follow.txt" ] || check "machine logs --follow: exited $code instead of following"

step "an added machine"
addr=$(jq -r .address "$OUT/local.json") || true; fp=$(jq -r .fingerprint "$OUT/local.json") || true
cozy --json machine add e2e-self "$addr" --fingerprint "$fp" > "$OUT/added.json" || check "machine add e2e-self $addr failed"
cozy --json machine list | J '.machines | any(.name == "e2e-self")' || check "machine add: e2e-self is not listed"
if (cd "$here" && cozy --json run ./weightless/reverse text=added --on e2e-self > "$OUT/on-added.ndjson"); then
  grep -q dedda "$OUT/on-added.ndjson" || check "a run on e2e-self: no reply"
else check "a run on the added machine failed"; fi
openssl genpkey -algorithm ed25519 -out "$OUT/controller.pem" 2> /dev/null || check "openssl could not make an Ed25519 key"
key=$(openssl pkey -in "$OUT/controller.pem" -pubout -outform DER | tail -c 32 | base64 | tr '+/' '-_' | tr -d '=\n') || true
cozy --json machine authorize "$key" > "$OUT/authorize.json" && J '.changed == true' "$OUT/authorize.json" || check "machine authorize: the key was not added"
grep -qx "$key" "$HOME/.cozy/keys/authorized" || check "machine authorize: the key is not in keys/authorized"
cozy --json machine authorize "$key" | J '.changed == false' || check "machine authorize twice: added it again"
cozy --json machine end e2e-self > "$OUT/end-added.json" || check "machine end e2e-self failed"
! cozy --json machine list | J '.machines | any(.name == "e2e-self")' || check "machine end: e2e-self is still listed"

step "doctor --on"
cozy --json doctor --on local > "$OUT/doctor-local.json" || check "cozy doctor --on local found a problem"

step completion
shells=(bash)
[ "$RUNNER_OS" != Linux ] || { sudo apt-get install -y -qq zsh fish > /dev/null 2>&1 || true; }
command -v zsh > /dev/null && shells+=(zsh)
command -v fish > /dev/null && shells+=(fish)
command -v pwsh > /dev/null && shells+=(powershell)
for sh in "${shells[@]}"; do cozy completion "$sh" > "$OUT/completion.$sh" && [ -s "$OUT/completion.$sh" ] || check "completion $sh: no script"; done
[ ! -s "$OUT/completion.powershell" ] || cp "$OUT/completion.powershell" "$OUT/completion.ps1"  # pwsh dot-sources .ps1 only
bash -c ". '$OUT/completion.bash'; complete -p cozy" > /dev/null || check "completion bash: does not complete cozy"
[ ! -s "$OUT/completion.zsh" ] || zsh -c "autoload -U compinit; compinit -u; . '$OUT/completion.zsh'; (( \$+functions[_cozy] ))" \
  || check "completion zsh: no _cozy"
[ ! -s "$OUT/completion.fish" ] || fish -c "source '$OUT/completion.fish'; complete -C 'cozy mach'" | grep -q machine \
  || check "completion fish: cozy mach does not complete to machine"
[ ! -s "$OUT/completion.powershell" ] || PATH="$(dirname "$COZY"):$PATH" pwsh -NoProfile -c ". '$OUT/completion.ps1'; (TabExpansion2 'cozy mach' 9).CompletionMatches.CompletionText" | grep -q machine \
  || check "completion powershell: cozy mach does not complete to machine"
echo "completion: ${shells[*]}"

# Last: it replaces this computer's cozy (the candidate) with releases.
step update
cp "$COZY" "$OUT/cozy-under-test"
releases=$(curl -fsSL ${GH_TOKEN:+-H "Authorization: Bearer $GH_TOKEN"} "https://api.github.com/repos/cozy-creator/cozy-creator/releases?per_page=30" \
  | jq -r '[.[] | select(.prerelease | not) | .tag_name | select(test("^v[0-9]+\\.[0-9]+\\.[0-9]+$"))] | .[:2] | .[]') || true
newest=$(sed -n 1p <<< "$releases") previous=$(sed -n 2p <<< "$releases")
if [ -z "$previous" ]; then check "update: no two releases to move between"; else
  t0=$SECONDS
  cozy --json update "${previous#v}" > "$OUT/update-previous.json" || check "update ${previous#v} failed"
  echo "update ${previous#v}: $((SECONDS - t0)) s"; t0=$SECONDS
  [ "$(cozy --version)" = "cozy ${previous#v}" ] || check "update ${previous#v}: cozy is $(cozy --version)"
  cozy --json update > "$OUT/update-newest.json" || check "update to the newest failed"
  echo "update ${newest#v}: $((SECONDS - t0)) s"
  [ "$(cozy --version)" = "cozy ${newest#v}" ] || check "update: cozy is $(cozy --version), not the newest ${newest#v}"
  cozy --json status | J --arg v "${newest#v}" '.calcifer.version == $v' || check "update: Calcifer did not restart on ${newest#v}"
fi
