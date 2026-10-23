#!/usr/bin/env bash
# The `mps` suite: the local path on an Apple GPU as a person takes it, on GitHub's macos-15 runner. GitHub's
# macOS VMs give torch 2.14 a working GPU on macOS 15 only (on 26+ its Metal 4 kernels compute wrong there,
# pytorch#196582; torch 2.15+ refuses the virtual GPU). BiRefNet must serve on the GPU; SDXL must serve or,
# past the GPU's working set, be refused naming it (memory_exhausted).
#   e2e/mps/run.sh --out DIR [--cozy-bin F] [--tensorfs-wheel F] [--runtime-wheel F] [--packages DIR]
. "$(dirname "$0")/../lib.sh"
parse "$@"
[ "$(uname -s)" = Darwin ] || fail "the mps suite runs on macOS"

step install; install_cozy
step "the local machine"; cozy --json up; cozy --json machine enable; machine_ready
step candidates; candidates
sysctl -n machdep.cpu.brand_string hw.memsize; sw_vers -productVersion
# An Apple GPU, unified.
grep -q '"family": *"apple' "$OUT/machine.json" && grep -q '"memory_class": *"unified"' "$OUT/machine.json" \
  || fail "the local machine reports no unified Apple GPU"

step "BiRefNet on the Apple GPU"
dog=$(picture dog1.jpg)
cell birefnet "$(birefnet)" --asset "image=$dog"
white_background "$(output birefnet webp)" "$dog"

step "SDXL on the Apple GPU, or refused past its working set"
sdxl=sdxl/generate
[ -z "$PACKAGES" ] || sdxl=./$PACKAGES/sdxl/generate
ALLOW=memory_exhausted cell sdxl "$sdxl" prompt="a lighthouse on a basalt cliff at dusk, oil painting" seed:=4021 steps:=12
[ -e "$OUT/sdxl.refused" ] || [ -n "$(output sdxl png)$(output sdxl webp)$(output sdxl jpg)" ] || fail "sdxl completed with no image"

step logs; cozy machine logs local > "$OUT/machine.log" 2>&1 || true
echo "::endgroup::"
echo "mps: birefnet run $(cat "$OUT/birefnet.number"), sdxl run $(cat "$OUT/sdxl.number")$([ -e "$OUT/sdxl.refused" ] && echo ' (refused past the working set)')"
