# Suite `stress-host` (gate.sh --cells, on a CPU rental): Klein and Qwen under the host RAM the rental's container
# allows, which must page or spill to disk instead of holding what does not fit (no host-RAM floor). The CLI caps no
# host memory (`cozy machine update` caps GPU memory only; `machine rent` picks no RAM), so the cap is the rental's own
# cgroup: host-cap.py reads its memory.max/high on the machine first. stress.py judges the cap and the page/spill bytes.
#
# Parked in known-failing.json (xs-058) until the varena host-tier fix: CPU mode SIGKILLed Klein 15370 and Qwen 15437
# at hitogami's 32 GB cgroup (released varena never pages; master sizes the host table from physical RAM). Until then
# STRESS_HOST_SKU stays empty, so no workflow rents for it. The proof, by hand on the fix:
#   gate.sh cpu --establish --only host-cap,host-klein,host-qwen --disk 160 --cells cells/stress-host.sh <candidate>
STRESS_HOST_SKU= STRESS_HOST_CELLS=host-cap,host-klein,host-qwen STRESS_HOST_DISK=160

ln -sfn "$HERE/cells/host-cap.py" host-cap.py
cell host-cap ./host-cap.py
cell host-klein "$(pkg flux-3-klein generate_turbo)" prompt="$CHAR, studio portrait" seed:=$((R + 151))
cell host-qwen "$(pkg qwen-image-2 generate_turbo)" prompt="$OBJECT on a windowsill above $SCENE, photo" seed:=$((R + 152))
