package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"

	"github.com/cozy-creator/cozy-creator-v2/internal/home"
)

// rowScript writes ONE artifact index row through the runtime's own writer, with `bytes`
// and `tensors` DERIVED BY THE RUNTIME'S OWN HEADER READER — the same derivation its own
// `pull` writes rows with (cl-028). The harness names WHICH artifact; every byte-plane
// fact comes from the resolver on the machine that holds the bytes. Nothing here parses
// a tensor header or spells a byte total, which is exactly what the fence's embed family
// now checks. `_header_facts` is the runtime's private spelling of that derivation; the
// tracker carries the ask to expose it as a public verb (artifacts.install deriving when
// bytes/tensors are omitted).
const rowScript = `
import json, sys
from pathlib import Path
from cozy_runtime.internal import artifacts, fill
from cozy_runtime.cli import pull

home, store_root, config, ref, lane, snaps_json = sys.argv[1:7]
snaps = json.loads(snaps_json)
tensorfs = fill.tensorfs_module()
store = tensorfs.Store.open(store_root)
total = tensors = 0
for sid in sorted(set(snaps.values())):
    facts = pull._header_facts(tensorfs, store, sid.split("sha256:")[-1])
    tensors += facts.tensors
    total += facts.stored_bytes
artifacts.install(Path(home), artifacts.Artifact(
    ref=ref, store=store_root, config=config, snapshots=snaps, lane=lane,
    bytes=total, tensors=tensors, variant="sm89"))
print(f"{ref}: {len(snaps)} components, {tensors} tensors, {total} B (runtime-derived, {lane})")
`

// installArtifactRow runs rowScript inside one generation's own venv — the runtime the
// release itself pinned — against the given store and component snapshots.
func installArtifactRow(root, generation, storeRoot, config, ref, lane string,
	snapshots map[string]string) {
	blob, err := json.Marshal(snapshots)
	must("rendering the component snapshots", err)
	python := home.VenvPython(filepath.Join(root, "generations", generation, "venv"))
	cmd := exec.Command("/usr/bin/nice", "-n", "19", python, "-c", rowScript,
		root, storeRoot, config, ref, lane, string(blob))
	cmd.Env = childEnv(root)
	data, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(data))
		must("installing the artifact index row", err)
	}
	fmt.Printf("  %s", string(data))
}
