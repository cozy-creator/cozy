package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// One nvidia-smi row this build cannot read (a newer driver's spelling, a device without
// a compute capability) no longer discards the whole local GPU inventory.
func TestLocalGPUInventoryKeepsEveryReadableDevice(t *testing.T) {
	root, path := t.TempDir(), t.TempDir()
	sshKeygen, err := exec.LookPath("ssh-keygen")
	must(t, err)
	must(t, os.Symlink(sshKeygen, filepath.Join(path, "ssh-keygen")))
	must(t, os.WriteFile(filepath.Join(path, "nvidia-smi"), []byte("#!/bin/sh\n"+
		"if [ \"$#\" -eq 0 ]; then printf 'CUDA Version: 13.0\\n'; exit 0; fi\n"+
		"printf '%s\\n' '0, NVIDIA GeForce RTX 4090, 20000, 24564, 580.82.09, 8.9' "+
		"'1, NVIDIA Future Accelerator, 90000, 98304, 580.82.09, [N/A]' 'unreadable'\n"), 0o700))
	t.Cleanup(func() { _, _ = runCozyPath(t, root, path, "down", "--json") })
	code, out := runCozyPath(t, root, path, "up", "--json")
	if code != 0 {
		t.Fatalf("cozy up [exit %d]: %s", code, out)
	}
	var up struct {
		GPUs []string `json:"gpus"`
	}
	must(t, json.Unmarshal([]byte(out), &up))
	if len(up.GPUs) != 2 || strings.Contains(out, "sm_[") {
		t.Fatalf("readable GPUs were discarded with an unreadable row: %s", out)
	}
}
