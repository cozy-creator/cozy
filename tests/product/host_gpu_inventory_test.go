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
// a compute capability) no longer discards the whole local GPU inventory. CUDA_VISIBLE_DEVICES
// names where the GPUs are: empty is none, and no driver is asked.
func TestLocalGPUInventoryKeepsEveryReadableDevice(t *testing.T) {
	path := t.TempDir()
	asked := filepath.Join(path, "asked")
	sshKeygen, err := exec.LookPath("ssh-keygen")
	must(t, err)
	must(t, os.Symlink(sshKeygen, filepath.Join(path, "ssh-keygen")))
	must(t, os.WriteFile(filepath.Join(path, "nvidia-smi"), []byte("#!/bin/sh\necho asked >> "+asked+"\n"+
		"if [ \"$#\" -eq 0 ]; then printf 'CUDA Version: 13.0\\n'; exit 0; fi\n"+
		"printf '%s\\n' '0, NVIDIA GeForce RTX 4090, 20000, 24564, 580.82.09, 8.9' "+
		"'1, NVIDIA Future Accelerator, 90000, 98304, 580.82.09, [N/A]' 'unreadable'\n"), 0o700))
	for visible, want := range map[string]int{"0,1": 2, "1": 1, "": 0} {
		root := t.TempDir()
		_ = os.Remove(asked)
		cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "up", "--json")
		cmd.Env = childEnv(t, root, "PATH="+path, "CUDA_VISIBLE_DEVICES="+visible)
		out, err := cmd.Output()
		t.Cleanup(func() { _, _ = runCozyPath(t, root, path, "down", "--json") })
		if err != nil {
			t.Fatalf("cozy up with CUDA_VISIBLE_DEVICES=%q: %v\n%s", visible, err, out)
		}
		var up struct {
			GPUs []string `json:"gpus"`
		}
		must(t, json.Unmarshal(out, &up))
		if len(up.GPUs) != want || strings.Contains(string(out), "sm_[") {
			t.Fatalf("CUDA_VISIBLE_DEVICES=%q listed %d GPUs, want %d: %s", visible, len(up.GPUs), want, out)
		}
		if _, err := os.Stat(asked); (err == nil) != (want > 0) {
			t.Fatalf("CUDA_VISIBLE_DEVICES=%q asked the driver: %v", visible, err == nil)
		}
	}
}
