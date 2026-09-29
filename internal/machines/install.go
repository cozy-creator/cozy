package machines

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/hostruntime"
)

func installMachinePair(ctx context.Context, uv, python string, requirements []string) error {
	// Repair the selected distributions even if their versions did not change. Do not
	// sync the environment or reinstall unrelated framework packages.
	args := []string{"pip", "install", "--no-config", "--python", python,
		"--reinstall-package", hostruntime.Distribution, "--reinstall-package", "tensorfs"}
	if output, err := exec.CommandContext(ctx, uv, append(args, requirements...)...).CombinedOutput(); err != nil {
		return fmt.Errorf("cannot install the Runtime and TensorFS: %s", tail(output))
	}
	for _, name := range []string{"cozy-runtime-worker", "tfs"} {
		if _, err := os.Stat(filepath.Join(filepath.Dir(python), name)); err != nil {
			return fmt.Errorf("the installed machine has no %s: %w", name, err)
		}
	}
	return nil
}

// restorePair restores the previous Runtime and TensorFS after a failed install.
// Dependency resolution can change shared dependencies; this restores the selected
// pair, not a snapshot of the entire Python environment.
func (h *Host) restorePair(ctx context.Context, uv string, previous *Installed) error {
	var requirements []string
	for _, pair := range []struct {
		distribution string
		artifact     installedArtifact
	}{{hostruntime.Distribution, previous.Runtime}, {"tensorfs", previous.TensorFS}} {
		name := pair.distribution
		if name == hostruntime.Distribution {
			name += "[media]"
		}
		if strings.HasSuffix(pair.artifact.Name, ".whl") {
			path := filepath.Join(h.wheels(), filepath.Base(pair.artifact.Name))
			digest, err := fileDigest(path)
			if err != nil || digest != pair.artifact.SHA256 {
				return fmt.Errorf("the previous %s wheel is unavailable or changed", pair.distribution)
			}
			requirements = append(requirements, name+" @ file://"+path)
		} else if version, ok := strings.CutPrefix(pair.artifact.Name, pair.distribution+" "); ok && version != "" {
			requirements = append(requirements, name+"=="+version)
		} else {
			return fmt.Errorf("the previous %s version is unknown", pair.distribution)
		}
	}
	return installMachinePair(ctx, uv, filepath.Join(h.python(), "bin/python"), requirements)
}
