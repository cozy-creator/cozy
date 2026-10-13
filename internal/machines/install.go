package machines

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

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
