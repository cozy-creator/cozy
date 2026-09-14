//go:build !linux

package producttest

import (
	"os/exec"
	"testing"
)

func bindPublishedTunnelParent(t *testing.T, _ *exec.Cmd) {
	t.Fatal("public fixture transport requires parent-death cleanup on Linux")
}
