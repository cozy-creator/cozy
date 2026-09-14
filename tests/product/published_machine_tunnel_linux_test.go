package producttest

import (
	"os/exec"
	"syscall"
	"testing"
)

func bindPublishedTunnelParent(_ *testing.T, command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
}
