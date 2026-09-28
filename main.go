// cozy is the short-lived CLI, the private owner daemon, and a machine's daemon.
package main

import (
	"os"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/host"
)

func main() {
	switch {
	case cli.DaemonProcess(os.Args[0]):
		os.Exit(cli.RunDaemon(os.Stderr))
	case host.GuardianProcess(os.Args[0]):
		os.Exit(host.RunGuardian(os.Args))
	case host.Invoked(os.Args):
		os.Exit(host.Main(os.Stderr))
	}
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
