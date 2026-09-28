// cozy is the short-lived CLI and the private daemon process entrypoint.
package main

import (
	"os"

	"github.com/cozy-creator/cozy/internal/cli"
)

func main() {
	if cli.DaemonProcess(os.Args[0]) {
		os.Exit(cli.RunDaemon(os.Stderr))
	}
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
