// cozy is the short-lived CLI and the private controller process entrypoint.
package main

import (
	"os"

	"github.com/cozy-creator/cozy-creator/internal/cli"
)

func main() {
	if cli.ControllerProcess(os.Args[0]) {
		os.Exit(cli.RunController(os.Stdout, os.Stderr))
	}
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
