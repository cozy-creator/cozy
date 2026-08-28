// cozy — the cozy-creator CLI. Everything lives behind the manifest in internal/app.
package main

import (
	"os"

	"github.com/cozy-creator/cozy-creator/internal/app"
)

func main() {
	os.Exit(app.Run(os.Args[1:], os.Stdout, os.Stderr))
}
