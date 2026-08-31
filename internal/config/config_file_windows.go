//go:build windows

package config

import (
	"fmt"
	"os"
)

func validateProviderSecretFile(named, opened os.FileInfo) error {
	if named == nil || named.Mode()&os.ModeSymlink != 0 || !named.Mode().IsRegular() {
		return fmt.Errorf("provider credentials require a regular non-symlink config file")
	}
	if !os.SameFile(named, opened) {
		return fmt.Errorf("the config file changed while it was being opened")
	}
	return nil
}
