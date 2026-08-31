//go:build !windows

package config

import (
	"fmt"
	"os"
	"syscall"
)

func validateProviderSecretFile(named, opened os.FileInfo) error {
	if named == nil || named.Mode()&os.ModeSymlink != 0 || !named.Mode().IsRegular() {
		return fmt.Errorf("provider credentials require a regular non-symlink config file")
	}
	if !os.SameFile(named, opened) {
		return fmt.Errorf("the config file changed while it was being opened")
	}
	if opened.Mode().Perm() != 0o600 {
		return fmt.Errorf("provider credentials require mode 0600, got %04o", opened.Mode().Perm())
	}
	stat, ok := opened.Sys().(*syscall.Stat_t)
	if !ok || stat.Uid != uint32(os.Geteuid()) {
		return fmt.Errorf("provider credentials require a config file owned by the current user")
	}
	return nil
}
