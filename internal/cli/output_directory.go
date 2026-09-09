package cli

import (
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
)

func requestedOutputDirectory(ctx *Context) (string, *exit.Error) {
	requested := ctx.Inv.Value("--out")
	if requested == "" {
		return "", nil
	}
	absolute, err := filepath.Abs(requested)
	if err != nil {
		return "", exit.Usagef("cannot resolve --out %q: %s", requested, err)
	}
	return filepath.Clean(absolute), nil
}
