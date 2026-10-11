package cli

import (
	"strconv"

	"github.com/cozy-creator/cozy/internal/exit"
)

func gpuCountArgument(count *uint32) (string, *exit.Error) {
	if count == nil {
		return "", nil
	}
	if *count == 0 {
		return "", exit.Usagef("--gpus must be a positive integer; omit it for automatic parallelism")
	}
	return strconv.FormatUint(uint64(*count), 10), nil
}

func runGPUCount(ctx *Context) (uint32, *exit.Error) {
	raw := ctx.Inv.Value("--gpus")
	if raw == "" {
		return 0, nil
	}
	count, err := strconv.ParseUint(raw, 10, 32)
	if err != nil || count == 0 {
		return 0, exit.Usagef("--gpus must be a positive integer; omit it for automatic parallelism")
	}
	return uint32(count), nil
}
