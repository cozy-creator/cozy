package cli

import (
	"strconv"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
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

// runBits is --bits: the run's weight precision, 16, 8 or 4 bits; absent, its machine's rules
// pick each slot's lane (xs-042).
func runBits(ctx *Context) (uint32, *exit.Error) {
	raw := ctx.Inv.Value("--bits")
	if raw == "" {
		return 0, nil
	}
	bits, err := strconv.ParseUint(raw, 10, 32)
	if err != nil || bits == 0 || launch.ValidateBits(uint32(bits)) != nil {
		return 0, exit.Named(exit.Usage, "bits_invalid", "--bits=%s is not 16, 8 or 4", raw)
	}
	return uint32(bits), nil
}
