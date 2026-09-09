package cli

import (
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

func requestedGPUCount(raw string) (int, *exit.Error) {
	if raw == "" {
		return 0, nil
	}
	count, err := strconv.ParseUint(raw, 10, 32)
	if err != nil || count == 0 {
		return 0, exit.Usagef("--gpus requires a positive integer GPU count")
	}
	return int(count), nil
}

// A width selects one existing Hub SKU. It never multiplies the price or adds
// provider quantity to the request. h100 consistently names the SXM 80 GB family.
func rentalGPUSKU(name, rawCount string) (string, *exit.Error) {
	count, problem := requestedGPUCount(rawCount)
	if problem != nil {
		return "", problem
	}
	if name == "" && count != 0 {
		return "", exit.Usagef("--gpus requires a GPU SKU, for example cozy rent h100 --gpus=4")
	}
	if name == "cpu" && count != 0 {
		return "", exit.Usagef("CPU rentals do not accept --gpus")
	}
	if split := strings.LastIndex(name, "-x"); split >= 0 {
		if width, err := strconv.Atoi(name[split+2:]); err == nil {
			if count != 0 && count != width {
				return "", exit.Usagef("SKU %s already specifies %d GPUs, conflicting with --gpus=%d", name, width, count)
			}
			return name, nil
		}
	}
	if name == "h100" {
		name = "h100-sxm5-80gb"
	}
	if count > 1 {
		name += "-x" + strconv.Itoa(count)
	}
	return name, nil
}
