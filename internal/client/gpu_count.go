package client

import "github.com/cozy-creator/cozy/internal/exit"

// An old controller could discard an unknown JSON field, so explicit counts need
// their own consumed-field capability. Automatic runs use the baseline contract.
func (c *Client) requireGPUCount(count uint32) *exit.Error {
	if count == 0 {
		return nil
	}
	caps, problem := c.Capabilities()
	if problem != nil {
		return problem
	}
	if !caps.RunGPUs {
		return exit.Named(exit.Unavailable, "daemon.gpu_count_unavailable",
			"this controller cannot retain an explicit GPU count; nothing was submitted").
			WithRemedy("update Cozy Creator before using --gpus")
	}
	return nil
}
