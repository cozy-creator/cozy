package client

import (
	"net/http"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func (c *Client) requireModelOverrides(pkg, function string, models []records.ModelRef) *exit.Error {
	request := records.Request{Package: pkg, Entrypoint: function, Models: models}
	if !request.RequiresModelOverrides() {
		return nil
	}
	var capabilities api.Capabilities
	problem := c.call(http.MethodGet, "/v1/capabilities", nil, &capabilities)
	if problem != nil && problem.Code != exit.NotFound {
		return problem
	}
	if !capabilities.ModelOverrides {
		return exit.Named(exit.Unavailable, "daemon.model_overrides_unavailable",
			"the running Cozy daemon cannot retain these model overrides; nothing was submitted").
			WithRemedy("update cozy and restart the daemon with `cozy down` followed by `cozy up`; existing machine work remains running")
	}
	return nil
}
