package client

import (
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
)

func (c *Client) PauseJob(id, actor string) (api.JobState, *exit.Error) {
	var state api.JobState
	problem := c.call("POST", "/v1/local/jobs/"+id+"/pause", map[string]string{"actor": actor}, &state)
	return state, problem
}

func (c *Client) ResumeJob(id, actor string) (api.JobState, *exit.Error) {
	var state api.JobState
	problem := c.call("POST", "/v1/local/jobs/"+id+"/resume", map[string]string{"actor": actor}, &state)
	return state, problem
}
