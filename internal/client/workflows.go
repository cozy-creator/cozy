package client

import (
	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

func (c *Client) SubmitWorkflow(sub api.WorkflowSubmission,
	key string) (api.WorkflowHandle, *exit.Error) {
	var handle api.WorkflowHandle
	problem := c.call("POST", "/v1/local/workflows", sub, &handle, "Idempotency-Key", key)
	return handle, problem
}

func (c *Client) Workflow(id string) (api.WorkflowState, *exit.Error) {
	var state api.WorkflowState
	problem := c.call("GET", "/v1/local/workflows/"+id, nil, &state)
	return state, problem
}

func (c *Client) CancelWorkflow(id string) (api.WorkflowState, *exit.Error) {
	var state api.WorkflowState
	problem := c.call("POST", "/v1/local/workflows/"+id+"/cancel", nil, &state)
	return state, problem
}
