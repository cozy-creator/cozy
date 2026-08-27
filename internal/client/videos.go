package client

import (
	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/video"
)

func (c *Client) ComposeVideo(request api.VideoComposeRequest) (video.Composition, *exit.Error) {
	var composition video.Composition
	problem := c.call("POST", "/v1/local/video-compositions", request, &composition)
	return composition, problem
}
