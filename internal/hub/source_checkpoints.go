package hub

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// VerifyPublicationObjects confirms storage custody without finalizing a Manifest.
func (c *Client) VerifyPublicationObjects(ctx context.Context, ref Ref, operation string,
	objectIDs []string,
) (Session, *exit.Error) {
	var out struct {
		Publication Session `json:"publication"`
	}
	problem := c.do(ctx, call{method: http.MethodPost,
		path: publications(ref) + "/" + url.PathEscape(operation) + "/verify",
		auth: true, reason: "retain source preparation progress", byBytes: true, patient: true, strict: true,
		body: map[string]any{"object_ids": objectIDs}}, &out)
	return out.Publication, problem
}

type PublicationReads struct {
	Reads          []Read `json:"reads"`
	ServerTimeUnix int64  `json:"server_time_unix"`
}

func (c *Client) ReadPublicationObjects(ctx context.Context, ref Ref, operation string,
	objectIDs []string,
) (PublicationReads, *exit.Error) {
	var out PublicationReads
	problem := c.do(ctx, call{method: http.MethodPost,
		path: publications(ref) + "/" + url.PathEscape(operation) + "/reads",
		auth: true, reason: "restore source preparation progress", byBytes: true, patient: true, strict: true,
		body: map[string]any{"object_ids": objectIDs}}, &out)
	return out, problem
}

func (c *Client) AbandonPublication(ctx context.Context, ref Ref, operation string) *exit.Error {
	var out struct {
		Publication Session `json:"publication"`
	}
	problem := c.do(ctx, call{method: http.MethodDelete,
		path: publications(ref) + "/" + url.PathEscape(operation), auth: true,
		reason: "release finished source preparation holds", byBytes: true, patient: true, strict: true}, &out)
	if problem != nil {
		return problem
	}
	if out.Publication.Operation != operation || out.Publication.State != "abandoned" {
		return exit.Named(exit.Conflict, "model_transfer.source_publication_not_abandoned",
			"Tensorhub did not acknowledge release of source preparation custody")
	}
	return nil
}
