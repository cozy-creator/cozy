package hub

import (
	"bytes"
	"context"
	"github.com/cozy-creator/cozy/internal/exit"
	"net/http"
	"net/url"
)

// AttachAssessment stores and reads back the exact canonical report. The caller
// verifies its invocation/receipt bindings first; Hub records a publisher assessment.
func (c *Client) AttachAssessment(ctx context.Context, ref Ref, checkpoint, digest, expectedVerdict string, raw []byte) *exit.Error {
	path := resourcePath("models", ref) + "/checkpoints/" + url.PathEscape(checkpoint) + "/assessments/" + url.PathEscape(digest)
	var out struct {
		Assessment struct {
			Checkpoint string `json:"checkpoint_id"`
			Scope      string `json:"scope"`
			Report     struct {
				Digest string `json:"digest"`
				Length int64  `json:"length"`
			} `json:"report"`
			Actor   string `json:"actor"`
			Created string `json:"created_at"`
		} `json:"assessment"`
		Verdict string `json:"publisher_reported_verdict"`
	}
	if problem := c.do(ctx, call{method: http.MethodPut, path: path, auth: true, reason: "attach verified producer evidence", bodyBytes: raw, strict: true}, &out); problem != nil {
		return problem
	}
	if out.Assessment.Checkpoint != checkpoint || out.Assessment.Scope != "publisher_assessment" || out.Assessment.Report.Digest != digest || out.Assessment.Report.Length != int64(len(raw)) || out.Verdict != expectedVerdict {
		return exit.Internalf("checkpoint assessment readback changed its identity or scope")
	}
	var received []byte
	if problem := c.do(ctx, call{method: http.MethodGet, path: path, auth: true, raw: &received, responseBytes: 8 << 20}, nil); problem != nil {
		return problem
	}
	if !bytes.Equal(received, raw) {
		return exit.Internalf("checkpoint assessment bytes differ after attachment")
	}
	return nil
}
