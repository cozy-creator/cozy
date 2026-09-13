package orchestrator

import (
	"regexp"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

var privateRootRequestID = regexp.MustCompile(`^job-[0-9a-f]{24}$`)

// The bootstrap authority has already passed the Host's rental-owner signature
// check. Match its frozen canonical grant, never an author-supplied scope label.
// This check runs before any durable admission, including an idempotent replay.
func (c *Orchestrator) validateExecutionSubmission(s Submission) *exit.Error {
	if s.ExecutionGrantDigest == "" {
		if s.RequestID != "" {
			return exit.Named(exit.Credential, "execution_request_id_not_authorized", "ordinary submission cannot select its durable request ID")
		}
		return nil
	}
	if c.opt.PrivateExecution == nil {
		return exit.Named(exit.Credential, "execution_grant_not_authorized", "ordinary request admission cannot claim a private execution grant")
	}
	if s.Kind != "job" || !privateRootRequestID.MatchString(s.RequestID) {
		return exit.Named(exit.Validation, "execution_request_id_invalid", "local script requires its stable job request ID")
	}
	body, bodyErr := canonical.Raw(s.BodyDigest)
	spelledBody, _ := canonical.Spell(body)
	if bodyErr != nil || spelledBody != s.BodyDigest {
		return exit.Named(exit.Validation, "execution_capsule_required", "private root admission requires the exact validated capsule digest")
	}
	raw, err := canonical.Bytes(c.opt.PrivateExecution.Authorization.Grant)
	if err != nil {
		return exit.Internalf("private execution grant cannot be encoded: %s", err)
	}
	digest, err := canonical.Spell(canonical.Digest(raw))
	if err != nil {
		return exit.Internalf("private execution grant cannot be identified: %s", err)
	}
	if s.ExecutionGrantDigest != digest {
		return exit.Named(exit.Credential, "execution_grant_changed", "request does not name this coordinator's accepted execution grant")
	}
	return nil
}
