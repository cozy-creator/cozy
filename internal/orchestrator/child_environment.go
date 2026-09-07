package orchestrator

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/canonical"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// qualifyChildReuse measures the current privileged execution environment before
// making a cache decision. A missing measurement permits ordinary execution but
// never reuses numerical work under an unqualified Runtime/framework/settings.
func (c *Orchestrator) qualifyChildReuse(s *session, call *pb.ChildCallRequest, spec *Submission, target string) string {
	if !spec.ChildReusable {
		return target
	}
	var result *pb.NumericalEnvironmentResult
	var err error
	if s.host != nil {
		result, err = s.host.NumericalEnvironment(s.ctx, &pb.NumericalEnvironmentCall{Claim: s.claim})
	} else if s.preparation != nil {
		result, err = s.preparation.NumericalEnvironment(s.ctx, &pb.NumericalEnvironmentRequest{})
	}
	if err != nil || result == nil || len(result.Digest) != 32 {
		spec.ChildReusable = false
		c.logf("child call %s/%d executes without cross-run reuse: numerical environment is unqualified", call.ParentRequestId, call.CallIndex)
		return target
	}
	environment, _ := canonical.Spell(result.Digest)
	raw, _ := json.Marshal(map[string]string{"target_digest": target, "numerical_environment_digest": environment})
	raw, _ = canonical.NormalizeJCS(raw)
	qualified, _ := canonical.Spell(canonical.Digest(raw))
	return qualified
}
