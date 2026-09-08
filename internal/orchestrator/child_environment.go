package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// This session belongs to the selected callee, after its preparation completed.
// Its numerical identity is independent of the script's CPU control process.
func (c *Orchestrator) qualifyOperation(s *session, request records.Request) (*records.OperationContext, *exit.Error) {
	environment, problem := numericalEnvironment(s)
	if problem != nil {
		return nil, problem
	}
	return c.opt.Store.BindOperationContext(request.ID, environment)
}

func numericalEnvironment(s *session) (string, *exit.Error) {
	var result *pb.NumericalEnvironmentResult
	var err error
	if s.host != nil {
		result, err = s.host.NumericalEnvironment(s.ctx, &pb.NumericalEnvironmentCall{Claim: s.claim})
	} else if s.preparation != nil {
		result, err = s.preparation.NumericalEnvironment(s.ctx, &pb.NumericalEnvironmentRequest{})
	}
	if err != nil || result == nil || len(result.Digest) != 32 {
		if err != nil {
			problem := operationCacheProblem(err)
			if problem.Code == exit.Structural {
				return "", exit.Named(exit.Structural, "operation.numerical_environment_unproven", "the selected callee's prepared interpreter could not prove its numerical identity")
			}
			return "", problem
		}
		return "", exit.Named(exit.Structural, "operation.numerical_environment_unproven", "the selected callee did not provide its numerical identity")
	}
	environment, _ := canonical.Spell(result.Digest)
	return environment, nil
}
