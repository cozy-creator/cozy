package launch

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// BuiltinOperationsTool uses the admitted host Runtime for base-owned metadata
// and environment preparation. All subprocess execution stays in launch.
func BuiltinOperationsTool(root, scratch string, env []string) (RuntimeCLI, *exit.Error) {
	bin, problem := hostruntime.Path(env)
	if problem != nil {
		return RuntimeCLI{}, problem
	}
	return RuntimeCLI{Bin: bin, Dir: root, Home: scratch, Env: env}, nil
}

func (r RuntimeCLI) RuntimeVersion(ctx context.Context) (string, *exit.Error) {
	var answer struct {
		Distribution string `json:"distribution"`
	}
	problem := r.callContext(ctx, &answer, "version")
	return answer.Distribution, problem
}
