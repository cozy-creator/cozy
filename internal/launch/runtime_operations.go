package launch

import (
	"context"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/runtimeoperation"
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

type BuiltinPreparation struct {
	EnvironmentPython    string          `json:"environment_python"`
	EnvironmentDigest    string          `json:"environment_digest"`
	ContentDigest        string          `json:"content_digest"`
	ReceiptDigest        string          `json:"receipt_digest"`
	ImplementationDigest string          `json:"implementation_digest"`
	Closure              string          `json:"closure"`
	RuntimeVersion       string          `json:"runtime_version"`
	PackageInterface     json.RawMessage `json:"package_interface"`
}

func (r RuntimeCLI) RuntimeVersion(ctx context.Context) (string, *exit.Error) {
	var answer struct {
		Distribution string `json:"distribution"`
	}
	problem := r.callContext(ctx, &answer, "version")
	return answer.Distribution, problem
}

func (r RuntimeCLI) BuiltinOperations(ctx context.Context) (*PackageInterface, *exit.Error) {
	var raw json.RawMessage
	if problem := r.callContext(ctx, &raw, "describe", "--builtin", "operations"); problem != nil {
		return nil, problem
	}
	surface, problem := DecodePackageInterface(raw)
	if problem != nil {
		return nil, problem
	}
	if surface.Application != runtimeoperation.Application || len(surface.Entrypoints) != 0 || len(surface.Jobs) != 1 {
		return nil, exit.New(exit.Conflict, "Runtime operations descriptor changed its fixed App")
	}
	job := surface.Jobs[0]
	if job.Name != "quantize" || job.Invocable == nil || !job.Invocable.Memoize || job.Invocable.Module != runtimeoperation.Module || job.Invocable.Export != "quantize" || job.Publishes || len(job.WeightsOutputs) != 1 || job.WeightsOutputs[0].OutputID != "model" || job.WeightsOutputs[0].MaxBytes == 0 {
		return nil, exit.New(exit.Conflict, "Runtime operations descriptor changed its fixed callable")
	}
	for _, capability := range job.Invocable.Capabilities {
		if capability == "egress" || capability == "secrets" {
			return nil, exit.New(exit.Conflict, "Runtime quantization cannot carry external effect capabilities")
		}
	}
	return surface, nil
}

func (r RuntimeCLI) PrepareBuiltin(ctx context.Context, wheel, output string) (BuiltinPreparation, *exit.Error) {
	var result BuiltinPreparation
	problem := r.callContext(ctx, &result, "builtin-prepare", "operations", "--wheel", wheel, "--out", output)
	return result, problem
}
