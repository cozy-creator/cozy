package launch

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/modelsource"
)

// ModelIngestionRecipe is data owned by Runtime's model module, never inferred by Creator.
type ModelIngestionRecipe struct {
	Name       string                              `json:"name"`
	Repository string                              `json:"repository"`
	Revision   string                              `json:"revision"`
	Profile    string                              `json:"profile"`
	Carriers   []string                            `json:"carriers"`
	Metadata   map[string]modelsource.MetadataFile `json:"metadata"`
}

func (r RuntimeCLI) ModelIngestionPlan(ctx context.Context, repository, revision string) (*ModelIngestionRecipe, *exit.Error) {
	var answer struct {
		Recipe *ModelIngestionRecipe `json:"recipe"`
	}
	problem := r.callContext(ctx, &answer, "model-ingestion-plan", repository, revision)
	if problem != nil && problem.Code == exit.Usage {
		return nil, problem.WithRemedy("install cozy-runtime 0.18.24 or newer for native model ingestion")
	}
	return answer.Recipe, problem
}
