package records

import (
	"bytes"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const servingPlacementsDDL = `CREATE TABLE IF NOT EXISTS attempt_serving_placements (
  request_id TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  body BLOB NOT NULL CHECK(length(body)>0 AND length(body)<=4194304),
  PRIMARY KEY(request_id,attempt),
  FOREIGN KEY(request_id,attempt) REFERENCES attempts(request_id,attempt)
)`

// BoundServingPlacement reads the exact prepared set retained with this attempt.
// Older unbound attempts have no evidence; new bound offers cannot omit it.
func BoundServingPlacement(request Request, attempt Attempt) (canonical.Doc, *exit.Error) {
	refuse := func() (canonical.Doc, *exit.Error) {
		return nil, exit.Named(exit.Conflict, "serving.placement_evidence_mismatch", "serving attempt differs from its exact prepared package and entrypoint bindings")
	}
	var envelope struct {
		Serving struct {
			Bindings string `json:"bindings_digest"`
		} `json:"serving"`
	}
	_ = json.Unmarshal(attempt.InvocationCanonical, &envelope)
	if len(attempt.ServingPlacementSet) == 0 && envelope.Serving.Bindings == "" {
		return nil, nil
	}
	if request.IsJob() || len(attempt.ServingPlacementSet) == 0 || len(attempt.ServingPlacementSet) > pb.MaxInlineControlBytes {
		return refuse()
	}
	var invocation pb.InvocationSpec
	var placementSet pb.PlacementSet
	if canonical.Unmarshal(attempt.InvocationCanonical, &invocation) != nil || canonical.Unmarshal(attempt.ServingPlacementSet, &placementSet) != nil {
		return refuse()
	}
	invocationID, _ := canonical.Spell(canonical.Digest(attempt.InvocationCanonical))
	serving := invocation.GetServing()
	if invocationID != attempt.InvocationDigest || serving == nil || serving.BindingsDigest == "" {
		return refuse()
	}
	set, _ := canonical.Read(attempt.ServingPlacementSet, &pb.PlacementSet{})
	var selected canonical.Doc
	for _, placement := range set.List("placements") {
		pkg := placement.Sub("package").Str("package")
		if pkg == "" {
			pkg = placement.Sub("development").Str("package")
		}
		if pkg != request.Package || placement.Str("bindings_digest") != serving.BindingsDigest {
			continue
		}
		if selected != nil || !bindingDigest(placement, []string{"entrypoints", "models"}, serving.BindingsDigest) {
			return refuse()
		}
		matches := 0
		for _, entrypoint := range placement.List("entrypoints") {
			if entrypoint.Str("name") == request.Entrypoint && entrypoint.Str("entrypoint_binding_digest") == serving.EntrypointBindingDigest {
				if !bindingDigest(entrypoint, []string{"name", "slots"}, serving.EntrypointBindingDigest) {
					return refuse()
				}
				matches++
			}
		}
		if matches != 1 {
			return refuse()
		}
		selected = placement
	}
	if selected == nil {
		return refuse()
	}
	return selected, nil
}

func bindingDigest(doc canonical.Doc, fields []string, expected string) bool {
	value := make(map[string]canonical.Value, len(fields))
	for _, field := range fields {
		item, ok := doc[field]
		if !ok {
			item = []canonical.Value{}
		}
		value[field] = item
	}
	body, err := canonical.Write(value)
	wanted, digestErr := canonical.Raw(expected)
	return err == nil && digestErr == nil && bytes.Equal(canonical.Digest(body), wanted)
}
