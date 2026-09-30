package orchestrator

import (
	"cmp"
	"slices"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ModelChoices is shared by release roots, frozen captures and private serving
// preparation. Explicit child targets keep the complete package/slot identity.
func ModelChoices(request records.Request, models []ModelRef) ([]*pb.ModelChoice, *exit.Error) {
	var out []*pb.ModelChoice
	for _, model := range models {
		path := model.BindingSlot()
		entrypoint, parameter, attributed := strings.Cut(path, ".models.")
		if !attributed {
			parameter = model.Slot
		}
		child := model.Package != "" && model.Package != request.Package || attributed && entrypoint != request.Entrypoint
		if child {
			if !model.Choice && len(model.Adapters) == 0 {
				continue // advisory child sizing is not an explicit override
			}
			parameter = path
			if model.Package != "" && model.Package != request.Package {
				parameter = model.Package + "/" + path
			}
		}
		choice := &pb.ModelChoice{Parameter: parameter, Repository: model.Model,
			Release: model.Release, Lane: model.Lane, Source: model.Source,
			Profiles: model.Profiles, Adapters: downloadAdapters(model.Adapters)}
		if model.Manifest != "" {
			digest, err := canonical.Raw(model.Manifest)
			if err != nil {
				return nil, exit.Named(exit.Validation, "model.manifest_invalid", "%s names no exact manifest", path)
			}
			choice.Manifest = &pb.Ref{Digest: digest, Length: uint64(max(model.ManifestLength, 0))}
		}
		out = append(out, choice)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Parameter < out[j].Parameter })
	for i := 1; i < len(out); i++ {
		if out[i-1].Parameter == out[i].Parameter {
			return nil, exit.Usagef("model slot %s was selected more than once", out[i].Parameter)
		}
	}
	return out, nil
}

// PrivateModelChoices attributes each slot because private preparation selects
// an installation without naming a root entrypoint beside these choices.
func PrivateModelChoices(request records.Request) ([]*pb.ModelChoice, *exit.Error) {
	choices, problem := ModelChoices(request, request.OwnModels())
	if problem != nil {
		return nil, problem
	}
	for _, choice := range choices {
		if !strings.Contains(choice.Parameter, ".models.") {
			choice.Parameter = request.Entrypoint + ".models." + choice.Parameter
		}
	}
	return choices, nil
}

func downloadAdapters(adapters []records.ModelAdapterRef) []*pb.DownloadAdapterRef {
	out := make([]*pb.DownloadAdapterRef, 0, len(adapters))
	for _, a := range adapters {
		out = append(out, &pb.DownloadAdapterRef{Component: a.Component, Model: a.Model, Release: a.Release, Lane: a.Lane, Manifest: a.Manifest, SourceComponent: a.SourceComponent, Scale: a.Scale, Source: a.Source, Profiles: a.Profiles})
	}
	return out
}

func hasModelAdapters(models []ModelRef) bool {
	for _, model := range models {
		if len(model.Adapters) > 0 {
			return true
		}
	}
	return false
}

// resolvePreparedAdapters verifies the machine's resolution of an open choice,
// then uses its concrete stack for this placement check only. Warm reuse still
// compares exact manifests through SameAdapters; the requested capture is unchanged.
func resolvePreparedAdapters(requested, observed []ModelRef) ([]ModelRef, *exit.Error) {
	out := append([]ModelRef(nil), requested...)
	refuse := func() ([]ModelRef, *exit.Error) {
		return nil, exit.Named(exit.Structural, "model_adapters_preparation_mismatch",
			"worker preparation omitted or changed the requested LoRA stack")
	}
	if len(out) == 0 && hasModelAdapters(observed) {
		return refuse()
	}
	for i, selected := range out {
		index := slices.IndexFunc(observed, func(model ModelRef) bool { return model.BindingSlot() == selected.BindingSlot() })
		if index < 0 {
			if len(selected.Adapters) > 0 {
				return refuse()
			}
			continue
		}
		actual := observed[index].Adapters
		if len(selected.Adapters) != len(actual) {
			return refuse()
		}
		for n, want := range selected.Adapters {
			got := actual[n]
			if got.Component == "" || want.Component != "" && want.Component != got.Component ||
				cmp.Or(want.SourceComponent, "adapter") != cmp.Or(got.SourceComponent, "adapter") ||
				cmp.Or(want.Scale, "1") != cmp.Or(got.Scale, "1") ||
				want.Model != "" && want.Model != got.Model || want.Release != "" && want.Release != got.Release ||
				want.Lane != "" && want.Lane != got.Lane || want.Manifest != "" && want.Manifest != got.Manifest {
				return refuse()
			}
			if _, err := canonical.Raw(got.Manifest); err != nil || got.Model == "" {
				return refuse()
			}
		}
		out[i].Adapters = actual
	}
	return out, nil
}

func placementAdapters(slot canonical.Doc, models map[string]canonical.Doc) []records.ModelAdapterRef {
	out := make([]records.ModelAdapterRef, 0, len(slot.List("adapters")))
	for _, a := range slot.List("adapters") {
		model := models[a.Str("model_id")]
		out = append(out, records.ModelAdapterRef{
			Component: a.Str("component"), Model: model.Str("repo"), Release: model.Str("version"), Lane: model.Str("lane"),
			Manifest: model.Sub("manifest").Str("digest"), ManifestLength: model.Sub("manifest").Int("length"),
			SourceComponent: a.Str("source_component"), Scale: a.Str("scale"),
		})
	}
	return out
}
