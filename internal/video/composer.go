package video

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/inputasset"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/workflow"
)

type Resolver interface {
	ResolvePlacement(endpoint string) (orchestrator.DesiredPlacement, *exit.Error)
	Entrypoint(installID, name string) (*launch.Entrypoint, *exit.Error)
}

type Options struct {
	Store            *records.Store
	Layout           home.Layout
	Resolver         Resolver
	Rentals          func(id string) (*orchestrator.DesiredPlacement, *exit.Error)
	RemoteEntrypoint func(worker, name string) (*launch.Entrypoint, *exit.Error)
}

type Composer struct{ opt Options }

const assemblyEndpoint = "cozy/video-assembly"

type ComposeRequest struct {
	Source             []byte
	BaseDir            string
	CreativePlanDigest string
	H3Endpoint         string
	RentalID           string
}

func Open(opt Options) (*Composer, *exit.Error) {
	if opt.Store == nil || opt.Resolver == nil || opt.Layout.Root == "" {
		return nil, exit.Internalf("video composer needs records, layout, and endpoint resolver")
	}
	return &Composer{opt: opt}, nil
}

func (c *Composer) Compose(request ComposeRequest) (Composition, *exit.Error) {
	if (len(request.Source) == 0) == (request.CreativePlanDigest == "") {
		return Composition{}, exit.New(exit.Validation,
			"video composition needs exactly one source document or creative_plan_digest")
	}
	if request.CreativePlanDigest != "" {
		profile, problem := c.resolveProfile(request)
		if problem != nil {
			return Composition{}, problem
		}
		return c.recompose(request.CreativePlanDigest, profile)
	}
	if !filepath.IsAbs(request.BaseDir) {
		return Composition{}, exit.New(exit.Validation,
			"video source base_dir must be one absolute local directory")
	}
	source, problem := DecodeSource(request.Source)
	if problem != nil {
		return Composition{}, problem
	}
	sourceDigest, err := canonical.Spell(canonical.Digest(request.Source))
	if err != nil {
		return Composition{}, exit.Internalf("cannot spell the source digest: %s", err)
	}
	profile, problem := c.resolveProfile(request)
	if problem != nil {
		return Composition{}, problem
	}

	unlock := inputasset.Guard()
	defer unlock()
	creative, assets, staged, problem := c.stageSource(source, request.BaseDir, profile)
	rollback := func() {
		if problem := inputasset.DropUnowned(c.opt.Layout, c.opt.Store, staged); problem != nil {
			// The next boot sweep sees the same unowned digests. Composition errors remain
			// about the source, not about best-effort rollback of content-addressed bytes.
			_ = problem
		}
	}
	if problem != nil {
		rollback()
		return Composition{}, problem
	}
	creativeBytes, creativeDigest, problem := encodeCreative(creative)
	if problem != nil {
		rollback()
		return Composition{}, problem
	}
	if !sameCompositionAssets(assets, expectedCompositionAssets(creative)) {
		rollback()
		return Composition{}, exit.Internalf("fresh composition assets disagree with creative plan")
	}
	plan, problem := buildWorkflow(creative, creativeDigest, assets, profile)
	if problem != nil {
		rollback()
		return Composition{}, problem
	}
	row, _, problem := c.opt.Store.RecordVideoComposition(records.VideoComposition{
		SourceDigest: sourceDigest, CreativePlanDigest: creativeDigest,
		CreativePlan: creativeBytes, Assets: assets,
	})
	if problem != nil {
		rollback()
		return Composition{}, problem
	}
	return compositionOf(row, creative, plan), nil
}

func (c *Composer) recompose(digest string, profile videoProfile) (Composition, *exit.Error) {
	if _, err := canonical.Raw(digest); err != nil {
		return Composition{}, exit.New(exit.Validation,
			"creative_plan_digest is not sha256:<64 lowercase hex>")
	}
	row, problem := c.opt.Store.VideoCompositionByCreativePlan(digest)
	if problem != nil {
		return Composition{}, problem
	}
	if row == nil {
		return Composition{}, exit.New(exit.NotFound,
			"no retained video composition has creative plan %s", digest)
	}
	creative, problem := decodeCreative(row.CreativePlan, row.CreativePlanDigest)
	if problem != nil {
		return Composition{}, problem
	}
	for _, asset := range row.Assets {
		entrypoint := profile.entrypointForStep(asset.Step, *creative)
		spec, ok := launch.AssetSpec(entrypoint, asset.FieldPath)
		if !ok || spec.Kind != asset.Kind || spec.MaxBytes <= 0 ||
			asset.Length > spec.MaxBytes || !spec.AcceptsMediaType(asset.MediaType) {
			return Composition{}, exit.Named(exit.Conflict, "video_profile_asset_mismatch",
				"current resolved profile no longer admits step %d asset %s",
				asset.Step, asset.FieldPath)
		}
		binding := compositionBinding(c.opt.Layout, asset, spec.MaxBytes)
		if problem := inputasset.Verify(binding, spec.MaxBytes); problem != nil {
			return Composition{}, problem
		}
	}
	row.SourceDigest = ""
	return c.render(*row, *creative, profile)
}

func (c *Composer) render(row records.VideoComposition, creative creativePlan,
	profile videoProfile) (Composition, *exit.Error) {
	if !sameCompositionAssets(row.Assets, expectedCompositionAssets(creative)) {
		return Composition{}, exit.Named(exit.Structural, "video_composition_assets_corrupt",
			"retained composition assets disagree with its canonical creative plan")
	}
	plan, problem := buildWorkflow(creative, row.CreativePlanDigest, row.Assets, profile)
	if problem != nil {
		return Composition{}, problem
	}
	return compositionOf(row, creative, plan), nil
}

func compositionOf(row records.VideoComposition, creative creativePlan, plan []byte) Composition {
	return Composition{
		SourceDigest: row.SourceDigest, CreativePlanDigest: row.CreativePlanDigest,
		CreativePlan: append([]byte(nil), row.CreativePlan...), WorkflowPlan: plan,
		Assets:    append([]records.CompositionAsset(nil), row.Assets...),
		ShotCount: len(creative.Shots),
	}
}

type actionResolution struct {
	Endpoint   string
	ReleaseID  string
	Entrypoint string
	PlanID     string
	Outputs    []string
	Schema     *launch.Entrypoint
}

type videoProfile struct {
	reference actionResolution
	firstLast actionResolution
	assembly  actionResolution
}

func (p videoProfile) entrypointForStep(step int, creative creativePlan) *launch.Entrypoint {
	if step == len(creative.Shots)+1 {
		return p.assembly.Schema
	}
	if step >= 1 && step <= len(creative.Shots) &&
		creative.Shots[step-1].Action == "reference_media_to_video" {
		return p.reference.Schema
	}
	return p.firstLast.Schema
}

func (c *Composer) resolveProfile(request ComposeRequest) (videoProfile, *exit.Error) {
	localEndpoint, rentalID := strings.TrimSpace(request.H3Endpoint), strings.TrimSpace(request.RentalID)
	if (localEndpoint == "") == (rentalID == "") {
		return videoProfile{}, exit.New(exit.Validation,
			"video composition requires exactly one local h3_endpoint or remote rental_id")
	}
	var h3 orchestrator.DesiredPlacement
	var problem *exit.Error
	remote := rentalID != ""
	if remote {
		if c.opt.Rentals == nil || c.opt.RemoteEntrypoint == nil {
			return videoProfile{}, exit.Unavailablef("this LocalService resolves no rented video target")
		}
		resolved, e := c.opt.Rentals(rentalID)
		if e != nil || resolved == nil {
			if e == nil {
				e = exit.New(exit.NotFound, "no attached rental %s", request.RentalID)
			}
			return videoProfile{}, e
		}
		h3 = *resolved
	} else {
		h3, problem = c.opt.Resolver.ResolvePlacement(localEndpoint)
		if problem != nil {
			return videoProfile{}, problem
		}
	}
	resolveH3 := func(name string) (actionResolution, *exit.Error) {
		var schema *launch.Entrypoint
		var problem *exit.Error
		if remote {
			schema, problem = c.opt.RemoteEntrypoint(request.RentalID, name)
		} else {
			schema, problem = c.opt.Resolver.Entrypoint(h3.InstallID, name)
		}
		if problem != nil {
			return actionResolution{}, problem
		}
		return resolveAction(h3, name, []string{"video", "continuation_frame"}, schema)
	}
	reference, problem := resolveH3("reference_media_to_video")
	if problem != nil {
		return videoProfile{}, problem
	}
	firstLast, problem := resolveH3("first_last_frame_to_video")
	if problem != nil {
		return videoProfile{}, problem
	}
	assemblyPlacement, problem := c.opt.Resolver.ResolvePlacement(assemblyEndpoint)
	if problem != nil {
		return videoProfile{}, problem
	}
	assemblySchema, problem := c.opt.Resolver.Entrypoint(
		assemblyPlacement.InstallID, "assemble_video")
	if problem != nil {
		return videoProfile{}, problem
	}
	assembly, problem := resolveAction(assemblyPlacement, "assemble_video", []string{"video"}, assemblySchema)
	if problem != nil {
		return videoProfile{}, problem
	}
	return videoProfile{reference: reference, firstLast: firstLast, assembly: assembly}, nil
}

func resolveAction(placement orchestrator.DesiredPlacement, name string, outputs []string,
	schema *launch.Entrypoint) (actionResolution, *exit.Error) {
	for _, binding := range placement.Bindings {
		if binding.Entrypoint != name {
			continue
		}
		planID, problem := binding.PlanID()
		if problem != nil {
			return actionResolution{}, problem
		}
		if !sameStrings(binding.Outputs, outputs) {
			return actionResolution{}, exit.Named(exit.Conflict, "video_action_outputs",
				"%s/%s publishes [%s], not [%s]", placement.Endpoint, name,
				strings.Join(binding.Outputs, ","), strings.Join(outputs, ","))
		}
		return actionResolution{Endpoint: placement.Endpoint, ReleaseID: placement.ReleaseID,
			Entrypoint: name, PlanID: planID, Outputs: append([]string(nil), binding.Outputs...),
			Schema: schema}, nil
	}
	return actionResolution{}, exit.New(exit.NotFound,
		"%s has no video-composer action %q", placement.Endpoint, name)
}

func (c *Composer) stageSource(source *Source, base string,
	profile videoProfile) (creativePlan, []records.CompositionAsset, []records.AssetBinding, *exit.Error) {
	creative := creativePlan{Format: CreativePlanFormat, Shots: make([]creativeShot, 0, len(source.Shots))}
	var assets []records.CompositionAsset
	var staged []records.AssetBinding
	stagedByPath := map[string]records.AssetBinding{}
	stage := func(step int, field, path string, order uint32,
		schema *launch.Entrypoint) (creativeAsset, *exit.Error) {
		resolved := path
		if !filepath.IsAbs(resolved) {
			resolved = filepath.Join(base, resolved)
		}
		absolute, err := filepath.Abs(resolved)
		if err != nil {
			return creativeAsset{}, exit.New(exit.NotFound, "cannot resolve video asset %s: %s", path, err)
		}
		spec, ok := launch.AssetSpec(schema, field)
		if !ok || spec.MaxBytes <= 0 {
			return creativeAsset{}, exit.Named(exit.Structural, "video_asset_field",
				"resolved action has no bounded asset field %s", field)
		}
		binding, reused := stagedByPath[absolute]
		if !reused {
			var problem *exit.Error
			binding, problem = inputasset.Stage(c.opt.Layout, records.AssetBinding{
				FieldPath: field, LocalPath: absolute, Order: order, MaxBytes: spec.MaxBytes,
			}, spec.MaxBytes)
			if problem != nil {
				return creativeAsset{}, exit.Named(problem.Code, problem.ErrName(),
					"video asset %s for field %s could not be staged",
					filepath.Base(path), field)
			}
			stagedByPath[absolute] = binding
			staged = append(staged, binding)
		}
		mediaKind, _, _ := strings.Cut(binding.MediaType, "/")
		if binding.Length > spec.MaxBytes || mediaKind != spec.Kind ||
			!spec.AcceptsMediaType(binding.MediaType) {
			return creativeAsset{}, exit.Named(exit.Validation, "video_asset_media_type",
				"asset %s is %s and field %s does not admit that media type",
				filepath.Base(path), binding.MediaType, field)
		}
		asset := creativeAsset{Digest: binding.Digest, Length: binding.Length,
			MediaType: binding.MediaType, Kind: spec.Kind}
		assets = append(assets, records.CompositionAsset{Step: step, FieldPath: field,
			Digest: asset.Digest, Length: asset.Length, MediaType: asset.MediaType,
			Kind: asset.Kind, Order: order})
		return asset, nil
	}
	for index, sourceShot := range source.Shots {
		step := index + 1
		shot := creativeShot{PromptBase64: base64.StdEncoding.EncodeToString([]byte(string(sourceShot.Prompt))),
			SeedDecimal: strconv.FormatInt(int64(*sourceShot.Seed), 10)}
		if sourceShot.Reference != nil {
			shot.Action = "reference_media_to_video"
			for refIndex, reference := range sourceShot.Reference.References {
				kind, path := referenceKind(reference)
				field := fmt.Sprintf("references.%d.%s", refIndex, kind)
				asset, problem := stage(step, field, path, uint32(refIndex), profile.reference.Schema)
				if problem != nil {
					return creativePlan{}, nil, staged, problem
				}
				shot.References = append(shot.References, asset)
			}
		} else {
			shot.Action = "first_last_frame_to_video"
			if sourceShot.FirstLast.FirstFrame.Previous {
				shot.Previous = true
			} else if path := sourceShot.FirstLast.FirstFrame.Path; path != "" {
				asset, problem := stage(step, "first_frame", path, 0, profile.firstLast.Schema)
				if problem != nil {
					return creativePlan{}, nil, staged, problem
				}
				shot.FirstAssets = []creativeAsset{asset}
			}
			if path := sourceShot.FirstLast.LastFrame.Path; path != "" {
				asset, problem := stage(step, "last_frame", path, 0, profile.firstLast.Schema)
				if problem != nil {
					return creativePlan{}, nil, staged, problem
				}
				shot.LastAssets = []creativeAsset{asset}
			}
		}
		creative.Shots = append(creative.Shots, shot)
	}
	if source.Assembly.MasterAudio != "" {
		asset, problem := stage(len(source.Shots)+1, "master_audio",
			string(source.Assembly.MasterAudio), 0, profile.assembly.Schema)
		if problem != nil {
			return creativePlan{}, nil, staged, problem
		}
		creative.Assembly = creativeAssembly{Audio: "master", MasterAssets: []creativeAsset{asset}}
	} else {
		creative.Assembly = creativeAssembly{Audio: "segments"}
	}
	sort.Slice(assets, func(i, j int) bool {
		if assets[i].Step != assets[j].Step {
			return assets[i].Step < assets[j].Step
		}
		return assets[i].FieldPath < assets[j].FieldPath
	})
	return creative, assets, staged, nil
}

func buildWorkflow(creative creativePlan, creativeDigest string,
	assets []records.CompositionAsset, profile videoProfile) ([]byte, *exit.Error) {
	byStep := map[int]map[string]records.CompositionAsset{}
	for _, asset := range assets {
		if byStep[asset.Step] == nil {
			byStep[asset.Step] = map[string]records.CompositionAsset{}
		}
		byStep[asset.Step][asset.FieldPath] = asset
	}
	steps := make([]workflow.Step, 0, len(creative.Shots)+1)
	for index, shot := range creative.Shots {
		ordinal := index + 1
		resolution := profile.firstLast
		prompt, err := base64.StdEncoding.Strict().DecodeString(shot.PromptBase64)
		if err != nil {
			return nil, exit.Named(exit.Structural, "video_creative_plan_corrupt",
				"creative shot %d prompt is unreadable", ordinal)
		}
		seed, err := strconv.ParseInt(shot.SeedDecimal, 10, 64)
		if err != nil {
			return nil, exit.Named(exit.Structural, "video_creative_plan_corrupt",
				"creative shot %d seed is unreadable", ordinal)
		}
		payload := map[string]any{"prompt": string(prompt), "seed": seed}
		var bindings []workflow.OutputBinding
		if shot.Action == "reference_media_to_video" {
			resolution = profile.reference
			references := make([]any, 0, len(shot.References))
			for refIndex, asset := range shot.References {
				field := fmt.Sprintf("references.%d.%s", refIndex, asset.Kind)
				spec, ok := launch.AssetSpec(profile.reference.Schema, field)
				if !ok || spec.TagField == "" || spec.TagValue == nil {
					return nil, exit.Named(exit.Structural, "video_reference_tag_missing",
						"resolved H3 descriptor publishes no tagged-union discriminator for %s", field)
				}
				references = append(references,
					map[string]any{spec.TagField: spec.TagValue, asset.Kind: ""})
			}
			payload["references"] = references
		} else {
			if shot.Previous {
				payload["first_frame"] = ""
				bindings = append(bindings, workflow.OutputBinding{FieldPath: "first_frame",
					PriorStep: ordinal - 1, OutputName: "continuation_frame",
					ExpectedMediaKind: "image"})
			} else if len(shot.FirstAssets) == 1 {
				payload["first_frame"] = ""
			}
			if len(shot.LastAssets) == 1 {
				payload["last_frame"] = ""
			}
		}
		claims := claimsOf(byStep[ordinal])
		paths := make([]string, 0, len(claims)+len(bindings))
		for _, claim := range claims {
			paths = append(paths, claim.FieldPath)
		}
		for _, binding := range bindings {
			paths = append(paths, binding.FieldPath)
		}
		encoded, problem := validatedPayloadBase64(resolution.Schema, payload, paths)
		if problem != nil {
			return nil, problem
		}
		steps = append(steps, workflow.Step{Endpoint: resolution.Endpoint,
			EndpointReleaseID: resolution.ReleaseID, Entrypoint: resolution.Entrypoint,
			EntrypointBindingPlanID: resolution.PlanID, PayloadBase64: encoded,
			Outputs: append([]string(nil), resolution.Outputs...), Assets: claims, Bindings: bindings})
	}
	assemblyOrdinal := len(creative.Shots) + 1
	videos := make([]string, len(creative.Shots))
	bindings := make([]workflow.OutputBinding, 0, len(creative.Shots))
	for index := range creative.Shots {
		bindings = append(bindings, workflow.OutputBinding{FieldPath: fmt.Sprintf("videos.%d", index),
			PriorStep: index + 1, OutputName: "video", ExpectedMediaKind: "video"})
	}
	payload := map[string]any{"videos": videos}
	if creative.Assembly.Audio == "master" {
		payload["master_audio"] = ""
	}
	paths := make([]string, 0, len(byStep[assemblyOrdinal])+len(bindings))
	for field := range byStep[assemblyOrdinal] {
		paths = append(paths, field)
	}
	for _, binding := range bindings {
		paths = append(paths, binding.FieldPath)
	}
	encoded, problem := validatedPayloadBase64(profile.assembly.Schema, payload, paths)
	if problem != nil {
		return nil, problem
	}
	steps = append(steps, workflow.Step{Endpoint: profile.assembly.Endpoint,
		EndpointReleaseID: profile.assembly.ReleaseID, Entrypoint: profile.assembly.Entrypoint,
		EntrypointBindingPlanID: profile.assembly.PlanID, PayloadBase64: encoded,
		Outputs: append([]string(nil), profile.assembly.Outputs...),
		Assets:  claimsOf(byStep[assemblyOrdinal]), Bindings: bindings})
	plan := workflow.Plan{Format: workflow.PlanFormat,
		CreativePlanDigest: creativeDigest, Steps: steps}
	raw, err := json.Marshal(plan)
	if err != nil {
		return nil, exit.Internalf("cannot render the video workflow plan: %s", err)
	}
	_, canonicalPlan, _, problem := workflow.DecodePlan(raw)
	return canonicalPlan, problem
}

func claimsOf(assets map[string]records.CompositionAsset) []workflow.AssetClaim {
	fields := make([]string, 0, len(assets))
	for field := range assets {
		fields = append(fields, field)
	}
	sort.Strings(fields)
	out := make([]workflow.AssetClaim, 0, len(fields))
	for _, field := range fields {
		asset := assets[field]
		out = append(out, workflow.AssetClaim{FieldPath: field, Digest: asset.Digest,
			Length: asset.Length, MediaType: asset.MediaType, Kind: asset.Kind, Order: asset.Order})
	}
	return out
}

func sameCompositionAssets(left, right []records.CompositionAsset) bool {
	if len(left) != len(right) {
		return false
	}
	a := append([]records.CompositionAsset(nil), left...)
	b := append([]records.CompositionAsset(nil), right...)
	sort.Slice(a, func(i, j int) bool {
		if a[i].Step != a[j].Step {
			return a[i].Step < a[j].Step
		}
		return a[i].FieldPath < a[j].FieldPath
	})
	sort.Slice(b, func(i, j int) bool {
		if b[i].Step != b[j].Step {
			return b[i].Step < b[j].Step
		}
		return b[i].FieldPath < b[j].FieldPath
	})
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}

func compositionBinding(layout home.Layout, asset records.CompositionAsset,
	maxBytes int64) records.AssetBinding {
	return records.AssetBinding{FieldPath: asset.FieldPath, LocalPath: layout.InputAsset(asset.Digest),
		Digest: asset.Digest, Length: asset.Length, MediaType: asset.MediaType,
		Order: asset.Order, MaxBytes: maxBytes}
}

func validatedPayloadBase64(schema *launch.Entrypoint, document map[string]any,
	assetPaths []string) (string, *exit.Error) {
	data, err := json.Marshal(document)
	if err != nil {
		return "", exit.Internalf("cannot encode a video step payload: %s", err)
	}
	if problem := launch.ValidatePayloadAssets(schema, data, assetPaths); problem != nil {
		return "", problem
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

func referenceKind(reference ReferenceSource) (string, string) {
	switch {
	case reference.Image != "":
		return "image", string(reference.Image)
	case reference.Video != "":
		return "video", string(reference.Video)
	default:
		return "audio", string(reference.Audio)
	}
}

func sameStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	a, b := append([]string(nil), left...), append([]string(nil), right...)
	sort.Strings(a)
	sort.Strings(b)
	return strings.Join(a, "\x00") == strings.Join(b, "\x00")
}
