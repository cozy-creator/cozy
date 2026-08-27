package app

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
	"github.com/cozy-creator/cozy-creator-v2/internal/video"
)

func handleVideoCompose(ctx *Context) *exit.Error {
	if output := ctx.Inv.Value("--out"); output != "" {
		if compositionOverwritesSource(ctx.Inv.Args[0], output) {
			return exit.New(exit.Conflict,
				"composition output may not overwrite its editable Cozy Video source")
		}
	}
	composition, problem := composeVideo(ctx, false)
	if problem != nil {
		return problem
	}
	if output := ctx.Inv.Value("--out"); output != "" {
		if problem := writeComposition(output, composition); problem != nil {
			return problem
		}
	}
	return emit(ctx, render.Record{Kind: "video_composition", Fields: []render.Field{
		{K: "source_digest", V: composition.SourceDigest},
		{K: "creative_plan_digest", V: composition.CreativePlanDigest},
		{K: "shots", V: composition.ShotCount}, {K: "steps", V: composition.ShotCount + 1},
		{K: "assets", V: len(composition.Assets)}, {K: "output", V: ctx.Inv.Value("--out")},
	}, Notes: []string{"composition stages immutable assets but starts no workflow"},
		Next: []string{"cozy video submit " + composition.CreativePlanDigest +
			videoResolutionFlags(ctx)}})
}

func compositionOverwritesSource(sourcePath, outputPath string) bool {
	source, sourceErr := filepath.Abs(sourcePath)
	output, outputErr := filepath.Abs(outputPath)
	if sourceErr != nil || outputErr != nil {
		return false
	}
	if source == output {
		return true
	}
	sourceInfo, sourceErr := os.Stat(source)
	outputInfo, outputErr := os.Stat(output)
	return sourceErr == nil && outputErr == nil && os.SameFile(sourceInfo, outputInfo)
}

func handleVideoSubmit(ctx *Context) *exit.Error {
	key := strings.TrimSpace(ctx.Inv.Value("--idempotency-key"))
	if key == "" {
		return exit.Usagef("video submit requires --idempotency-key so a lost response can be retried")
	}
	composition, problem := composeVideo(ctx, true)
	if problem != nil {
		return problem
	}
	assets := make([]api.WorkflowAssetResolution, 0, len(composition.Assets))
	for _, asset := range composition.Assets {
		assets = append(assets, api.WorkflowAssetResolution{Step: asset.Step,
			FieldPath: asset.FieldPath, Digest: asset.Digest, Length: asset.Length,
			MediaType: asset.MediaType, Order: asset.Order})
	}
	rentalID := strings.TrimSpace(ctx.Inv.Value("--rental"))
	var targets []api.WorkflowTargetResolution
	if rentalID != "" {
		for step := 1; step <= composition.ShotCount; step++ {
			targets = append(targets, api.WorkflowTargetResolution{Step: step, Worker: rentalID})
		}
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	handle, problem := client.SubmitWorkflow(api.WorkflowSubmission{
		Plan: composition.WorkflowPlan, Assets: assets, Targets: targets,
	}, key)
	if problem != nil {
		return problem
	}
	notes := []string{"source and creative identities are retained separately from workflow execution"}
	if handle.Replay {
		notes = append(notes, "this workflow key already existed; no child was duplicated")
	}
	return emit(ctx, render.Record{Kind: "workflow", Fields: []render.Field{
		{K: "workflow", V: handle.WorkflowID}, {K: "status", V: handle.Status},
		{K: "source_digest", V: composition.SourceDigest},
		{K: "creative_plan_digest", V: composition.CreativePlanDigest},
	}, Notes: notes, Next: []string{"cozy workflow status " + handle.WorkflowID}})
}

func composeVideo(ctx *Context, allowDigest bool) (video.Composition, *exit.Error) {
	h3, assembler := strings.TrimSpace(ctx.Inv.Value("--h3")), strings.TrimSpace(ctx.Inv.Value("--assembler"))
	if h3 == "" || assembler == "" {
		return video.Composition{}, exit.Usagef("video composition requires --h3 and --assembler endpoint refs")
	}
	input := ctx.Inv.Args[0]
	request := api.VideoComposeRequest{H3Endpoint: h3, AssemblyEndpoint: assembler,
		RentalID: strings.TrimSpace(ctx.Inv.Value("--rental"))}
	if allowDigest && strings.HasPrefix(input, "sha256:") {
		request.CreativePlanDigest = input
	} else {
		data, base, problem := readVideoSource(input)
		if problem != nil {
			return video.Composition{}, problem
		}
		request.Source, request.BaseDir = data, base
	}
	client, problem := dial(ctx)
	if problem != nil {
		return video.Composition{}, problem
	}
	return client.ComposeVideo(request)
}

func readVideoSource(path string) ([]byte, string, *exit.Error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, "", exit.New(exit.NotFound, "cannot resolve video source %s: %s", path, err)
	}
	file, err := os.Open(absolute)
	if err != nil {
		return nil, "", exit.New(exit.NotFound, "cannot open video source %s: %s", path, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return nil, "", exit.New(exit.Validation,
			"video source %s is not one readable regular file", filepath.Base(path))
	}
	data, err := io.ReadAll(io.LimitReader(file, video.MaxSourceBytes+1))
	if err != nil || len(data) > video.MaxSourceBytes {
		return nil, "", exit.New(exit.Validation,
			"video source %s is unreadable or larger than %d bytes", path, video.MaxSourceBytes)
	}
	return data, filepath.Dir(absolute), nil
}

func writeComposition(path string, composition video.Composition) *exit.Error {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return exit.New(exit.NotFound, "cannot resolve composition output %s: %s", path, err)
	}
	data, err := json.MarshalIndent(composition, "", "  ")
	if err != nil {
		return exit.Internalf("cannot render video composition: %s", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(absolute), 0o755); err != nil {
		return exit.Internalf("cannot create composition output directory: %s", err)
	}
	file, err := os.CreateTemp(filepath.Dir(absolute), "."+filepath.Base(absolute)+".staging-*")
	if err != nil {
		return exit.Internalf("cannot stage video composition: %s", err)
	}
	staging := file.Name()
	keep := false
	defer func() {
		_ = file.Close()
		if !keep {
			_ = os.Remove(staging)
		}
	}()
	if _, err := file.Write(data); err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return exit.Internalf("cannot write video composition: %s", err)
	}
	if err := os.Rename(staging, absolute); err != nil {
		return exit.Internalf("cannot publish video composition: %s", err)
	}
	keep = true
	return nil
}

func videoResolutionFlags(ctx *Context) string {
	flags := " --h3 " + ctx.Inv.Value("--h3") + " --assembler " + ctx.Inv.Value("--assembler") +
		" --idempotency-key <key>"
	if rentalID := ctx.Inv.Value("--rental"); rentalID != "" {
		flags += " --rental " + rentalID
	}
	return flags
}
