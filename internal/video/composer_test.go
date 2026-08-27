package video

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/workflow"
)

const (
	h3Endpoint       = "cozy/h3"
	assemblyEndpoint = "cozy/assembly"
	testPlanID       = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

type composerResolver struct {
	placements map[string]orchestrator.DesiredPlacement
	functions  map[string]*launch.Entrypoint
}

func (r composerResolver) ResolvePlacement(endpoint string) (orchestrator.DesiredPlacement, *exit.Error) {
	placement, ok := r.placements[endpoint]
	if !ok {
		return orchestrator.DesiredPlacement{}, exit.New(exit.NotFound, "unknown endpoint %s", endpoint)
	}
	return placement, nil
}

func (r composerResolver) Entrypoint(installID, name string) (*launch.Entrypoint, *exit.Error) {
	entrypoint := r.functions[installID+"/"+name]
	if entrypoint == nil {
		return nil, exit.New(exit.NotFound, "unknown function %s/%s", installID, name)
	}
	return entrypoint, nil
}

func (r composerResolver) ResolveInstall(installID string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	for _, placement := range r.placements {
		if placement.InstallID == installID {
			return orchestrator.WorkerLaunchSpec{Placement: placement}, nil
		}
	}
	return orchestrator.WorkerLaunchSpec{}, exit.New(exit.NotFound, "unknown install %s", installID)
}

func testComposer(t *testing.T) (*Composer, home.Layout, *records.Store) {
	t.Helper()
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	t.Cleanup(store.Close)
	entrypoint := func(document string) *launch.Entrypoint {
		var out launch.Entrypoint
		if err := json.Unmarshal([]byte(document), &out); err != nil {
			t.Fatal(err)
		}
		return &out
	}
	reference := entrypoint(`{"name":"reference_media_to_video","request":{"fields":[
      {"name":"prompt","type":"str","wire":"required","constraints":{"min_length":1,"max_length":4096}},
      {"name":"references","type":{"list":{"tag_field":"type","union":[
        {"tag":"image","tag_field":"type","fields":[
          {"name":"type","type":{"literal":["image"]},"wire":"required","discriminator":true},
          {"name":"image","type":{"asset":"image"},"wire":"required","asset_bound":{"max_bytes":67108864}}]},
        {"tag":"video","tag_field":"type","fields":[
          {"name":"type","type":{"literal":["video"]},"wire":"required","discriminator":true},
          {"name":"video","type":{"asset":"video"},"wire":"required","asset_bound":{"max_bytes":2147483648}}]},
        {"tag":"audio","tag_field":"type","fields":[
          {"name":"type","type":{"literal":["audio"]},"wire":"required","discriminator":true},
          {"name":"audio","type":{"asset":"audio"},"wire":"required","asset_bound":{"max_bytes":134217728}}]}
      ]}},"wire":"required","constraints":{"min_length":1,"max_length":12}},
      {"name":"seed","type":{"union":["int","null"]},"wire":"optional"}
    ]}}`)
	firstLast := entrypoint(`{"name":"first_last_frame_to_video","request":{"fields":[
      {"name":"prompt","type":"str","wire":"required","constraints":{"min_length":1,"max_length":4096}},
      {"name":"first_frame","type":{"union":["null",{"asset":"image"}]},"wire":"optional","asset_bound":{"max_bytes":67108864}},
      {"name":"last_frame","type":{"union":["null",{"asset":"image"}]},"wire":"optional","asset_bound":{"max_bytes":67108864}},
      {"name":"seed","type":{"union":["int","null"]},"wire":"optional"}
    ]}}`)
	assembly := entrypoint(`{"name":"assemble_video","request":{"fields":[
      {"name":"videos","type":{"list":{"asset":"video"}},"wire":"required","constraints":{"min_length":2,"max_length":8},"asset_bound":{"max_bytes":536870912,"media_types":["video/mp4"]}},
      {"name":"master_audio","type":{"union":["null",{"asset":"audio"}]},"wire":"optional","asset_bound":{"max_bytes":536870912}}
    ]}}`)
	binding := func(name string, outputs []string) *orchestrator.Binding {
		return &orchestrator.Binding{Entrypoint: name, Outputs: outputs,
			RuntimePlan: &orchestrator.BindingPlanSubject{SubjectID: testPlanID,
				Digest: testPlanID, Kind: "plan", Length: 1}}
	}
	resolver := composerResolver{placements: map[string]orchestrator.DesiredPlacement{
		h3Endpoint: {Endpoint: h3Endpoint, ReleaseID: h3Endpoint + "@1.0.0", InstallID: "h3-install",
			Bindings: []*orchestrator.Binding{binding(reference.Name, []string{"video", "continuation_frame"}),
				binding(firstLast.Name, []string{"video", "continuation_frame"})}},
		assemblyEndpoint: {Endpoint: assemblyEndpoint, ReleaseID: assemblyEndpoint + "@1.0.0",
			InstallID: "assembly-install", Bindings: []*orchestrator.Binding{
				binding(assembly.Name, []string{"video"})}},
	}, functions: map[string]*launch.Entrypoint{
		"h3-install/reference_media_to_video":  reference,
		"h3-install/first_last_frame_to_video": firstLast,
		"assembly-install/assemble_video":      assembly,
	}}
	for _, install := range []records.EndpointInstall{
		{ID: "h3-install", Endpoint: h3Endpoint, Major: 1, Version: "1.0.0"},
		{ID: "assembly-install", Endpoint: assemblyEndpoint, Major: 1, Version: "1.0.0"},
	} {
		install.SourceKind, install.SourceRef, install.SourceDigest = "dir", ".", testPlanID
		install.Dir, install.Python, install.UV, install.LockDigest = ".", "python", "uv", testPlanID
		install.Platform, install.LinkMode, install.Closure, install.Descriptor =
			"test", "copy", "none", testPlanID
		if _, problem := store.Activate(install); problem != nil {
			t.Fatal(problem)
		}
	}
	composer, problem := Open(Options{Store: store, Layout: layout, Resolver: resolver})
	if problem != nil {
		t.Fatal(problem)
	}
	return composer, layout, store
}

func writePNG(t *testing.T, path string, tail byte) {
	t.Helper()
	data := append([]byte("\x89PNG\r\n\x1a\n"), bytes.Repeat([]byte{tail}, 64)...)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeMP4(t *testing.T, path string) {
	t.Helper()
	data := append([]byte{0, 0, 0, 24}, []byte("ftypmp42\x00\x00\x00\x00mp42isom")...)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeWAV(t *testing.T, path string) {
	t.Helper()
	data := append([]byte("RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"), bytes.Repeat([]byte{0}, 32)...)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func composeRequest(source []byte, base string) ComposeRequest {
	return ComposeRequest{Source: source, BaseDir: base,
		H3Endpoint: h3Endpoint, AssemblyEndpoint: assemblyEndpoint}
}

func TestComposerBuildsOrdinaryH3AndAssemblyWorkflow(t *testing.T) {
	composer, layout, store := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 1)
	composition, problem := composer.Compose(composeRequest([]byte(validSource), base))
	if problem != nil {
		t.Fatal(problem)
	}
	plan, _, _, problem := workflow.DecodePlan(composition.WorkflowPlan)
	if problem != nil || composition.ShotCount != 2 ||
		len(plan.Steps) != 3 || len(composition.Assets) != 1 {
		t.Fatalf("composition=%#v plan=%#v problem=%v", composition, plan, problem)
	}
	if plan.Steps[0].Entrypoint != "reference_media_to_video" ||
		plan.Steps[1].Entrypoint != "first_last_frame_to_video" ||
		len(plan.Steps[1].Bindings) != 1 ||
		plan.Steps[1].Bindings[0].OutputName != "continuation_frame" ||
		plan.Steps[2].Entrypoint != "assemble_video" || len(plan.Steps[2].Bindings) != 2 {
		t.Fatalf("workflow mapping=%#v", plan.Steps)
	}
	asset := composition.Assets[0]
	if asset.FieldPath != "references.0.image" || asset.Kind != "image" || asset.Order != 0 {
		t.Fatalf("asset=%#v", asset)
	}
	if used, problem := store.AssetInUse(asset.Digest); problem != nil || !used {
		t.Fatalf("composition ownership=%v problem=%v", used, problem)
	}
	if _, err := os.Stat(layout.InputAsset(asset.Digest)); err != nil {
		t.Fatal(err)
	}
}

func TestCompositionPassesTheRealWorkflowAdmissionAndMintsOneOrdinaryChild(t *testing.T) {
	composer, layout, store := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 9)
	composition, problem := composer.Compose(composeRequest([]byte(validSource), base))
	if problem != nil {
		t.Fatal(problem)
	}
	owner, problem := orchestrator.Open(orchestrator.Options{Layout: layout, Store: store})
	if problem != nil {
		t.Fatal(problem)
	}
	engine, problem := workflow.Open(workflow.Options{Store: store, Owner: owner,
		Resolver: composer.opt.Resolver.(composerResolver), Layout: layout})
	if problem != nil {
		t.Fatal(problem)
	}
	assets := map[int][]records.AssetBinding{}
	for _, asset := range composition.Assets {
		entrypoint := composer.opt.Resolver.(composerResolver).functions["h3-install/reference_media_to_video"]
		spec, ok := launch.AssetSpec(entrypoint, asset.FieldPath)
		if !ok {
			t.Fatalf("missing asset spec %s", asset.FieldPath)
		}
		assets[asset.Step] = append(assets[asset.Step], compositionBinding(layout, asset, spec.MaxBytes))
	}
	row, fresh, problem := engine.Submit(workflow.Submission{IdempotencyKey: "video-workflow",
		Plan: composition.WorkflowPlan, Assets: assets})
	if problem != nil || !fresh {
		t.Fatalf("workflow=%#v fresh=%v problem=%v", row, fresh, problem)
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, problem := store.WorkflowSteps(row.ID)
	if problem != nil || len(steps) != 3 || steps[0].ChildRequestID == "" ||
		steps[1].ChildRequestID != "" {
		t.Fatalf("steps=%#v problem=%v", steps, problem)
	}
}

func TestTaggedMixedMediaPayloadMatchesExactH3Descriptor(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 10)
	writeMP4(t, filepath.Join(base, "reference.mp4"))
	writeWAV(t, filepath.Join(base, "reference.wav"))
	source := strings.Replace(validSource, "        - image: reference.png",
		"        - image: reference.png\n        - video: reference.mp4\n        - audio: reference.wav", 1)
	composition, problem := composer.Compose(composeRequest([]byte(source), base))
	if problem != nil {
		t.Fatal(problem)
	}
	plan, _, _, problem := workflow.DecodePlan(composition.WorkflowPlan)
	if problem != nil {
		t.Fatal(problem)
	}
	payloadBytes, _ := base64.StdEncoding.DecodeString(plan.Steps[0].PayloadBase64)
	var payload struct {
		References []map[string]any `json:"references"`
	}
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		t.Fatal(err)
	}
	want := []string{"image", "video", "audio"}
	if len(payload.References) != len(want) || len(composition.Assets) != len(want) {
		t.Fatalf("payload=%s assets=%#v", payloadBytes, composition.Assets)
	}
	for index, kind := range want {
		if payload.References[index]["type"] != kind || payload.References[index][kind] != "" {
			t.Fatalf("reference %d=%#v", index, payload.References[index])
		}
	}
}

func TestCreativeIdentityIgnoresCommentsPathsAndShotIDs(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 2)
	writePNG(t, filepath.Join(base, "same-bytes.png"), 2)
	first, problem := composer.Compose(composeRequest([]byte(validSource), base))
	if problem != nil {
		t.Fatal(problem)
	}
	changed := "# editor comment\n" + strings.ReplaceAll(validSource, "id: opening", "id: opener")
	changed = strings.ReplaceAll(changed, "reference.png", "same-bytes.png")
	second, problem := composer.Compose(composeRequest([]byte(changed), base))
	if problem != nil {
		t.Fatal(problem)
	}
	if first.SourceDigest == second.SourceDigest || first.CreativePlanDigest != second.CreativePlanDigest ||
		!bytes.Equal(first.CreativePlan, second.CreativePlan) {
		t.Fatalf("first=%s/%s second=%s/%s", first.SourceDigest, first.CreativePlanDigest,
			second.SourceDigest, second.CreativePlanDigest)
	}
	crlf := strings.ReplaceAll(validSource, "\n", "\r\n")
	third, problem := composer.Compose(composeRequest([]byte(crlf), base))
	if problem != nil || third.SourceDigest == first.SourceDigest ||
		third.CreativePlanDigest != first.CreativePlanDigest {
		t.Fatalf("CRLF composition=%#v problem=%v", third, problem)
	}
	link := filepath.Join(base, "linked.png")
	if err := os.Symlink(filepath.Join(base, "reference.png"), link); err == nil {
		linked := strings.ReplaceAll(validSource, "reference.png", "linked.png")
		fourth, problem := composer.Compose(composeRequest([]byte(linked), base))
		if problem != nil || fourth.CreativePlanDigest != first.CreativePlanDigest {
			t.Fatalf("symlink composition=%#v problem=%v", fourth, problem)
		}
	}
}

func TestSameSourceBytesCanResolveDifferentAssetMeaning(t *testing.T) {
	composer, _, _ := testComposer(t)
	left, right := t.TempDir(), t.TempDir()
	writePNG(t, filepath.Join(left, "reference.png"), 3)
	writePNG(t, filepath.Join(right, "reference.png"), 4)
	first, problem := composer.Compose(composeRequest([]byte(validSource), left))
	if problem != nil {
		t.Fatal(problem)
	}
	second, problem := composer.Compose(composeRequest([]byte(validSource), right))
	if problem != nil {
		t.Fatal(problem)
	}
	if first.SourceDigest != second.SourceDigest || first.CreativePlanDigest == second.CreativePlanDigest {
		t.Fatalf("first=%s/%s second=%s/%s", first.SourceDigest, first.CreativePlanDigest,
			second.SourceDigest, second.CreativePlanDigest)
	}
}

func TestRetainedCreativePlanNeverReopensOriginalPath(t *testing.T) {
	composer, layout, _ := testComposer(t)
	base := t.TempDir()
	original := filepath.Join(base, "reference.png")
	writePNG(t, original, 5)
	first, problem := composer.Compose(composeRequest([]byte(validSource), base))
	if problem != nil {
		t.Fatal(problem)
	}
	if err := os.Remove(original); err != nil {
		t.Fatal(err)
	}
	replayed, problem := composer.Compose(ComposeRequest{CreativePlanDigest: first.CreativePlanDigest,
		H3Endpoint: h3Endpoint, AssemblyEndpoint: assemblyEndpoint})
	if problem != nil || replayed.CreativePlanDigest != first.CreativePlanDigest ||
		replayed.SourceDigest != "" {
		t.Fatalf("replayed=%#v problem=%v", replayed, problem)
	}
	if err := os.WriteFile(layout.InputAsset(first.Assets[0].Digest), []byte("corrupt"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, problem := composer.Compose(ComposeRequest{CreativePlanDigest: first.CreativePlanDigest,
		H3Endpoint: h3Endpoint, AssemblyEndpoint: assemblyEndpoint}); problem == nil {
		t.Fatal("corrupted retained asset accepted")
	}
}

func TestMasterAudioIsExclusiveAtAssemblyWithoutChangingH3Soundtracks(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 14)
	writeWAV(t, filepath.Join(base, "master.wav"))
	source := strings.Replace(validSource, "audio: segments", "audio: master\n  master_audio: master.wav", 1)
	composition, problem := composer.Compose(composeRequest([]byte(source), base))
	if problem != nil {
		t.Fatal(problem)
	}
	plan, _, _, _ := workflow.DecodePlan(composition.WorkflowPlan)
	for index := 0; index < 2; index++ {
		payloadBytes, _ := base64.StdEncoding.DecodeString(plan.Steps[index].PayloadBase64)
		var payload map[string]any
		_ = json.Unmarshal(payloadBytes, &payload)
		if _, present := payload["mute"]; present {
			t.Fatalf("shot %d unexpectedly changed soundtrack mode: %s", index+1, payloadBytes)
		}
	}
	assembly := plan.Steps[2]
	if len(assembly.Assets) != 1 || assembly.Assets[0].FieldPath != "master_audio" {
		t.Fatalf("assembly=%#v", assembly)
	}
}

func TestCompositionRefusesAProfileRequiredFieldBeforeRetention(t *testing.T) {
	composer, layout, _ := testComposer(t)
	resolver := composer.opt.Resolver.(composerResolver)
	assembly := resolver.functions["assembly-install/assemble_video"]
	assembly.Request.Fields = append(assembly.Request.Fields,
		launch.Field{Name: "future_required", Type: json.RawMessage(`"str"`), Wire: "required"})
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 15)
	if _, problem := composer.Compose(composeRequest([]byte(validSource), base)); problem == nil {
		t.Fatal("incompatible profile composed")
	}
	entries, err := os.ReadDir(layout.Inputs)
	if err != nil || len(entries) != 0 {
		t.Fatalf("staged entries=%v error=%v", entries, err)
	}
}

func TestWorkerResolutionDoesNotEnterCreativeMeaning(t *testing.T) {
	local, layout, store := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 8)
	first, problem := local.Compose(composeRequest([]byte(validSource), base))
	if problem != nil {
		t.Fatal(problem)
	}
	remote, problem := Open(Options{Store: store, Layout: layout, Resolver: local.opt.Resolver,
		Rentals: func(id string) (*orchestrator.DesiredPlacement, *exit.Error) {
			if id != "rnt-h3" {
				t.Fatalf("worker=%s", id)
			}
			placement, problem := local.opt.Resolver.ResolvePlacement(h3Endpoint)
			return &placement, problem
		},
		RemoteEntrypoint: func(worker, name string) (*launch.Entrypoint, *exit.Error) {
			return local.opt.Resolver.Entrypoint("h3-install", name)
		},
	})
	if problem != nil {
		t.Fatal(problem)
	}
	request := composeRequest([]byte(validSource), base)
	request.Worker = "rnt-h3"
	second, problem := remote.Compose(request)
	if problem != nil || second.CreativePlanDigest != first.CreativePlanDigest ||
		!bytes.Equal(second.CreativePlan, first.CreativePlan) {
		t.Fatalf("remote=%#v problem=%v", second, problem)
	}
}

func TestWrongAssetMediaRefusesWithoutRetainedComposition(t *testing.T) {
	composer, layout, _ := testComposer(t)
	base := t.TempDir()
	if err := os.WriteFile(filepath.Join(base, "reference.png"), []byte("plain text"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, problem := composer.Compose(composeRequest([]byte(validSource), base))
	if problem == nil || problem.ErrName() != "video_asset_media_type" ||
		strings.Contains(problem.Message, base) {
		t.Fatalf("wrong media problem=%#v", problem)
	}
	entries, err := os.ReadDir(layout.Inputs)
	if err != nil || len(entries) != 0 {
		t.Fatalf("staged entries=%v error=%v", entries, err)
	}
}

func TestMissingAssetErrorDoesNotEchoPrivateBaseDirectory(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	_, problem := composer.Compose(composeRequest([]byte(validSource), base))
	if problem == nil || strings.Contains(problem.Message, base) ||
		!strings.Contains(problem.Message, "reference.png") {
		t.Fatalf("problem=%#v", problem)
	}
}

func TestUnicodePromptRemainsMeaningThroughASCIICanonicalPlan(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 6)
	source := strings.Replace(validSource, "A traveler enters a quiet station.", "旅人が静かな駅に入る。", 1)
	composition, problem := composer.Compose(composeRequest([]byte(source), base))
	if problem != nil {
		t.Fatal(problem)
	}
	plan, _, _, _ := workflow.DecodePlan(composition.WorkflowPlan)
	payload, _ := base64.StdEncoding.DecodeString(plan.Steps[0].PayloadBase64)
	if !bytes.Contains(payload, []byte("旅人が静かな駅に入る。")) {
		t.Fatalf("payload=%s", payload)
	}
}

func TestFullInt64SeedRemainsMeaning(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 11)
	source := strings.Replace(validSource, "seed: 17", "seed: 9223372036854775807", 1)
	composition, problem := composer.Compose(composeRequest([]byte(source), base))
	if problem != nil {
		t.Fatal(problem)
	}
	plan, _, _, _ := workflow.DecodePlan(composition.WorkflowPlan)
	payload, _ := base64.StdEncoding.DecodeString(plan.Steps[0].PayloadBase64)
	if !bytes.Contains(payload, []byte(`"seed":9223372036854775807`)) {
		t.Fatalf("payload=%s", payload)
	}
}

func TestRepeatedSourcePathReusesOneContentIdentity(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 12)
	source := strings.Replace(validSource, "        - image: reference.png",
		"        - image: reference.png\n        - image: reference.png", 1)
	composition, problem := composer.Compose(composeRequest([]byte(source), base))
	if problem != nil || len(composition.Assets) != 2 ||
		composition.Assets[0].Digest != composition.Assets[1].Digest {
		t.Fatalf("composition=%#v problem=%v", composition, problem)
	}
}

func TestFrameFreeT2VAComposesWithoutAnAssetGrant(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 13)
	source := strings.Replace(validSource,
		"    first_last_frame_to_video:\n      first_frame: previous",
		"    first_last_frame_to_video: {}", 1)
	composition, problem := composer.Compose(composeRequest([]byte(source), base))
	if problem != nil {
		t.Fatal(problem)
	}
	plan, _, _, _ := workflow.DecodePlan(composition.WorkflowPlan)
	if len(plan.Steps[1].Assets) != 0 || len(plan.Steps[1].Bindings) != 0 {
		t.Fatalf("frame-free step=%#v", plan.Steps[1])
	}
}

func TestEightShotSourceBecomesEightH3ChildrenAndOneAssemblyChild(t *testing.T) {
	composer, _, _ := testComposer(t)
	base := t.TempDir()
	writePNG(t, filepath.Join(base, "reference.png"), 7)
	var source strings.Builder
	source.WriteString("format: cozy.video/1\nshots:\n")
	source.WriteString("  - id: shot1\n    prompt: Opening shot\n    seed: 1\n")
	source.WriteString("    reference_media_to_video:\n      references:\n        - image: reference.png\n")
	for ordinal := 2; ordinal <= 8; ordinal++ {
		source.WriteString("  - id: shot" + strconv.Itoa(ordinal) + "\n")
		source.WriteString("    prompt: Continued motion\n    seed: " + strconv.Itoa(ordinal) + "\n")
		source.WriteString("    first_last_frame_to_video:\n      first_frame: previous\n")
	}
	source.WriteString("assembly:\n  audio: segments\n")
	composition, problem := composer.Compose(composeRequest([]byte(source.String()), base))
	if problem != nil {
		t.Fatal(problem)
	}
	plan, _, _, problem := workflow.DecodePlan(composition.WorkflowPlan)
	if problem != nil || composition.ShotCount != 8 ||
		len(plan.Steps) != 9 || plan.Steps[8].Entrypoint != "assemble_video" ||
		len(plan.Steps[8].Bindings) != 8 {
		t.Fatalf("composition=%#v plan=%#v problem=%v", composition, plan, problem)
	}
	for index := 1; index < 8; index++ {
		if len(plan.Steps[index].Bindings) != 1 ||
			plan.Steps[index].Bindings[0].PriorStep != index {
			t.Fatalf("continuation step %d = %#v", index+1, plan.Steps[index])
		}
	}
}
