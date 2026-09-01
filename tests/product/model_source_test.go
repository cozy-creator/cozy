package producttest

import (
	"fmt"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/modelsource"
)

func TestH3SourceSelectionMovesOnlyReviewedCarriersAndTheirShards(t *testing.T) {
	const (
		commit        = "42ed227ee7df40d41602854ae760620d6eb651fe"
		expectedBytes = int64(210_297_121_677)
	)
	selected := []string{
		"FL2VA/transformer/model.safetensors.index.json",
		"Ref2VA/transformer/model.safetensors.index.json",
		"audio_vae/diffusion_pytorch_model.safetensors",
		"text_encoder/model.safetensors.index.json",
		"vae/diffusion_pytorch_model.safetensors.index.json",
	}
	carrierLengths := map[string]int64{
		selected[0]: 38_323,
		selected[1]: 38_323,
		selected[2]: 605_429_340,
		selected[3]: 97_831,
		selected[4]: 74_228,
	}
	files := make([]modelsource.File, 0, 50)
	sequence := 0
	file := func(member string, length int64, carrier bool) modelsource.File {
		sequence++
		return modelsource.File{Member: member, Length: length, Carrier: carrier,
			SHA256: fmt.Sprintf("%064x", sequence),
			URL:    "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/" + commit + "/" + member}
	}
	addShards := func(index string, count int, pattern string) {
		carrier := file(index, carrierLengths[index], true)
		for shard := 1; shard <= count; shard++ {
			member := fmt.Sprintf(pattern, shard)
			carrier.Requires = append(carrier.Requires, member)
			files = append(files, file(member, 1, false))
		}
		files = append(files, carrier)
	}
	addShards(selected[0], 13, "FL2VA/transformer/model-%05d-of-00013.safetensors")
	addShards(selected[1], 13, "Ref2VA/transformer/model-%05d-of-00013.safetensors")
	addShards(selected[3], 14, "text_encoder/model-%05d-of-00014.safetensors")
	addShards(selected[4], 3, "vae/diffusion_pytorch_model-%05d-of-00003.safetensors")
	files = append(files, file(selected[2], carrierLengths[selected[2]], true))

	// Preserve the observed aggregate exactly without embedding 43 irrelevant
	// per-shard sizes in this behavior test.
	var total int64
	for _, candidate := range files {
		total += candidate.Length
	}
	files[0].Length += expectedBytes - total
	files = append(files,
		file("transformer/model.safetensors.index.json", 1, true),
		file("transformer_ref/model.safetensors.index.json", 1, true))

	source := modelsource.Source{Kind: modelsource.HuggingFace,
		Canonical: "hf://MiniMaxAI/MiniMax-H3@" + commit, Org: "MiniMaxAI", Repo: "MiniMax-H3",
		Reference: commit, Revision: commit}
	plan := modelsource.Plan{Source: source, Canonical: source.Canonical, Files: files}
	narrowed, problem := plan.Select(selected)
	if problem != nil {
		t.Fatal(problem)
	}
	if len(narrowed.Files) != 48 || narrowed.Bytes != expectedBytes {
		t.Fatalf("selected %d files / %d bytes, want 48 / %d",
			len(narrowed.Files), narrowed.Bytes, expectedBytes)
	}
	for _, candidate := range narrowed.Files {
		if strings.HasPrefix(candidate.Member, "transformer/") ||
			strings.HasPrefix(candidate.Member, "transformer_ref/") {
			t.Fatalf("unreviewed duplicate carrier survived selection: %s", candidate.Member)
		}
	}
}
