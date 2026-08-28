package video

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

const CreativePlanFormat = "cozy.video.CreativePlan/1"

type Composition struct {
	SourceDigest       string                     `json:"source_digest"`
	CreativePlanDigest string                     `json:"creative_plan_digest"`
	CreativePlan       json.RawMessage            `json:"creative_plan"`
	WorkflowPlan       json.RawMessage            `json:"workflow_plan"`
	Assets             []records.CompositionAsset `json:"assets"`
	ShotCount          int                        `json:"shot_count"`
}

type creativePlan struct {
	Format   string           `json:"format"`
	Shots    []creativeShot   `json:"shots"`
	Assembly creativeAssembly `json:"assembly"`
}

type creativeShot struct {
	PromptBase64 string          `json:"prompt_b64"`
	SeedDecimal  string          `json:"seed_decimal"`
	Action       string          `json:"action"`
	References   []creativeAsset `json:"references"`
	Previous     bool            `json:"previous"`
	FirstAssets  []creativeAsset `json:"first_assets"`
	LastAssets   []creativeAsset `json:"last_assets"`
}

type creativeAssembly struct {
	Audio        string          `json:"audio"`
	MasterAssets []creativeAsset `json:"master_assets"`
}

type creativeAsset struct {
	Digest    string `json:"digest"`
	Length    int64  `json:"length"`
	MediaType string `json:"media_type"`
	Kind      string `json:"kind"`
}

func encodeCreative(plan creativePlan) ([]byte, string, *exit.Error) {
	data, err := canonical.Write(plan.document())
	if err != nil {
		return nil, "", exit.Internalf("cannot canonicalize the creative plan: %s", err)
	}
	digest, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return nil, "", exit.Internalf("cannot spell the creative-plan digest: %s", err)
	}
	return data, digest, nil
}

func decodeCreative(data []byte, wantDigest string) (*creativePlan, *exit.Error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var plan creativePlan
	if err := decoder.Decode(&plan); err != nil {
		return nil, exit.Named(exit.Structural, "video_creative_plan_corrupt",
			"stored creative-plan bytes are unreadable: %s", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF || plan.Format != CreativePlanFormat {
		return nil, exit.Named(exit.Structural, "video_creative_plan_corrupt",
			"stored creative-plan bytes are not one %s document", CreativePlanFormat)
	}
	if problem := plan.validate(); problem != nil {
		return nil, problem
	}
	canonicalBytes, digest, problem := encodeCreative(plan)
	if problem != nil {
		return nil, problem
	}
	if !bytes.Equal(canonicalBytes, data) || digest != wantDigest {
		return nil, exit.Named(exit.Conflict, "video_creative_plan_changed",
			"stored creative-plan bytes no longer match %s", wantDigest)
	}
	return &plan, nil
}

func (p creativePlan) validate() *exit.Error {
	if p.Format != CreativePlanFormat || len(p.Shots) < 2 || len(p.Shots) > MaxShots {
		return exit.Named(exit.Structural, "video_creative_plan_corrupt",
			"creative plan has an invalid format or shot count")
	}
	for index, shot := range p.Shots {
		seed, err := strconv.ParseInt(shot.SeedDecimal, 10, 64)
		if err != nil || strconv.FormatInt(seed, 10) != shot.SeedDecimal {
			return corruptShot(index)
		}
		prompt, err := base64.StdEncoding.Strict().DecodeString(shot.PromptBase64)
		if err != nil || !utf8.Valid(prompt) || strings.TrimSpace(string(prompt)) == "" ||
			utf8.RuneCount(prompt) > MaxPromptRunes {
			return exit.Named(exit.Structural, "video_creative_plan_corrupt",
				"creative shot %d has an invalid prompt", index+1)
		}
		switch shot.Action {
		case "reference_media_to_video":
			if len(shot.References) < 1 || len(shot.References) > 12 || shot.Previous ||
				len(shot.FirstAssets) != 0 || len(shot.LastAssets) != 0 {
				return corruptShot(index)
			}
			images, videos, audios := 0, 0, 0
			for _, asset := range shot.References {
				if !validCreativeAsset(asset) {
					return corruptShot(index)
				}
				images += map[bool]int{true: 1}[asset.Kind == "image"]
				videos += map[bool]int{true: 1}[asset.Kind == "video"]
				audios += map[bool]int{true: 1}[asset.Kind == "audio"]
			}
			if images > 9 || videos > 3 || audios > 3 || images+videos == 0 {
				return corruptShot(index)
			}
		case "first_last_frame_to_video":
			if len(shot.References) != 0 || len(shot.FirstAssets) > 1 || len(shot.LastAssets) > 1 {
				return corruptShot(index)
			}
			if shot.Previous && index == 0 || shot.Previous && len(shot.FirstAssets) != 0 {
				return corruptShot(index)
			}
			for _, asset := range append(append([]creativeAsset{}, shot.FirstAssets...), shot.LastAssets...) {
				if asset.Kind != "image" || !validCreativeAsset(asset) {
					return corruptShot(index)
				}
			}
		default:
			return corruptShot(index)
		}
	}
	if p.Assembly.Audio == "segments" {
		if len(p.Assembly.MasterAssets) != 0 {
			return corruptAssembly()
		}
	} else if p.Assembly.Audio == "master" {
		if len(p.Assembly.MasterAssets) != 1 || p.Assembly.MasterAssets[0].Kind != "audio" ||
			!validCreativeAsset(p.Assembly.MasterAssets[0]) {
			return corruptAssembly()
		}
	} else {
		return corruptAssembly()
	}
	return nil
}

func validCreativeAsset(asset creativeAsset) bool {
	_, err := canonical.Raw(asset.Digest)
	prefix, _, _ := strings.Cut(asset.MediaType, "/")
	return err == nil && asset.Length > 0 && prefix == asset.Kind &&
		(asset.Kind == "image" || asset.Kind == "video" || asset.Kind == "audio")
}

func corruptShot(index int) *exit.Error {
	return exit.Named(exit.Structural, "video_creative_plan_corrupt",
		"creative shot %d has an invalid action shape", index+1)
}

func corruptAssembly() *exit.Error {
	return exit.Named(exit.Structural, "video_creative_plan_corrupt",
		"creative assembly has an invalid audio shape")
}

func expectedCompositionAssets(plan creativePlan) []records.CompositionAsset {
	var out []records.CompositionAsset
	add := func(step int, field string, order uint32, asset creativeAsset) {
		out = append(out, records.CompositionAsset{Step: step, FieldPath: field,
			Digest: asset.Digest, Length: asset.Length, MediaType: asset.MediaType,
			Kind: asset.Kind, Order: order})
	}
	for index, shot := range plan.Shots {
		step := index + 1
		for order, asset := range shot.References {
			add(step, "references."+strconv.Itoa(order)+"."+asset.Kind, uint32(order), asset)
		}
		if len(shot.FirstAssets) == 1 {
			add(step, "first_frame", 0, shot.FirstAssets[0])
		}
		if len(shot.LastAssets) == 1 {
			add(step, "last_frame", 0, shot.LastAssets[0])
		}
	}
	if len(plan.Assembly.MasterAssets) == 1 {
		add(len(plan.Shots)+1, "master_audio", 0, plan.Assembly.MasterAssets[0])
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Step != out[j].Step {
			return out[i].Step < out[j].Step
		}
		return out[i].FieldPath < out[j].FieldPath
	})
	return out
}

func (p creativePlan) document() map[string]canonical.Value {
	shots := make([]canonical.Value, 0, len(p.Shots))
	for _, shot := range p.Shots {
		shots = append(shots, shot.document())
	}
	return map[string]canonical.Value{
		"format": CreativePlanFormat, "shots": shots, "assembly": p.Assembly.document(),
	}
}

func (s creativeShot) document() map[string]canonical.Value {
	references := make([]canonical.Value, 0, len(s.References))
	for _, asset := range s.References {
		references = append(references, asset.document())
	}
	first := make([]canonical.Value, 0, len(s.FirstAssets))
	for _, asset := range s.FirstAssets {
		first = append(first, asset.document())
	}
	last := make([]canonical.Value, 0, len(s.LastAssets))
	for _, asset := range s.LastAssets {
		last = append(last, asset.document())
	}
	return map[string]canonical.Value{
		"prompt_b64": s.PromptBase64, "seed_decimal": s.SeedDecimal, "action": s.Action,
		"references": references, "previous": s.Previous,
		"first_assets": first, "last_assets": last,
	}
}

func (a creativeAssembly) document() map[string]canonical.Value {
	master := make([]canonical.Value, 0, len(a.MasterAssets))
	for _, asset := range a.MasterAssets {
		master = append(master, asset.document())
	}
	return map[string]canonical.Value{"audio": a.Audio, "master_assets": master}
}

func (a creativeAsset) document() map[string]canonical.Value {
	return map[string]canonical.Value{
		"digest": a.Digest, "length": a.Length, "media_type": a.MediaType,
		"kind": a.Kind,
	}
}
