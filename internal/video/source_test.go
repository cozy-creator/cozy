package video

import (
	"strings"
	"testing"
)

const validSource = `format: cozy.video/1
shots:
  - id: opening
    prompt: |
      A traveler enters a quiet station.
    seed: 17
    reference_media_to_video:
      references:
        - image: reference.png
  - id: continuation
    prompt: The traveler boards the train.
    seed: -2
    first_last_frame_to_video:
      first_frame: previous
assembly:
  audio: segments
`

func TestDecodeSourceStrictValidShape(t *testing.T) {
	source, problem := DecodeSource([]byte(validSource))
	if problem != nil || len(source.Shots) != 2 || source.Shots[0].Seed == nil ||
		int64(*source.Shots[0].Seed) != 17 || !source.Shots[1].FirstLast.FirstFrame.Previous {
		t.Fatalf("source=%#v problem=%v", source, problem)
	}
}

func TestExplicitAssetCanNamePrevious(t *testing.T) {
	text := strings.Replace(validSource, "first_frame: previous", "first_frame: {asset: previous}", 1)
	source, problem := DecodeSource([]byte(text))
	if problem != nil || source.Shots[1].FirstLast.FirstFrame.Previous ||
		source.Shots[1].FirstLast.FirstFrame.Path != "previous" {
		t.Fatalf("frame=%#v problem=%v", source.Shots[1].FirstLast.FirstFrame, problem)
	}
}

func TestSourceRejectsYAMLAmbiguityAndUnknownFields(t *testing.T) {
	arms := map[string]string{
		"duplicate root": strings.Replace(validSource, "format: cozy.video/1",
			"format: cozy.video/1\nformat: cozy.video/1", 1),
		"duplicate nested": strings.Replace(validSource, "seed: 17", "seed: 17\n    seed: 18", 1),
		"unknown":          strings.Replace(validSource, "seed: 17", "seed: 17\n    duration: 15", 1),
		"alias":            strings.Replace(validSource, "prompt: |", "prompt: &shared |", 1) + "x: *shared\n",
		"merge":            strings.Replace(validSource, "seed: 17", "<<: {seed: 17}\n    seed: 17", 1),
		"custom tag":       strings.Replace(validSource, "seed: 17", "seed: !custom 17", 1),
		"second document":  validSource + "---\nformat: cozy.video/1\n",
		"hex seed":         strings.Replace(validSource, "seed: 17", "seed: 0x11", 1),
		"leading seed":     strings.Replace(validSource, "seed: 17", "seed: 017", 1),
		"boolean id":       strings.Replace(validSource, "id: opening", "id: true", 1),
		"numeric prompt":   strings.Replace(validSource, "prompt: |\n      A traveler enters a quiet station.", "prompt: 123", 1),
		"numeric path":     strings.Replace(validSource, "image: reference.png", "image: 17", 1),
		"boolean master": strings.Replace(validSource, "audio: segments",
			"audio: master\n  master_audio: false", 1),
	}
	for name, text := range arms {
		t.Run(name, func(t *testing.T) {
			if _, problem := DecodeSource([]byte(text)); problem == nil {
				t.Fatal("ambiguous source accepted")
			}
		})
	}
}

func TestFrameFreeFirstLastActionIsValid(t *testing.T) {
	text := strings.Replace(validSource, "first_frame: previous", "{}", 1)
	text = strings.Replace(text, "first_last_frame_to_video:\n      {}",
		"first_last_frame_to_video: {}", 1)
	source, problem := DecodeSource([]byte(text))
	if problem != nil || source.Shots[1].FirstLast == nil {
		t.Fatalf("source=%#v problem=%v", source, problem)
	}
}

func TestSourceRejectsInvalidVideoShapes(t *testing.T) {
	arms := map[string]string{
		"missing seed": strings.Replace(validSource, "    seed: 17\n", "", 1),
		"duplicate id": strings.Replace(validSource, "id: continuation", "id: opening", 1),
		"both actions": strings.Replace(validSource, "    reference_media_to_video:",
			"    first_last_frame_to_video:\n      first_frame: reference.png\n    reference_media_to_video:", 1),
		"first previous": strings.Replace(validSource,
			"reference_media_to_video:\n      references:\n        - image: reference.png",
			"first_last_frame_to_video:\n      first_frame: previous", 1),
		"last previous": strings.Replace(validSource, "first_frame: previous",
			"first_frame: previous\n      last_frame: previous", 1),
		"audio only":     strings.Replace(validSource, "- image: reference.png", "- audio: voice.wav", 1),
		"bad audio mode": strings.Replace(validSource, "audio: segments", "audio: false", 1),
		"two audio modes": strings.Replace(validSource, "audio: segments",
			"audio: segments\n  master_audio: music.wav", 1),
	}
	for name, text := range arms {
		t.Run(name, func(t *testing.T) {
			if _, problem := DecodeSource([]byte(text)); problem == nil {
				t.Fatal("invalid video shape accepted")
			}
		})
	}
}

func TestSourceBoundsYAMLTreeBeforeTypedDecode(t *testing.T) {
	deep := "format: cozy.video/1\nshots:\n" + strings.Repeat("  - x:\n", maxYAMLDepth+4)
	if _, problem := DecodeSource([]byte(deep)); problem == nil {
		t.Fatal("over-deep YAML accepted")
	}
	large := "format: cozy.video/1\nvalue: " + strings.Repeat("x", maxYAMLScalar+1)
	if _, problem := DecodeSource([]byte(large)); problem == nil {
		t.Fatal("oversized scalar accepted")
	}
}
