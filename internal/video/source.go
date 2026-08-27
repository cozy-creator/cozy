// Package video owns the editable Cozy Video source and its composition into the
// Creator-owned ordered workflow form. No endpoint, Runtime, or provider package parses
// YAML or learns shot semantics.
package video

import (
	"bytes"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"go.yaml.in/yaml/v3"
)

const (
	SourceFormat   = "cozy.video/1"
	MaxSourceBytes = 1 << 20
	MaxShots       = 8
	MaxPromptRunes = 4096
	maxYAMLDepth   = 32
	maxYAMLNodes   = 8192
	maxYAMLScalar  = 64 << 10
)

var shotID = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_-]{0,63}$`)
var decimalSeed = regexp.MustCompile(`^-?(0|[1-9][0-9]*)$`)

type Source struct {
	Format   SourceString   `yaml:"format"`
	Shots    []ShotSource   `yaml:"shots"`
	Assembly AssemblySource `yaml:"assembly"`
}

type ShotSource struct {
	ID        SourceString          `yaml:"id"`
	Prompt    SourceString          `yaml:"prompt"`
	Seed      *DecimalSeed          `yaml:"seed"`
	Reference *ReferenceMediaSource `yaml:"reference_media_to_video,omitempty"`
	FirstLast *FirstLastSource      `yaml:"first_last_frame_to_video,omitempty"`
}

type ReferenceMediaSource struct {
	References []ReferenceSource `yaml:"references"`
}

type DecimalSeed int64

func (s *DecimalSeed) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind != yaml.ScalarNode || node.Tag != "!!int" ||
		!decimalSeed.MatchString(node.Value) {
		return fmt.Errorf("seed is one explicit decimal int64")
	}
	value, err := strconv.ParseInt(node.Value, 10, 64)
	if err != nil {
		return fmt.Errorf("seed is outside int64: %w", err)
	}
	*s = DecimalSeed(value)
	return nil
}

type ReferenceSource struct {
	Image SourceString `yaml:"image,omitempty"`
	Video SourceString `yaml:"video,omitempty"`
	Audio SourceString `yaml:"audio,omitempty"`
}

type SourceString string

func (s *SourceString) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind != yaml.ScalarNode || node.Tag != "!!str" {
		return fmt.Errorf("value is explicitly a YAML string")
	}
	*s = SourceString(node.Value)
	return nil
}

type FirstLastSource struct {
	FirstFrame FrameSource `yaml:"first_frame,omitempty"`
	LastFrame  FrameSource `yaml:"last_frame,omitempty"`
}

// FrameSource reserves the concise scalar `previous` for continuation while keeping an
// explicit `{asset: previous}` spelling for a real file with that name.
type FrameSource struct {
	Path     string
	Previous bool
}

func (f *FrameSource) UnmarshalYAML(node *yaml.Node) error {
	switch node.Kind {
	case yaml.ScalarNode:
		if node.Tag != "!!str" || node.Value == "" {
			return fmt.Errorf("frame is a non-empty path, `previous`, or {asset: <path>}")
		}
		if node.Value == "previous" {
			f.Previous = true
		} else {
			f.Path = node.Value
		}
		return nil
	case yaml.MappingNode:
		if len(node.Content) != 2 || node.Content[0].Value != "asset" ||
			node.Content[1].Kind != yaml.ScalarNode || node.Content[1].Tag != "!!str" ||
			node.Content[1].Value == "" {
			return fmt.Errorf("explicit frame asset is exactly {asset: <path>}")
		}
		f.Path = node.Content[1].Value
		return nil
	default:
		return fmt.Errorf("frame is a non-empty path, `previous`, or {asset: <path>}")
	}
}

type AssemblySource struct {
	Audio       SourceString `yaml:"audio"`
	MasterAudio SourceString `yaml:"master_audio,omitempty"`
}

func DecodeSource(data []byte) (*Source, *exit.Error) {
	if len(data) == 0 || len(data) > MaxSourceBytes || !utf8.Valid(data) {
		return nil, exit.Named(exit.Validation, "video_source_size",
			"a Cozy Video source is non-empty UTF-8 of at most %d bytes", MaxSourceBytes)
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	var document yaml.Node
	if err := decoder.Decode(&document); err != nil {
		return nil, malformed("the Cozy Video source is not YAML: %s", err)
	}
	nodes := 0
	if problem := auditYAML(&document, 0, &nodes); problem != nil {
		return nil, problem
	}
	var trailing yaml.Node
	if err := decoder.Decode(&trailing); err != io.EOF {
		return nil, malformed("a Cozy Video source contains exactly one YAML document")
	}
	strict := yaml.NewDecoder(bytes.NewReader(data))
	strict.KnownFields(true)
	var source Source
	if err := strict.Decode(&source); err != nil {
		return nil, malformed("the Cozy Video source does not match %s: %s", SourceFormat, err)
	}
	if problem := source.validate(); problem != nil {
		return nil, problem
	}
	return &source, nil
}

func auditYAML(node *yaml.Node, depth int, nodes *int) *exit.Error {
	*nodes++
	if depth > maxYAMLDepth || *nodes > maxYAMLNodes || len(node.Value) > maxYAMLScalar {
		return malformed("the YAML tree exceeds the Cozy Video depth, node, or scalar bound")
	}
	if node.Anchor != "" || node.Kind == yaml.AliasNode || node.Tag == "!!merge" {
		return malformed("YAML aliases, anchors, and merge keys are not part of %s", SourceFormat)
	}
	switch node.Kind {
	case yaml.DocumentNode, yaml.MappingNode, yaml.SequenceNode:
	case yaml.ScalarNode:
		switch node.Tag {
		case "!!str", "!!int", "!!bool", "!!null":
		default:
			return malformed("YAML tag %q is not part of %s", node.Tag, SourceFormat)
		}
	default:
		return malformed("YAML node kind %d is not part of %s", node.Kind, SourceFormat)
	}
	if node.Kind == yaml.MappingNode {
		seen := map[string]bool{}
		for index := 0; index < len(node.Content); index += 2 {
			key := node.Content[index]
			if key.Kind != yaml.ScalarNode || key.Tag != "!!str" || key.Value == "" {
				return malformed("every Cozy Video mapping key is a non-empty string")
			}
			if seen[key.Value] {
				return malformed("YAML key %q appears more than once", key.Value)
			}
			seen[key.Value] = true
		}
	}
	for _, child := range node.Content {
		if problem := auditYAML(child, depth+1, nodes); problem != nil {
			return problem
		}
	}
	return nil
}

func (s *Source) validate() *exit.Error {
	if string(s.Format) != SourceFormat {
		return invalid("format is %q, not %q", s.Format, SourceFormat)
	}
	if len(s.Shots) < 2 || len(s.Shots) > MaxShots {
		return invalid("a Cozy Video has 2-%d shots; this source has %d", MaxShots, len(s.Shots))
	}
	ids := map[string]bool{}
	for index := range s.Shots {
		shot := &s.Shots[index]
		ordinal := index + 1
		id, prompt := string(shot.ID), string(shot.Prompt)
		if !shotID.MatchString(id) || ids[id] {
			return invalid("shot %d id %q is not unique portable identifier", ordinal, shot.ID)
		}
		ids[id] = true
		promptRunes := utf8.RuneCountInString(prompt)
		if strings.TrimSpace(prompt) == "" || promptRunes > MaxPromptRunes {
			return invalid("shot %s prompt is empty or longer than %d characters",
				shot.ID, MaxPromptRunes)
		}
		if shot.Seed == nil {
			return invalid("shot %s omits its explicit seed", shot.ID)
		}
		if (shot.Reference == nil) == (shot.FirstLast == nil) {
			return invalid("shot %s needs exactly one H3 action shape", shot.ID)
		}
		if shot.Reference != nil {
			if problem := validateReferences(id, shot.Reference.References); problem != nil {
				return problem
			}
		}
		if shot.FirstLast != nil {
			if shot.FirstLast.FirstFrame.Previous && ordinal == 1 {
				return invalid("the first shot cannot use first_frame: previous")
			}
			if shot.FirstLast.LastFrame.Previous {
				return invalid("shot %s last_frame cannot be previous", shot.ID)
			}
			for _, frame := range []FrameSource{shot.FirstLast.FirstFrame, shot.FirstLast.LastFrame} {
				if frame.Path != "" && !validSourcePath(frame.Path) {
					return invalid("shot %s contains an unusable frame path", shot.ID)
				}
			}
		}
	}
	audio, master := string(s.Assembly.Audio), string(s.Assembly.MasterAudio)
	if audio != "segments" && audio != "master" ||
		audio == "segments" && master != "" || audio == "master" && master == "" {
		return invalid("assembly audio is segments, or master with one master_audio path")
	}
	if master != "" && !validSourcePath(master) {
		return invalid("assembly master_audio path is unusable")
	}
	return nil
}

func validateReferences(shot string, refs []ReferenceSource) *exit.Error {
	if len(refs) < 1 || len(refs) > 12 {
		return invalid("shot %s has %d references; reference_media_to_video accepts 1-12",
			shot, len(refs))
	}
	images, videos, audios := 0, 0, 0
	for index, ref := range refs {
		values := []string{string(ref.Image), string(ref.Video), string(ref.Audio)}
		present := 0
		for _, value := range values {
			if value != "" {
				present++
				if !validSourcePath(value) {
					return invalid("shot %s reference %d path is unusable", shot, index)
				}
			}
		}
		if present != 1 {
			return invalid("shot %s reference %d needs exactly one image, video, or audio", shot, index)
		}
		images += map[bool]int{true: 1}[ref.Image != ""]
		videos += map[bool]int{true: 1}[ref.Video != ""]
		audios += map[bool]int{true: 1}[ref.Audio != ""]
	}
	if images > 9 || videos > 3 || audios > 3 || images+videos == 0 {
		return invalid("shot %s references resolve to %d images, %d videos, %d audio; "+
			"limits are 9/3/3 and audio-only is invalid", shot, images, videos, audios)
	}
	return nil
}

func validSourcePath(path string) bool {
	return strings.TrimSpace(path) == path && path != "" && len(path) <= 4096 &&
		!strings.ContainsRune(path, 0)
}

func malformed(format string, args ...any) *exit.Error {
	return exit.Named(exit.Validation, "video_source_malformed", format, args...)
}

func invalid(format string, args ...any) *exit.Error {
	return exit.Named(exit.Validation, "video_source_invalid", format, args...)
}
