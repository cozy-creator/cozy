// Package modelproduction owns the stable identity of one source-to-release
// instruction. It deliberately contains no scheduler, provider capability, URL,
// worker, rental, grant, clock, or retry field.
package modelproduction

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"

	"github.com/cozy-creator/cozy/internal/launch"
)

type JobPin struct {
	Step          string
	Callable      string
	InstallID     string
	Release       string
	ReleaseDigest string
	DescriptorID  string
}

type SourceFile struct {
	Member string `json:"member"`
	SHA256 string `json:"sha256"`
	Length int64  `json:"length"`
}

// Instruction is the canonical caller intent recorded before any mutable package
// selector is resolved. Its identity deliberately excludes every resolved release,
// descriptor, source inventory, worker, rental, price, credential, and attempt fact.
type Instruction struct {
	Destination string `json:"destination"`
	Source      string `json:"source"`
	InputLane   string `json:"input_lane,omitempty"`
	Producer    string `json:"producer"`
	Rental      bool   `json:"rental"`
}

func (i Instruction) Bytes() ([]byte, error) { return json.Marshal(i) }

func ParseInstruction(data []byte) (Instruction, error) {
	var instruction Instruction
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&instruction); err != nil {
		return instruction, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return instruction, fmt.Errorf("model production instruction carries trailing JSON")
	}
	canonical, err := instruction.Bytes()
	if err != nil || !bytes.Equal(canonical, data) {
		return instruction, fmt.Errorf("model production instruction is not the one canonical spelling")
	}
	return instruction, nil
}

func (i Instruction) Digest() (string, error) {
	data, err := i.Bytes()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func (i Instruction) ID() string {
	data, _ := i.Bytes()
	sum := sha256.Sum256(append([]byte("cozy-model-upload-instruction/1\x00"), data...))
	return "modelupload-" + hex.EncodeToString(sum[:])
}

type Plan struct {
	Instruction       Instruction
	Destination       string
	Source            string
	SourceSelection   string
	SourceLicense     string
	SourceFiles       []SourceFile
	InputLane         string
	Producer          string
	ProducerInstallID string
	ProducerRelease   string
	ProducerDigest    string
	DescriptorDigest  string
	Production        *launch.ModelProduction
	Jobs              []JobPin
	Resources         ResourceNeeds
}

type ResourceNeeds struct {
	GPUCount int64 `json:"gpu_count"`
	MinSM    int64 `json:"min_sm"`
	VRAMGB   int64 `json:"vram_gb"`
	RAMGB    int64 `json:"ram_gb"`
}

// Bytes is the restart record. It contains only immutable identities and
// reviewed declarations; URLs, credentials, grants, workers, rentals, prices,
// clocks, and attempts have no field.
func (p Plan) Bytes() ([]byte, error) { return json.Marshal(p) }

func Parse(data []byte) (Plan, error) {
	var plan Plan
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&plan); err != nil {
		return plan, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return plan, fmt.Errorf("model production plan carries trailing JSON")
	}
	canonical, err := plan.Bytes()
	if err != nil || !bytes.Equal(canonical, data) {
		return plan, fmt.Errorf("model production plan is not the one canonical spelling")
	}
	return plan, nil
}

func (p Plan) Digest() (string, error) {
	data, err := p.Bytes()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// ID is stable across detach/follow, attempts, rental replacement, capability
// refresh, pricing, and replay. Descriptor identity already binds step edges,
// assets, resources, required output names, and required contracts.
func (p Plan) ID() string {
	if p.Instruction.Destination != "" {
		return p.Instruction.ID()
	}
	hash := sha256.New()
	for _, value := range []string{
		"cozy-model-upload/1", p.Destination, p.Source,
		p.SourceSelection, p.SourceLicense, p.InputLane, p.Producer,
		p.ProducerInstallID, p.ProducerRelease, p.ProducerDigest,
		p.DescriptorDigest,
	} {
		_, _ = io.WriteString(hash, value)
		_, _ = hash.Write([]byte{0})
	}
	if p.Production != nil {
		_, _ = io.WriteString(hash, p.Production.Name)
		_, _ = hash.Write([]byte{0})
	}
	jobs := append([]JobPin(nil), p.Jobs...)
	sort.Slice(jobs, func(i, j int) bool { return jobs[i].Step < jobs[j].Step })
	for _, job := range jobs {
		for _, value := range []string{
			job.Step, job.Callable, job.InstallID, job.Release, job.ReleaseDigest,
			job.DescriptorID,
		} {
			_, _ = io.WriteString(hash, value)
			_, _ = hash.Write([]byte{0})
		}
	}
	for _, value := range []int64{p.Resources.GPUCount, p.Resources.MinSM,
		p.Resources.VRAMGB, p.Resources.RAMGB} {
		_, _ = io.WriteString(hash, fmt.Sprint(value))
		_, _ = hash.Write([]byte{0})
	}
	return "modelupload-" + hex.EncodeToString(hash.Sum(nil))
}

func (p Plan) OutputNames() []string {
	if p.Production == nil {
		return nil
	}
	names := make([]string, 0, len(p.Production.Outputs))
	for _, output := range p.Production.Outputs {
		names = append(names, output.Name)
	}
	sort.Strings(names)
	return names
}

func (p Plan) SourceProfiles() []string {
	if p.Production == nil {
		return nil
	}
	profiles := make([]string, 0, len(p.Production.Sources))
	for _, profile := range p.Production.Sources {
		profiles = append(profiles, profile)
	}
	sort.Strings(profiles)
	return profiles
}
