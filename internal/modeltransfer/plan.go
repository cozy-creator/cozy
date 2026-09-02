// Package modeltransfer owns the stable identity of one source-to-destination request.
// instruction. It deliberately contains no scheduler, provider capability, URL,
// worker, rental, grant, clock, or retry field.
package modeltransfer

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

type JobPin struct {
	Callable      string `json:"callable"`
	Package       string `json:"package"`
	Function      string `json:"function"`
	InstallID     string `json:"install_id,omitempty"`
	Release       string `json:"release"`
	ReleaseDigest string `json:"release_digest"`
	DescriptorID  string `json:"descriptor_id"`
}

type OutputPin struct {
	Name string `json:"name"`
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
	Kind        string `json:"kind"`
	Destination string `json:"destination"`
	Source      string `json:"source"`
	InputLane   string `json:"input_lane,omitempty"`
	Producer    string `json:"producer,omitempty"`
	Placement   string `json:"placement,omitempty"`
	// SourceProfiles is the caller's slot=profile narrowing for a producer whose
	// descriptor declares none. Caller intent, so it is identity; a declared
	// profile is descriptor-derived and stays out.
	SourceProfiles map[string]string `json:"source_profiles,omitempty"`
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
		return instruction, fmt.Errorf("model transfer instruction carries trailing JSON")
	}
	canonical, err := instruction.Bytes()
	if err != nil || !bytes.Equal(canonical, data) {
		return instruction, fmt.Errorf("model transfer instruction is not the one canonical spelling")
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
	sum := sha256.Sum256(append([]byte("cozy-model-transfer-instruction/1\x00"), data...))
	return "modeltransfer-" + hex.EncodeToString(sum[:])
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
	Job               *JobPin
	SourceProfiles    map[string]string
	Outputs           []OutputPin
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
		return plan, fmt.Errorf("model transfer plan carries trailing JSON")
	}
	canonical, err := plan.Bytes()
	if err != nil || !bytes.Equal(canonical, data) {
		return plan, fmt.Errorf("model transfer plan is not the one canonical spelling")
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
// assets, required output names, and required contracts.
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
	if p.Job != nil {
		job := *p.Job
		for _, value := range []string{
			job.Callable, job.Package, job.Function, job.InstallID, job.Release, job.ReleaseDigest,
			job.DescriptorID,
		} {
			_, _ = io.WriteString(hash, value)
			_, _ = hash.Write([]byte{0})
		}
	}
	return "modeltransfer-" + hex.EncodeToString(hash.Sum(nil))
}

func (p Plan) OutputNames() []string {
	names := make([]string, 0, len(p.Outputs))
	for _, output := range p.Outputs {
		names = append(names, output.Name)
	}
	sort.Strings(names)
	return names
}

func (p Plan) ProfileNames() []string {
	profiles := make([]string, 0, len(p.SourceProfiles))
	for _, profile := range p.SourceProfiles {
		profiles = append(profiles, profile)
	}
	sort.Strings(profiles)
	return profiles
}
