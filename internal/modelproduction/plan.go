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
	Node          string
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

type Plan struct {
	Destination       string
	Release           string
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
// refresh, pricing, and replay. Descriptor identity already binds node edges,
// assets, resources, required outputs, lane keys, and required contracts.
func (p Plan) ID() string {
	hash := sha256.New()
	for _, value := range []string{
		"cozy-model-production/1", p.Destination, p.Release, p.Source,
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
	sort.Slice(jobs, func(i, j int) bool { return jobs[i].Node < jobs[j].Node })
	for _, job := range jobs {
		for _, value := range []string{
			job.Node, job.Callable, job.InstallID, job.Release, job.ReleaseDigest,
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
	return "modelpub-" + hex.EncodeToString(hash.Sum(nil))
}

func (p Plan) Lanes() []string {
	if p.Production == nil {
		return nil
	}
	lanes := make([]string, 0, len(p.Production.Outputs))
	for _, output := range p.Production.Outputs {
		lanes = append(lanes, output.LaneKey)
	}
	sort.Strings(lanes)
	return lanes
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
