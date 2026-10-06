package machines

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/google/uuid"
)

// updateLocked updates a running machine in place with Run kind: update over its
// cozy.machine.v1 API: local wheels go up with Write, published versions (the source's, or
// the newest) are fetched by the machine, and versions it already runs are not installed
// again. The machine restarts its service on the candidate and rolls it back if it never
// proves ready; this waits for the outcome.
func (h *Host) updateLocked(ctx context.Context, source Source) (*Installed, *exit.Error) {
	launch, problem := h.ensureLocked(ctx, "", nil, true)
	if problem != nil {
		return nil, problem
	}
	pin, problem := h.Pin()
	if problem != nil {
		return nil, problem
	}
	owner, problem := h.ExistingOwner()
	if problem != nil {
		return nil, problem
	}
	client, err := machinev1.Dial(launch.Addr, pin.TLSConfig(), launch.WorkerID, owner.Signer())
	if err != nil {
		return nil, Transport(err)
	}
	defer client.Close()
	running, err := client.Status(ctx)
	if err != nil {
		return nil, Transport(err)
	}
	cohort := machinev1.Cohort{Agent: "bundled"}
	if source.Host != "" {
		cohort.Agent = "explicit" // keep the running agent; a Runtime wheel's bundled one is not taken
	}
	current := true
	for _, item := range []struct {
		name, path, version, running string
		member                       **machinev1.Member
	}{{hostruntime.Distribution, source.RuntimeWheel, source.RuntimeVersion, running.GetRuntime(), &cohort.Runtime},
		{"tensorfs", source.TensorFSWheel, source.TensorFSVersion, running.GetTensorfs(), &cohort.TensorFS}} {
		if item.path == "" {
			version := item.version
			if version == "" {
				if version, problem = NewestPublished(ctx, item.name); problem != nil {
					return nil, problem
				}
			}
			*item.member = &machinev1.Member{Version: version}
			current = current && version == item.running
			continue
		}
		member, problem := writeWheel(ctx, client, item.path)
		if problem != nil {
			return nil, problem
		}
		*item.member, current = member, false
	}
	if !current {
		operation := uuid.NewString()
		outcome, err := client.Update(ctx, operation, cohort, nil)
		if err != nil {
			return nil, exit.Named(exit.Unavailable, "machine.update_observation_lost", "update %s may continue on the machine; observation ended: %s", operation, Transport(err).Message)
		}
		if outcome.GetStatus() != "succeeded" {
			return nil, exit.Named(exit.Failed, "machine.update_failed", "update %s: %s", operation, outcome.GetReason().GetMessage())
		}
	}
	frame, err := client.Status(ctx)
	if err != nil {
		return nil, Transport(err)
	}
	installed := &Installed{InstalledAt: time.Now().UTC(), Pinned: source.pinned(),
		Host:    installedArtifact{Name: "cozy-machine " + frame.GetVersion()},
		Runtime: installedArtifact{Name: hostruntime.Distribution + " " + frame.GetRuntime()}, TensorFS: installedArtifact{Name: "tensorfs " + frame.GetTensorfs()}}
	for _, pair := range []struct {
		file   string
		target *installedArtifact
	}{{source.RuntimeWheel, &installed.Runtime}, {source.TensorFSWheel, &installed.TensorFS}, {source.Host, &installed.Host}} {
		if pair.file != "" {
			pair.target.Name = filepath.Base(pair.file)
			pair.target.SHA256, _ = fileDigest(pair.file)
		}
	}
	if err := h.recordInstalled(*installed); err != nil {
		return nil, exit.Internalf("updated Runtime but cannot retain installation metadata: %s", err)
	}
	return installed, nil
}

// follow brings a machine boot to the Hub's target software unless its owner pinned the files
// it runs. It answers why the machine kept its software, or "": a machine that cannot follow
// keeps serving what it runs.
func (h *Host) follow(ctx context.Context, account *hub.Client) string {
	installed, problem := h.Installed()
	if account == nil || problem != nil || installed == nil || installed.Pinned {
		return ""
	}
	target, problem := account.Software(ctx)
	if problem != nil {
		return "it keeps its software: the Hub's target could not be read: " + problem.Message
	}
	if target.Runtime == "" {
		return ""
	}
	if _, problem := h.updateLocked(ctx, Source{RuntimeVersion: target.Runtime, TensorFSVersion: target.TensorFS}); problem != nil {
		return "it keeps its software: " + problem.Message
	}
	return ""
}

// writeWheel sends one local wheel to the machine with Write and names it for an update.
func writeWheel(ctx context.Context, client *machinev1.Client, path string) (*machinev1.Member, *exit.Error) {
	digest, err := fileDigest(path)
	if err != nil {
		return nil, exit.New(exit.NotFound, "cannot read update wheel: %s", err)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, exit.New(exit.NotFound, "cannot read update wheel: %s", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, exit.New(exit.NotFound, "cannot read update wheel: %s", err)
	}
	member := &machinev1.Member{Wheel: filepath.Base(path), Digest: "sha256:" + digest, Length: uint64(info.Size())}
	if err := client.Write(ctx, member.Digest, member.Length, file); err != nil {
		return nil, Transport(err)
	}
	return member, nil
}

func (h *Host) recordInstalled(installed Installed) error {
	raw, _ := json.MarshalIndent(installed, "", "  ")
	return writePrivate(h.path("installed.json"), raw)
}

// NewestPublished is a distribution's newest release on the package index.
func NewestPublished(ctx context.Context, name string) (string, *exit.Error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://pypi.org/pypi/"+name+"/json", nil)
	if err != nil {
		return "", exit.Internalf("cannot address the package index: %s", err)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return "", exit.Unavailablef("the package index did not answer: %s", err)
	}
	defer response.Body.Close()
	var project struct {
		Info struct {
			Version string `json:"version"`
		} `json:"info"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 32<<20)).Decode(&project) != nil || project.Info.Version == "" {
		return "", exit.New(exit.Unavailable, "the package index has no release of %s", name)
	}
	return project.Info.Version, nil
}
