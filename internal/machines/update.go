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
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/google/uuid"
)

// updateLocked updates a running machine in place with Run kind: update over its
// cozy.machine.v1 API: local wheels go up with Write, published versions are fetched by the
// machine. The machine restarts its service on the candidate and rolls it back if it never
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
	cohort := machinev1.Cohort{Agent: "bundled"}
	if source.Host != "" {
		cohort.Agent = "explicit" // keep the running agent; a Runtime wheel's bundled one is not taken
	}
	for _, item := range []struct {
		name, path string
		member     **machinev1.Member
	}{{hostruntime.Distribution, source.RuntimeWheel, &cohort.Runtime}, {"tensorfs", source.TensorFSWheel, &cohort.TensorFS}} {
		if item.path == "" {
			version, problem := NewestPublished(ctx, item.name)
			if problem != nil {
				return nil, problem
			}
			*item.member = &machinev1.Member{Version: version}
			continue
		}
		member, problem := writeWheel(ctx, client, item.path)
		if problem != nil {
			return nil, problem
		}
		*item.member = member
	}
	operation := uuid.NewString()
	outcome, err := client.Update(ctx, operation, cohort, nil)
	if err != nil {
		return nil, exit.Named(exit.Unavailable, "machine.update_observation_lost", "update %s may continue on the machine; observation ended: %s", operation, Transport(err).Message)
	}
	if outcome.GetStatus() != "succeeded" {
		return nil, exit.Named(exit.Failed, "machine.update_failed", "update %s: %s", operation, outcome.GetReason().GetMessage())
	}
	frame, err := client.Status(ctx)
	if err != nil {
		return nil, Transport(err)
	}
	installed := &Installed{InstalledAt: time.Now().UTC(), HostPinned: source.Host != "",
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
	if err := writePrivate(h.path("installed.json"), raw); err != nil {
		return err
	}
	// Atomically replace the obsolete entry point; never follow its old symlink
	// into a shared CLI executable.
	target := filepath.Join(h.Root(), "usr/local/bin/pod-supervisor")
	file, err := os.CreateTemp(filepath.Dir(target), ".legacy-refusal-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	_, err = file.WriteString("#!/bin/sh\necho 'legacy machine control retired; use the current cozy CLI' >&2\nexit 6\n")
	if err == nil {
		err = file.Chmod(0755)
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(file.Name(), target)
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
