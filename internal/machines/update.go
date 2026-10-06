package machines

import (
	"context"
	"encoding/json"
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
// current installed member) are fetched by the machine, and versions it already runs are not installed
// again. The machine restarts its service on the candidate and rolls it back if it never
// proves ready; this waits for the outcome.
func (h *Host) updateLocked(ctx context.Context, source Source) (*Installed, *exit.Error) {
	launch, problem := h.ensureLocked(ctx, nil, true)
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
				version = item.running // an omitted member keeps its installed version
				if version == "" {
					return nil, exit.New(exit.Validation, "name an exact %s version or wheel", item.name)
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
// it runs. It answers whether the software changed, else why not, and whether that was a
// failure: a machine that cannot follow keeps serving what it runs.
func (h *Host) follow(ctx context.Context, account *hub.Client) (changed bool, why string, failed bool) {
	before, problem := h.Installed()
	switch {
	case account == nil || problem != nil || before == nil:
		return false, "", false
	case before.Pinned:
		return false, "it runs the files it was installed from", false
	}
	target, problem := account.Software(ctx)
	if problem != nil {
		return false, "it keeps its software: the Hub's target could not be read: " + problem.Message, true
	}
	if target.Runtime == "" {
		return false, "the Hub names no target software", false
	}
	after, problem := h.updateLocked(ctx, Source{RuntimeVersion: target.Runtime, TensorFSVersion: target.TensorFS})
	if problem != nil {
		return false, "it keeps its software: " + problem.Message, true
	}
	if after.Runtime.Name == before.Runtime.Name && after.TensorFS.Name == before.TensorFS.Name {
		return false, "it already runs the Hub's target software", false
	}
	return true, "", false
}

// FollowTarget brings the running machine to the Hub's target software, as a boot does: it
// answers whether the software changed, else why not.
func (h *Host) FollowTarget(ctx context.Context, account *hub.Client) (bool, string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return false, problem.Message
	}
	defer unlock()
	changed, why, _ := h.follow(ctx, account)
	return changed, why
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
