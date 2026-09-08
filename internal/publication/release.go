// Package publication owns explicit checkpoint/release effects outside CLI parsing.
package publication

import (
	"context"
	"maps"
	"regexp"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/tfs"
)

var labelPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+!-]{0,63}$`)

// ReleaseRequest is caller intent. A nil revision reads and freezes the current one.
type ReleaseRequest struct {
	Destination      string            `json:"destination"`
	Release          string            `json:"release"`
	Lanes            map[string]string `json:"lanes"`
	ExpectedRevision *int64            `json:"expected_revision"`
}

// ReleaseIntent is recorded durably before mutation. It never rebases on later state.
type ReleaseIntent struct {
	Request          ReleaseRequest    `json:"request"`
	BaselineRevision int64             `json:"baseline_revision"`
	Baseline         map[string]string `json:"baseline"`
	Desired          map[string]string `json:"desired"`
}

type ReleaseReceipt struct {
	Destination string            `json:"destination"`
	Release     string            `json:"release"`
	Revision    int64             `json:"revision"`
	Lanes       map[string]string `json:"lanes"`
	Observation string            `json:"observation"`
}

type Releases interface {
	ModelRelease(context.Context, hub.Ref, string) (hub.ModelRelease, *exit.Error)
	UpdateModelRelease(context.Context, hub.Ref, string, int64, map[string]string, []string, string) (hub.ModelRelease, *exit.Error)
}

func Label(value string) bool { return labelPattern.MatchString(value) }

func ReleaseLanes(release hub.ModelRelease) (map[string]string, *exit.Error) {
	if release.Revision < 1 || release.Yanked || !Label(release.Release) {
		return nil, exit.Named(exit.Conflict, "publication.release_invalid", "release is not an active positive revision")
	}
	lanes := make(map[string]string, len(release.Lanes))
	for _, lane := range release.Lanes {
		if !Label(lane.Lane) || lanes[lane.Lane] != "" {
			return nil, exit.Named(exit.Conflict, "publication.release_invalid", "release has a repeated or invalid lane")
		}
		if _, problem := tfs.ManifestID(lane.CheckpointID); problem != nil {
			return nil, exit.Named(exit.Conflict, "publication.release_invalid", "release has an invalid checkpoint identity")
		}
		lanes[lane.Lane] = lane.CheckpointID
	}
	if len(lanes) == 0 {
		return nil, exit.Named(exit.Conflict, "publication.release_invalid", "release has no lanes")
	}
	return lanes, nil
}

func readRelease(ctx context.Context, client Releases, ref hub.Ref, label string) (int64, map[string]string, *exit.Error) {
	current, problem := client.ModelRelease(ctx, ref, label)
	if problem != nil {
		if problem.Code == exit.NotFound {
			return 0, map[string]string{}, nil
		}
		return 0, nil, problem
	}
	if current.Release != label {
		return 0, nil, exit.Named(exit.Conflict, "publication.release_invalid", "readback names another release")
	}
	lanes, problem := ReleaseLanes(current)
	return current.Revision, lanes, problem
}

func PrepareRelease(ctx context.Context, client Releases, request ReleaseRequest) (ReleaseIntent, *exit.Error) {
	ref, problem := hub.ParseRef(request.Destination)
	if problem != nil || ref.Org == "local" || !Label(request.Release) || len(request.Lanes) == 0 || len(request.Lanes) > 32 {
		return ReleaseIntent{}, exit.New(exit.Validation, "release publication needs a public destination, label and 1..32 lanes")
	}
	if request.ExpectedRevision != nil && (*request.ExpectedRevision < 0 || *request.ExpectedRevision > (1<<53)-2) {
		return ReleaseIntent{}, exit.New(exit.Validation, "release expected revision is outside its exact integer range")
	}
	request.Lanes = maps.Clone(request.Lanes)
	if request.ExpectedRevision != nil {
		revision := *request.ExpectedRevision
		request.ExpectedRevision = &revision
	}
	for lane, checkpoint := range request.Lanes {
		if !Label(lane) {
			return ReleaseIntent{}, exit.New(exit.Validation, "release lane name is invalid")
		}
		if _, problem := tfs.ManifestID(checkpoint); problem != nil {
			return ReleaseIntent{}, problem
		}
	}
	revision, baseline, problem := readRelease(ctx, client, ref, request.Release)
	if problem != nil {
		return ReleaseIntent{}, problem
	}
	if request.ExpectedRevision != nil && *request.ExpectedRevision != revision {
		return ReleaseIntent{}, exit.Named(exit.Conflict, "publication.release_conflict", "explicit expected revision differs from the current release")
	}
	desired := maps.Clone(baseline)
	maps.Copy(desired, request.Lanes)
	return ReleaseIntent{Request: request, BaselineRevision: revision, Baseline: baseline, Desired: desired}, nil
}

// ApplyRelease reconciles one frozen intent. BeforeSend must durably record that a
// mutation may have been sent; errors leave that uncertainty in the owner's journal.
// Retries may repeat the original CAS, never apply against a newer revision.
func ApplyRelease(ctx context.Context, client Releases, intent ReleaseIntent, sent bool, beforeSend func() *exit.Error) (ReleaseReceipt, *exit.Error) {
	if !Label(intent.Request.Release) || intent.BaselineRevision < 0 || intent.BaselineRevision > (1<<53)-2 || len(intent.Request.Lanes) == 0 {
		return ReleaseReceipt{}, exit.New(exit.Validation, "release effect intent is malformed")
	}
	desired := maps.Clone(intent.Baseline)
	if desired == nil {
		desired = map[string]string{}
	}
	maps.Copy(desired, intent.Request.Lanes)
	if !maps.Equal(desired, intent.Desired) || (intent.Request.ExpectedRevision != nil && *intent.Request.ExpectedRevision != intent.BaselineRevision) {
		return ReleaseReceipt{}, exit.Named(exit.Conflict, "publication.intent_changed", "release effect no longer matches its frozen request and baseline")
	}

	ref, problem := hub.ParseRef(intent.Request.Destination)
	if problem != nil {
		return ReleaseReceipt{}, problem
	}
	revision, lanes, problem := readRelease(ctx, client, ref, intent.Request.Release)
	if problem != nil {
		return ReleaseReceipt{}, problem
	}
	result := func(observation string) ReleaseReceipt {
		return ReleaseReceipt{Destination: ref.String(), Release: intent.Request.Release, Revision: revision, Lanes: maps.Clone(lanes), Observation: observation}
	}
	if revision == intent.BaselineRevision+1 && maps.Equal(lanes, intent.Desired) {
		return result("observed_convergence"), nil
	}
	if revision != intent.BaselineRevision || !maps.Equal(lanes, intent.Baseline) {
		code := "publication.release_conflict"
		if sent {
			code = "publication.outcome_unknown"
		}
		return ReleaseReceipt{}, exit.Named(exit.Conflict, code, "release changed outside the frozen publication intent; no automatic rebase is permitted")
	}
	if maps.Equal(lanes, intent.Desired) {
		return result("observed_noop"), nil
	}
	if beforeSend == nil {
		return ReleaseReceipt{}, exit.Internalf("release effect has no durable pre-send recorder")
	}
	if problem := beforeSend(); problem != nil {
		return ReleaseReceipt{}, problem
	}
	updated, problem := client.UpdateModelRelease(ctx, ref, intent.Request.Release, intent.BaselineRevision, intent.Request.Lanes, nil, "private script release publication")
	if problem != nil {
		return ReleaseReceipt{}, problem
	}
	updatedLanes, problem := ReleaseLanes(updated)
	if problem != nil {
		return ReleaseReceipt{}, problem
	}
	if updated.Release != intent.Request.Release || updated.Revision != intent.BaselineRevision+1 || !maps.Equal(updatedLanes, intent.Desired) {
		return ReleaseReceipt{}, exit.Named(exit.Conflict, "publication.outcome_unknown", "release mutation acknowledgement differs from the frozen expected successor")
	}
	return ReleaseReceipt{Destination: ref.String(), Release: updated.Release, Revision: updated.Revision, Lanes: updatedLanes, Observation: "acknowledged"}, nil
}
