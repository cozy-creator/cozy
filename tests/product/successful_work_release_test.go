package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func successReleaseStore(t *testing.T) *records.Store {
	t.Helper()
	s, p := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, p)
	t.Cleanup(func() { s.Close() })
	fatal(t, s.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	return s
}

func successReleaseRoot(t *testing.T, s *records.Store, id string, armed, output bool) records.Request {
	t.Helper()
	r, _, p := s.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: childDigest("a"), PlanID: childDigest("b"), Package: "local/test", Entrypoint: "main", Kind: "job", RetainWork: true, ReleaseImplicitWork: armed, ChildArtifacts: output, Payload: []byte(`{}`)})
	fatal(t, p)
	return r
}

func TestSuccessfulReleaseRequiresNewNoArtifactRootAndExactSuccess(t *testing.T) {
	for _, mode := range []string{"new-none", "legacy-unknown", "returned-model", "failed", "paused", "active"} {
		t.Run(mode, func(t *testing.T) {
			s := successReleaseStore(t)
			r := successReleaseRoot(t, s, "root-"+mode, mode != "legacy-unknown", mode == "returned-model")
			r = offerChildParent(t, s, r)
			if mode != "active" {
				status, state := "SUCCEEDED", "succeeded"
				if mode == "failed" {
					status, state = "FAILED", "blocked"
				}
				if mode == "paused" {
					status, state = "CANCELED", "paused"
				}
				closeChild(t, s, r, status, state)
			}
			current, p := s.RequestRow(r.ID)
			fatal(t, p)
			a, p := s.AttemptRow(r.ID, 1)
			fatal(t, p)
			started, p := s.BeginSuccessfulWorkRelease(*current, *a)
			fatal(t, p)
			if started != (mode == "new-none") {
				t.Fatalf("%s started=%v", mode, started)
			}
			intent, p := s.SuccessfulWorkRelease(r.ID)
			fatal(t, p)
			if (mode == "legacy-unknown" || mode == "returned-model") && intent != nil {
				t.Fatal("unverified or explicit-output root was armed")
			}
			if !current.RetainWork {
				t.Fatal("eligibility changed retention before native drain")
			}
		})
	}
}

func TestSuccessfulReleasePreservesHistoryIdentityAndRestartedIntent(t *testing.T) {
	s := successReleaseStore(t)
	submitted := successReleaseRoot(t, s, "successful-root", true, false)
	root := offerChildParent(t, s, submitted)
	child, _, p := s.SubmitChild(records.Request{ID: "successful-child", IdemKey: "successful-child", BodyDigest: childDigest("c"), Package: "local/test", Entrypoint: "operation", Kind: "job", Payload: []byte(`{}`), ParentRequestID: root.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("d"), ChildTargetDigest: childDigest("e")}, 1, childDigest("1"), "private-boot", nil)
	fatal(t, p)
	closeChild(t, s, child, "SUCCEEDED", "succeeded")
	closeChild(t, s, root, "SUCCEEDED", "succeeded")
	current, p := s.RequestRow(root.ID)
	fatal(t, p)
	a, p := s.AttemptRow(root.ID, 1)
	fatal(t, p)
	started, p := s.BeginSuccessfulWorkRelease(*current, *a)
	fatal(t, p)
	if !started {
		t.Fatal("verified success did not start")
	}
	changed := *a
	changed.InvocationDigest = childDigest("f")
	started, p = s.BeginSuccessfulWorkRelease(*current, changed)
	fatal(t, p)
	if started {
		t.Fatal("changed attempt matched release authority")
	}
	family, p := s.SuccessfulWorkFamily(root.ID)
	fatal(t, p)
	if len(family) != 2 || family[0].ID != child.ID || family[1].ID != root.ID {
		t.Fatalf("family order: %+v", family)
	}
	if _, p := s.ReadySuccessfulWorkRelease(root.ID, family[:1]); p == nil {
		t.Fatal("partial membership could discard an unvisited root")
	}
	ready, p := s.ReadySuccessfulWorkRelease(root.ID, family)
	fatal(t, p)
	if !ready {
		t.Fatal("drained family did not cross durable release cut")
	}
	// Reading after the phase cut is the restart input; no in-memory worker owns it.
	intent, p := s.SuccessfulWorkRelease(root.ID)
	fatal(t, p)
	if intent.State != "release_work" || intent.Attempt != 1 || intent.Invocation != a.InvocationDigest || intent.TerminalDigest != a.TerminalDigest || len(intent.Members) != 2 {
		t.Fatalf("release intent lost original identity: %+v", intent)
	}
	for _, id := range []string{root.ID, child.ID} {
		r, p := s.RequestRow(id)
		fatal(t, p)
		if r.State != "succeeded" || r.RetainWork {
			t.Fatalf("release rewrote execution or kept implicit ownership: %+v", r)
		}
	}
	// Duplicate admission still compares the original true retention intent.
	duplicate, fresh, p := s.Submit(submitted)
	fatal(t, p)
	if fresh || duplicate.ID != root.ID || duplicate.State != "succeeded" {
		t.Fatal("completion broke request idempotency")
	}
	fatal(t, s.CompleteSuccessfulWorkRelease(root.ID))
	fatal(t, s.CompleteSuccessfulWorkRelease(root.ID))
	queued, p := s.PendingSuccessfulWorkReleases()
	fatal(t, p)
	if len(queued) != 0 {
		t.Fatalf("completed cleanup remained queued: %v", queued)
	}
	last, p := s.AttemptRow(root.ID, 1)
	fatal(t, p)
	if last.TerminalStatus != a.TerminalStatus || last.TerminalDigest != a.TerminalDigest || last.TerminalID != a.TerminalID || last.ClosedAt != a.ClosedAt {
		t.Fatal("successful cleanup changed execution history")
	}
}
