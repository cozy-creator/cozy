package records

import (
	"path/filepath"
	"testing"
)

func TestCountsSeparateServingRequestsAndJobs(t *testing.T) {
	store, problem := Open(filepath.Join(t.TempDir(), "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()

	for _, request := range []Request{
		{ID: "req-serving", IdemKey: "serving-key", BodyDigest: "sha256:serving", Kind: "serving", Payload: []byte("{}")},
		{ID: "job-one", IdemKey: "job-key", BodyDigest: "sha256:job", Kind: "job", Payload: []byte("{}")},
	} {
		if _, fresh, problem := store.Submit(request); problem != nil || !fresh {
			t.Fatalf("submit %s: fresh=%t problem=%v", request.ID, fresh, problem)
		}
	}

	counts, problem := store.Counts()
	if problem != nil {
		t.Fatal(problem)
	}
	for key, want := range map[string]int{
		"requests": 1, "active_requests": 1, "jobs": 1, "active_jobs": 1,
	} {
		if counts[key] != want {
			t.Errorf("initial %s = %d, want %d", key, counts[key], want)
		}
	}

	if problem := store.SettleRequest("req-serving", "succeeded"); problem != nil {
		t.Fatal(problem)
	}
	counts, problem = store.Counts()
	if problem != nil {
		t.Fatal(problem)
	}
	if counts["requests"] != 1 || counts["active_requests"] != 0 ||
		counts["jobs"] != 1 || counts["active_jobs"] != 1 {
		t.Fatalf("settled counts = %#v", counts)
	}
}
