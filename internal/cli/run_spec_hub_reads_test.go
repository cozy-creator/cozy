package cli

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

// Run 6349 spent 47 s before submission walking H3's published closure at the Hub. A published
// composition's spec is its own choices only: its machine gives each child slot the callee's
// default, so building the spec reads nothing at any Hub.
func TestPublishedCompositionSpecReadsNoHub(t *testing.T) {
	var reads atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reads.Add(1)
		t.Errorf("run submission read the Hub: %s %s", r.Method, r.URL.Path)
		http.Error(w, "no Hub per run", http.StatusInternalServerError)
	}))
	defer server.Close()
	root := t.TempDir()
	layout, problem := home.Open(root)
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	cfg := config.Config{Home: root, HubURL: server.URL}
	m := machineRuns{context: &Context{Cfg: cfg}, layout: layout, store: store, resolver: NewResolver(store, cfg)}
	request := records.Request{ID: "film", Kind: "job", Package: "alice/film", Release: "1.36.1", Entrypoint: "long_form"}
	if !request.ComposesChildren() {
		t.Fatal("the request must be a composition root")
	}
	spec, problem := m.specV1(t.Context(), request, &machines.V1{Name: "pod"})
	if problem != nil {
		t.Fatal(problem)
	}
	if len(spec.GetModels()) != 0 || spec.GetRelease().GetPackage() != "alice/film" || reads.Load() != 0 {
		t.Fatalf("spec models=%v release=%v hub reads=%d", spec.GetModels(), spec.GetRelease(), reads.Load())
	}
}
