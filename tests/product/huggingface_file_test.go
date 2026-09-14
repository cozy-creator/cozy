package modelsource

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHuggingFaceFileURLsPreserveSelectionAcrossCanonicalRoundTrip(t *testing.T) {
	commit := strings.Repeat("a", 40)
	for _, raw := range []string{
		"https://huggingface.co/owner/model/blob/main/adapters/my%20lora.safetensors?download=true",
		"https://huggingface.co/owner/model/resolve/" + commit + "/adapters/my%20lora.safetensors",
		"hf://owner/model@" + commit + "/adapters/my%20lora.safetensors",
	} {
		source, problem := Parse(raw, t.TempDir())
		if problem != nil {
			t.Fatal(problem)
		}
		again, problem := Parse(source.Canonical, t.TempDir())
		if problem != nil || again != source {
			t.Fatalf("lost selection: %#v -> %#v: %v", source, again, problem)
		}
		if source.Member != "adapters/my lora.safetensors" {
			t.Fatal(source)
		}
	}
	for _, suffix := range []string{"tool.exe", "../bad.safetensors", "%2e%2e/bad.safetensors", "foo%5cbar.safetensors", "foo%00.safetensors", "", "model.safetensors/"} {
		_, problem := Parse("https://huggingface.co/owner/model/blob/main/"+suffix, t.TempDir())
		if problem == nil {
			t.Fatalf("accepted unsafe/non-carrier %q", suffix)
		}
	}
}

func TestHuggingFaceExplicitFileDoesNotFetchOtherCheckpointsOrIndexes(t *testing.T) {
	commit := strings.Repeat("a", 40)
	var requests []string
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, r.URL.Path)
		if !strings.HasPrefix(r.URL.Path, "/api/models/") {
			t.Errorf("unselected body fetched: %s", r.URL.Path)
			http.Error(w, "no", 500)
			return
		}
		fmt.Fprintf(w, `{"sha":%q,"siblings":[{"rfilename":"first.safetensors","lfs":{"sha256":%q,"size":12}},{"rfilename":"second.safetensors","lfs":{"sha256":%q,"size":20}},{"rfilename":"broken.safetensors.index.json","size":20},{"rfilename":"tool.exe","size":100}]}`, commit, strings.Repeat("1", 64), strings.Repeat("2", 64))
	}))
	defer server.Close()
	client := server.Client()
	client.Transport.(*http.Transport).TLSClientConfig.ServerName = "example.com"
	client.Transport.(*http.Transport).DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
	}
	resolver := Resolver{kind: HuggingFace, http: client}
	for _, name := range []string{"first.safetensors", "second.safetensors"} {
		source, problem := Parse("https://huggingface.co/owner/model/blob/main/"+name, "")
		if problem != nil {
			t.Fatal(problem)
		}
		plan, problem := resolver.Resolve(context.Background(), source)
		if problem != nil {
			t.Fatal(problem)
		}
		if len(plan.Files) != 1 || plan.Files[0].Member != name || !plan.Files[0].Carrier {
			t.Fatalf("wrong selection: %+v", plan)
		}
		if plan.Canonical != "hf://owner/model@"+commit+"/"+name {
			t.Fatal(plan.Canonical)
		}
		again, problem := Parse(plan.Canonical, "")
		if problem != nil || again.Member != name || again.Revision != commit {
			t.Fatalf("invalid frozen source: %+v %v", again, problem)
		}
	}
	if len(requests) != 2 {
		t.Fatal(requests)
	}
	source, _ := Parse("hf://owner/model@"+strings.Repeat("b", 40)+"/first.safetensors", "")
	if _, problem := resolver.Resolve(context.Background(), source); problem == nil || problem.Name != "model_source_revision_mismatch" {
		t.Fatalf("accepted wrong revision: %v", problem)
	}
}

func TestHuggingFaceIndexSelectionRetainsOnlyItsShardClosure(t *testing.T) {
	commit := strings.Repeat("a", 40)
	var paths []string
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if strings.HasSuffix(r.URL.Path, "selected.safetensors.index.json") {
			fmt.Fprint(w, `{"weight_map":{"a":"part-1.safetensors","b":"part-2.safetensors"}}`)
			return
		}
		if !strings.HasPrefix(r.URL.Path, "/api/models/") {
			t.Errorf("unexpected body: %s", r.URL.Path)
			http.Error(w, "no", 500)
			return
		}
		fmt.Fprintf(w, `{"sha":%q,"siblings":[{"rfilename":"folder/selected.safetensors.index.json","size":80},{"rfilename":"folder/part-1.safetensors","lfs":{"sha256":%q,"size":12}},{"rfilename":"folder/part-2.safetensors","lfs":{"sha256":%q,"size":20}},{"rfilename":"ignored.safetensors.index.json","size":20},{"rfilename":"ignored.safetensors","size":10}]}`, commit, strings.Repeat("1", 64), strings.Repeat("2", 64))
	}))
	defer server.Close()
	client := server.Client()
	client.Transport.(*http.Transport).TLSClientConfig.ServerName = "example.com"
	client.Transport.(*http.Transport).DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
	}
	resolver := Resolver{kind: HuggingFace, http: client}
	source, problem := Parse("https://huggingface.co/owner/model/blob/main/folder/selected.safetensors.index.json", "")
	if problem != nil {
		t.Fatal(problem)
	}
	plan, problem := resolver.Resolve(context.Background(), source)
	if problem != nil {
		t.Fatal(problem)
	}
	if len(plan.Files) != 3 || len(paths) != 2 {
		t.Fatalf("unexpected closure: %+v %v", plan.Files, paths)
	}
	for _, file := range plan.Files {
		if !strings.HasPrefix(file.Member, "folder/") {
			t.Fatalf("unrelated file: %+v", file)
		}
	}
	if plan.Files[0].Carrier || plan.Files[1].Carrier || !plan.Files[2].Carrier || len(plan.Files[2].Requires) != 2 {
		t.Fatalf("wrong index facts: %+v", plan.Files)
	}
}
