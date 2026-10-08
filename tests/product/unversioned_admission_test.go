package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/secret"
)

type describedAdmissionResolver struct {
	api.Resolver
	reads int
}

func (r *describedAdmissionResolver) ResolveRemoteRelease(_, _, _, _ string, _ []orchestrator.ModelRef) (orchestrator.LogicalPackage, *launch.Entrypoint, *exit.Error) {
	r.reads++
	return orchestrator.LogicalPackage{}, nil, exit.Named(exit.Unavailable, "proof.source_lookup", "explicit release lookup observed")
}

func (r *describedAdmissionResolver) ResolveRemoteJob(_, _, _, _ string, _ []orchestrator.ModelRef, _ bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	r.reads++
	return orchestrator.LogicalJob{}, nil, exit.Named(exit.Unavailable, "proof.source_lookup", "explicit release lookup observed")
}

func describedAdmissionInterface(kind, internal string) json.RawMessage {
	return json.RawMessage(`{"format":"cozy.package.interface/1","application":"proof:app","` + kind + `":[{"name":"generate","publishes":false,` + internal + `"request":{"fields":[{"name":"prompt","type":"str","wire":"required"}]},"result":{"fields":[{"name":"clip","type":{"asset":"video"},"asset_bound":{"max_bytes":200000,"media_types":["video/mp4"]}}]}}]}`)
}

// The actual API admits described metadata without selecting a release or reading
// the package catalog. Runtime still receives the bare package/function request.
func TestDescribedPublishedAdmissionKeepsReleaseUnversioned(t *testing.T) {
	for _, kind := range []string{"entrypoints", "jobs"} {
		t.Run(kind, func(t *testing.T) {
			o := hostOwner(t, "unversioned-"+kind)
			resolver := &describedAdmissionResolver{}
			const bearer = "described-admission-fixture"
			handler, problem := api.New(api.Options{Orchestrator: o.c, Cfg: o.cfg,
				Creds: api.Credentials{CLI: secret.New(bearer)}, Addr: "127.0.0.1:11111",
				Web: http.NotFoundHandler(), Packages: resolver, MachineExecutions: publishedRouteObserver{}}).Handler()
			fatal(t, problem)
			path := "/v1/requests"
			if kind == "jobs" {
				path = "/v1/local/jobs"
			}
			send := func(key string, body map[string]any) *httptest.ResponseRecorder {
				t.Helper()
				raw, err := json.Marshal(body)
				must(t, err)
				request := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:11111"+path, bytes.NewReader(raw))
				request.RemoteAddr = "127.0.0.1:12345"
				request.Header.Set("Authorization", "Bearer "+bearer)
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Idempotency-Key", key)
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, request)
				return response
			}
			body := func(raw json.RawMessage) map[string]any {
				return map[string]any{"package": "proof/render", "function": "generate",
					"input": map[string]any{"prompt": "A walking cat"}, "package_interface": raw, "rental": true}
			}
			raw := describedAdmissionInterface(kind, "")
			response := send("bare", body(raw))
			if response.Code != http.StatusAccepted {
				t.Fatalf("bare package refused: %d %s", response.Code, response.Body)
			}
			row, problem := o.store.RequestByIdempotencyKey("bare")
			fatal(t, problem)
			if row == nil || row.Release != "" || row.InstallID != "" || row.LocalInstallationID != "" || row.Outputs != "clip" {
				t.Fatalf("admission pinned code or accepted invented exports: %+v", row)
			}
			export, problem := o.store.OutputExportOf(row.ID)
			fatal(t, problem)
			if export == nil || len(export.Outputs) != 1 || export.Outputs[0].OutputID != "clip" || export.Outputs[0].MediaType != "video/mp4" {
				t.Fatalf("described output metadata was lost: %+v", export)
			}
			if resolver.reads != 0 {
				t.Fatal("bare package performed a controller-side release lookup")
			}
			var scalarDoc map[string]any
			must(t, json.Unmarshal(raw, &scalarDoc))
			scalarDoc[kind].([]any)[0].(map[string]any)["result"] = map[string]any{"fields": []any{}}
			scalarRaw, err := json.Marshal(scalarDoc)
			must(t, err)
			scalar := body(scalarRaw)
			scalar["package"] = "proof/scalar"
			response = send("scalar", scalar)
			if response.Code != http.StatusAccepted {
				t.Fatalf("scalar description refused: %d %s", response.Code, response.Body)
			}
			scalarRow, problem := o.store.RequestByIdempotencyKey("scalar")
			fatal(t, problem)
			scalarExport, problem := o.store.OutputExportOf(scalarRow.ID)
			fatal(t, problem)
			if scalarExport == nil || scalarExport.Directory != o.l.PackageOutputs("proof/scalar") || len(scalarExport.Outputs) != 0 {
				t.Fatalf("scalar description lost destination for later actual outputs: %+v", scalarExport)
			}
			if _, err := os.Stat(scalarExport.Directory); !os.IsNotExist(err) {
				t.Fatalf("scalar admission created an output directory before a Product exists: %v", err)
			}
			// A harmless newer descriptor field does not create an equality gate on replay.
			newer := json.RawMessage(strings.Replace(string(raw), `"application":"proof:app"`, `"application":"proof:app","newer_metadata":true`, 1))
			response = send("bare", body(newer))
			if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"idempotent_replay":true`) {
				t.Fatalf("unversioned replay changed identity: %d %s", response.Code, response.Body)
			}
			for _, refusal := range []struct {
				name, code string
				body       map[string]any
			}{
				{"internal", "callable_internal", body(describedAdmissionInterface(kind, `"internal":true,`))},
				{"unknown", "not_found", body(json.RawMessage(`{"format":"cozy.package.interface/1","application":"proof:app","jobs":[],"entrypoints":[]}`))},
				{"malformed", `"code":"validation"`, body(json.RawMessage(`[]`))},
			} {
				response := send(refusal.name, refusal.body)
				if response.Code < 400 || !strings.Contains(response.Body.String(), refusal.code) {
					t.Fatalf("%s: %d %s", refusal.name, response.Code, response.Body)
				}
				if recorded, problem := o.store.RequestByIdempotencyKey(refusal.name); problem != nil || recorded != nil {
					t.Fatalf("%s refusal entered durable queue: %+v %v", refusal.name, recorded, problem)
				}
			}
			invalid := body(raw)
			invalid["input"] = map[string]any{"prompt": 7}
			response = send("wrong-type", invalid)
			if response.Code < 400 || !strings.Contains(response.Body.String(), "request_payload_invalid") {
				t.Fatalf("payload was not validated: %d %s", response.Code, response.Body)
			}
			if kind == "entrypoints" {
				invented := body(raw)
				invented["outputs"] = []string{"caller-invented-output"}
				response = send("invented-output", invented)
				if response.Code < 400 || !strings.Contains(response.Body.String(), "output_export_set_mismatch") {
					t.Fatalf("caller invented an undeclared export: %d %s", response.Code, response.Body)
				}
			}
			explicit := body(raw)
			explicit["release"] = "1.0.0"
			response = send("explicit", explicit)
			if response.Code < 400 || !strings.Contains(response.Body.String(), "proof.source_lookup") || resolver.reads != 1 {
				t.Fatalf("described metadata bypassed explicit release resolution: %d %s", response.Code, response.Body)
			}
		})
	}
}
