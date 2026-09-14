package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

type nestedPayloadResolver struct{ publishedRouteResolver }

func (nestedPayloadResolver) ResolveRemoteJob(pkg, release, function string, models []orchestrator.ModelRef, deferred bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	return orchestrator.LogicalJob{Package: pkg, Release: release, Function: function, DescriptorID: childDigest("9")}, &launch.Entrypoint{Name: function, Kind: "job", Request: launch.Struct{Fields: []launch.Field{{Name: "shots", Type: json.RawMessage(`{"list":{"fields":[{"name":"prompt","type":"str"},{"name":"seed","type":"int"},{"name":"duration_s","type":"int"}]}}`)}}}, Result: launch.Struct{Fields: []launch.Field{}}}, nil
}

func TestJobNestedPayloadIsCanonicalBeforeIdentity(t *testing.T) {
	o := hostOwner(t, "nested-job-payload", func(options *orchestrator.Options) { options.Cfg.RentalsMaxHourlySpendUSDMicros = 1_000_000 })
	o.cfg.RentalsMaxHourlySpendUSDMicros = 1_000_000
	fatal(t, o.store.RecordRental(records.Rental{ID: "nested-rental", MachineName: "nested-host", SKU: "cpu", AcceleratorModel: "CPU", State: "ready", Hub: "http://127.0.0.1:1", Address: "127.0.0.1:1", AcceleratorCount: 1, HourlyRateUSDMicros: 1}))
	const bearer = "nested-payload-test"
	handler, problem := api.New(api.Options{Orchestrator: o.c, Cfg: o.cfg, Creds: api.Credentials{CLI: secret.New(bearer)}, Addr: "127.0.0.1:11111", Web: http.NotFoundHandler(), Packages: nestedPayloadResolver{}, MachineExecutions: publishedRouteObserver{}}).Handler()
	fatal(t, problem)
	var id, digest string
	for _, payload := range []string{`{"shots":[{"prompt":"rover","seed":1234,"duration_s":15}]}`, `{ "shots": [ {"duration_s":15,"seed":1234,"prompt":"rover"} ] }`} {
		body := []byte(`{"package":"alice/ops","release":"1.0.0","function":"main","input":` + payload + `,"rental":true,"requested_rental":"nested-rental"}`)
		request := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:11111/v1/local/jobs", bytes.NewReader(body))
		request.RemoteAddr = "127.0.0.1:12345"
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Authorization", "Bearer "+bearer)
		request.Header.Set("Idempotency-Key", "nested-shots")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusAccepted && response.Code != http.StatusOK {
			t.Fatalf("nested payload %d %s", response.Code, response.Body.String())
		}
		row, problem := o.store.RequestByIdempotencyKey("nested-shots")
		fatal(t, problem)
		if string(row.Payload) != `{"shots":[{"duration_s":15,"prompt":"rover","seed":1234}]}` {
			t.Fatalf("noncanonical nested payload retained: %s", row.Payload)
		}
		if id != "" && (id != row.ID || digest != row.BodyDigest) {
			t.Fatal("nested object order changed request identity")
		}
		id, digest = row.ID, row.BodyDigest
	}
}
