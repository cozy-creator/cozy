package machinev1

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"encoding/pem"
	"errors"
	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/workertls"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestReclaimUsesPinnedTLSMachineAuthorityAndRefusesOnlyUnsupportedOperation(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	responseStatus := http.StatusOK
	responseMalformed := false
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/machine/memory/reclaim" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		grant, err := capability.Verify(strings.TrimPrefix(r.Header.Get("Authorization"), "Cozy-Cap "), "owned-machine", []ed25519.PublicKey{public}, time.Now(), "")
		if err != nil || grant.Action != ScopeMachine || grant.Run != "" || len(grant.Outputs) != 0 {
			t.Errorf("maintenance authority was not machine scope: %+v %v", grant, err)
		}
		w.WriteHeader(responseStatus)
		if responseStatus == http.StatusOK {
			if responseMalformed {
				_, _ = w.Write([]byte(`{}`))
				return
			}
			_ = json.NewEncoder(w).Encode(IdleMemoryReclaim{ExecutorsBefore: 2, ExecutorsEnded: 2, RootExportBytesReleased: 1234})
		}
	}))
	defer server.Close()
	pin, err := workertls.ParsePin(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw}))
	if err != nil {
		t.Fatal(err)
	}
	client, err := Dial(strings.TrimPrefix(server.URL, "https://"), pin.TLSConfig(), "owned-machine", Signer{Public: public, Sign: func(raw []byte) []byte { return ed25519.Sign(private, raw) }})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	receipt, err := client.ReclaimIdleMemory(context.Background())
	if err != nil || receipt.ExecutorsEnded != 2 || receipt.RootExportBytesReleased != 1234 {
		t.Fatalf("%+v %v", receipt, err)
	}
	responseStatus = http.StatusNotFound
	if _, err := client.ReclaimIdleMemory(context.Background()); !errors.Is(err, ErrReclaimUnavailable) {
		t.Fatalf("older machine did not refuse only new operation: %v", err)
	}
	// A missing operation does not replace or close the authenticated client.
	responseStatus = http.StatusOK
	if _, err := client.ReclaimIdleMemory(context.Background()); err != nil {
		t.Fatalf("unsupported operation damaged client: %v", err)
	}
	responseStatus = http.StatusConflict
	if _, err := client.ReclaimIdleMemory(context.Background()); !errors.Is(err, ErrReclaimBusy) {
		t.Fatalf("busy refusal lost its operation meaning: %v", err)
	}
	responseStatus = http.StatusOK
	responseMalformed = true
	if _, err := client.ReclaimIdleMemory(context.Background()); err == nil {
		t.Fatal("missing counters were interpreted as successful zero unknowns")
	}
}
