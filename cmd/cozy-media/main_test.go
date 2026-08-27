package main

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

type zeroReader int64

func (r *zeroReader) Read(p []byte) (int, error) {
	remaining := int64(*r)
	if remaining == 0 {
		return 0, io.EOF
	}
	n := len(p)
	if int64(n) > remaining {
		n = int(remaining)
	}
	clear(p[:n])
	*r -= zeroReader(n)
	return n, nil
}

type discardResponse struct {
	*httptest.ResponseRecorder
	bytes int64
}

func (w *discardResponse) Write(p []byte) (int, error) {
	w.bytes += int64(len(p))
	return len(p), nil
}

func TestPlanUploadAcceptsOnlyTensorhubsExactCanonicalBytes(t *testing.T) {
	base := t.TempDir()
	plans := filepath.Join(base, "plans")
	if err := os.MkdirAll(plans, 0o755); err != nil {
		t.Fatal(err)
	}
	token := secret.New("test-token")
	tokens := filepath.Join(base, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(token)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &server{opt: options{root: filepath.Join(base, "media"), plans: plans,
		tokens: tokens, quota: 16 << 20, maxBody: 16 << 20}}
	if err := s.reloadTokens(); err != nil {
		t.Fatal(err)
	}
	plan := []byte(`{"bindings":[],"descriptor":{"digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","length":1},"entrypoint":"generate","format":"cozy.endpoint.EntrypointBindingPlan/1"}`)
	id, err := canonical.Spell(canonical.Digest(plan))
	if err != nil {
		t.Fatal(err)
	}
	put := func(raw []byte, claimed string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPut, "/v1/plans/"+strings.TrimPrefix(claimed, "sha256:"), bytes.NewReader(raw))
		req.Header.Set("Authorization", "Bearer test-token")
		out := httptest.NewRecorder()
		s.routes().ServeHTTP(out, req)
		return out
	}
	if got := put(plan, id); got.Code != http.StatusCreated {
		t.Fatalf("exact Tensorhub plan status=%d body=%s", got.Code, got.Body.String())
	}
	stored, err := os.ReadFile(filepath.Join(plans, strings.TrimPrefix(id, "sha256:")+".json"))
	if err != nil || !bytes.Equal(stored, plan) {
		t.Fatalf("stored plan = %q, %v", stored, err)
	}
	legacy := []byte(`{"entrypoint_binding_plan_id":"` + id + `","format":"cozy.local.EntrypointBindingRecord/2"}`)
	legacyID, _ := canonical.Spell(canonical.Digest(legacy))
	if got := put(legacy, legacyID); got.Code != http.StatusBadRequest ||
		!strings.Contains(got.Body.String(), "binding_plan_invalid") {
		t.Fatalf("legacy local record status=%d body=%s", got.Code, got.Body.String())
	}
	oversized := httptest.NewRequest(http.MethodPut,
		"/v1/plans/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		strings.NewReader(""))
	oversized.ContentLength = canonical.DocMax + 1
	oversized.Header.Set("Authorization", "Bearer test-token")
	refusal := httptest.NewRecorder()
	s.routes().ServeHTTP(refusal, oversized)
	if refusal.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized plan status=%d body=%s", refusal.Code, refusal.Body.String())
	}
}

func TestBootstrapReceiptServesExactBytesOnlyAfterPublication(t *testing.T) {
	root := t.TempDir()
	receipt := filepath.Join(root, "bootstrap.json")
	s := &server{opt: options{bootstrapReceipt: receipt}}
	call := func() *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/v1/bootstrap/receipt", nil)
		out := httptest.NewRecorder()
		s.routes().ServeHTTP(out, req)
		return out
	}
	if got := call(); got.Code != http.StatusTooEarly {
		t.Fatalf("pending receipt status = %d, want 425", got.Code)
	}
	want := []byte(`{"kind":"cozy.pod_bootstrap_receipt/1","attempt_id":"ra-1"}`)
	if err := os.WriteFile(receipt, want, 0o600); err != nil {
		t.Fatal(err)
	}
	got := call()
	if got.Code != http.StatusOK || !bytes.Equal(got.Body.Bytes(), want) {
		t.Fatalf("receipt = status %d body %q, want 200 %q", got.Code, got.Body.Bytes(), want)
	}
}

func TestDropAttemptRemovesOnlyOwnedMedia(t *testing.T) {
	root := t.TempDir()
	token := secret.New("test-token")
	tokens := filepath.Join(root, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(token)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for path, data := range map[string]string{
		"inputs/a111-payload": "payload", "inputs/a111-input-0": "asset",
		"inputs/a222-payload": "other", "outputs/a111/image": "result",
		"outputs/a222/image": "other-result", ".reservations/a111": "64",
	} {
		full := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s := &server{opt: options{root: root, tokens: tokens}}
	request := httptest.NewRequest(http.MethodDelete, "/v1/attempts/a111", nil)
	request.Header.Set("Authorization", "Bearer test-token")
	response := httptest.NewRecorder()
	s.routes().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("DELETE status = %d, body %s", response.Code, response.Body.String())
	}
	for _, path := range []string{"inputs/a111-payload", "inputs/a111-input-0", "outputs/a111", ".reservations/a111"} {
		if _, err := os.Stat(filepath.Join(root, path)); !os.IsNotExist(err) {
			t.Fatalf("owned media %s remains after DELETE: %v", path, err)
		}
	}
	for _, path := range []string{"inputs/a222-payload", "outputs/a222/image"} {
		if _, err := os.Stat(filepath.Join(root, path)); err != nil {
			t.Fatalf("unrelated media %s was touched: %v", path, err)
		}
	}
}

func TestWorkerWriteAndHTTPUploadShareReservedQuota(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "media")
	for _, dir := range []string{filepath.Join(root, "inputs"), filepath.Join(root, "outputs")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	token := secret.New("test-token")
	tokens := filepath.Join(base, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(token)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &server{opt: options{root: root, tokens: tokens, quota: 10, maxBody: 10}}
	if err := s.reloadTokens(); err != nil {
		t.Fatal(err)
	}
	authorized := func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer test-token")
	}

	reserve := httptest.NewRequest(http.MethodPost, "/v1/outputs/a111?max_bytes=6", nil)
	authorized(reserve)
	reserved := httptest.NewRecorder()
	s.routes().ServeHTTP(reserved, reserve)
	if reserved.Code != http.StatusCreated {
		t.Fatalf("reserve status = %d, body %s", reserved.Code, reserved.Body.String())
	}

	start := make(chan struct{})
	workerDone := make(chan error, 1)
	uploadCode := make(chan int, 1)
	go func() {
		<-start
		workerDone <- os.WriteFile(filepath.Join(root, "outputs", "a111", "image"),
			[]byte("123456"), 0o644)
	}()
	go func() {
		<-start
		req := httptest.NewRequest(http.MethodPut, "/v1/inputs/blob", bytes.NewReader([]byte("12345")))
		authorized(req)
		response := httptest.NewRecorder()
		s.routes().ServeHTTP(response, req)
		uploadCode <- response.Code
	}()
	close(start)
	if err := <-workerDone; err != nil {
		t.Fatal(err)
	}
	code := <-uploadCode
	if code != http.StatusRequestEntityTooLarge && code != http.StatusInsufficientStorage {
		t.Fatalf("HTTP upload racing the worker = %d, want quota refusal", code)
	}
	if got := s.used(); got != 6 {
		t.Fatalf("accounted quota after direct worker write = %d, want exact 6 B reservation", got)
	}
	if _, err := os.Stat(filepath.Join(root, "inputs", "blob")); !os.IsNotExist(err) {
		t.Fatalf("refused HTTP bytes landed despite the output reservation: %v", err)
	}
}

func TestConcurrentUploadsShareOneQuotaDecision(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "media")
	if err := os.MkdirAll(filepath.Join(root, "inputs"), 0o755); err != nil {
		t.Fatal(err)
	}
	token := secret.New("test-token")
	tokens := filepath.Join(base, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(token)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &server{opt: options{root: root, tokens: tokens, quota: 10, maxBody: 10}}
	if err := s.reloadTokens(); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	codes := make(chan int, 2)
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			req := httptest.NewRequest(http.MethodPut, "/v1/inputs/blob", bytes.NewReader([]byte("123456")))
			req.Header.Set("Authorization", "Bearer test-token")
			response := httptest.NewRecorder()
			s.routes().ServeHTTP(response, req)
			codes <- response.Code
		}()
	}
	close(start)
	wg.Wait()
	close(codes)
	created, refused := 0, 0
	for code := range codes {
		if code == http.StatusCreated {
			created++
		} else if code == http.StatusRequestEntityTooLarge || code == http.StatusInsufficientStorage {
			refused++
		}
	}
	if created != 1 || refused != 1 || s.used() > s.opt.quota {
		t.Fatalf("concurrent uploads: created=%d refused=%d used=%d quota=%d",
			created, refused, s.used(), s.opt.quota)
	}
	if _, err := os.Stat(filepath.Join(root, "inputs", ".blob.staging")); !os.IsNotExist(err) {
		t.Fatalf("shared staging name remains: %v", err)
	}
}

func TestLargeInputUploadIsStreamingBounded(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "media")
	if err := os.MkdirAll(filepath.Join(root, "inputs"), 0o755); err != nil {
		t.Fatal(err)
	}
	tokens := filepath.Join(base, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(secret.New("test-token"))+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &server{opt: options{root: root, tokens: tokens, quota: 128 << 20, maxBody: 128 << 20}}
	if err := s.reloadTokens(); err != nil {
		t.Fatal(err)
	}
	const size = int64(64 << 20)
	reader := zeroReader(size)
	request := httptest.NewRequest(http.MethodPut, "/v1/inputs/large", &reader)
	request.ContentLength = size
	request.Header.Set("Authorization", "Bearer test-token")
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	response := httptest.NewRecorder()
	s.routes().ServeHTTP(response, request)
	runtime.ReadMemStats(&after)
	if response.Code != http.StatusCreated {
		t.Fatalf("large upload = %d %s", response.Code, response.Body.String())
	}
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 8<<20 {
		t.Fatalf("64 MiB streaming upload allocated %d B", allocated)
	}
	if info, err := os.Stat(filepath.Join(root, "inputs", "large")); err != nil || info.Size() != size {
		t.Fatalf("large stored input = %#v %v", info, err)
	}
}

func TestLargeOutputDownloadIsStreamingBounded(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "media")
	outputDir := filepath.Join(root, "outputs", "attempt")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatal(err)
	}
	const size = int64(64 << 20)
	output := filepath.Join(outputDir, "video")
	file, err := os.Create(output)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(size); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	tokens := filepath.Join(base, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(secret.New("test-token"))+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &server{opt: options{root: root, tokens: tokens, quota: 128 << 20, maxBody: 128 << 20}}
	if err := s.reloadTokens(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "/v1/outputs/attempt/video", nil)
	request.Header.Set("Authorization", "Bearer test-token")
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	response := &discardResponse{ResponseRecorder: httptest.NewRecorder()}
	s.routes().ServeHTTP(response, request)
	runtime.ReadMemStats(&after)
	if response.Code != http.StatusOK || response.bytes != size ||
		response.Header().Get("Content-Length") != "67108864" {
		t.Fatalf("large output = status %d bytes %d headers %v",
			response.Code, response.bytes, response.Header())
	}
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 8<<20 {
		t.Fatalf("64 MiB streaming download allocated %d B", allocated)
	}
}
