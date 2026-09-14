package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"google.golang.org/grpc/codes"
	"strings"
	"testing"
)

func TestLocalWheelHeaderRefusalTerminatesRequestWithoutRetry(t *testing.T) {
	for _, code := range []codes.Code{codes.InvalidArgument, codes.FailedPrecondition, codes.Unimplemented, codes.PermissionDenied, codes.NotFound, codes.DataLoss} {
		t.Run(code.String(), func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public, uploadEnds: []codes.Code{code}}
			root := t.TempDir()
			connection, _ := startFakePod(t, root, pod)
			revision := stageLocalRevision(t, root)
			o := hostOwner(t, "upload-refused", rentalWiring(connection, private), func(opt *orchestrator.Options) { opt.Packages = localLauncher{revision: revision} })
			requestID := submitPrivateRental(t, o, revision, "upload-refused")
			waitUntil(t, "terminal upload refusal", func() bool {
				row, problem := o.store.RequestRow(requestID)
				fatal(t, problem)
				return row.State == "failed"
			})
			events, problem := o.store.EventsAfter(requestID, 0, 100)
			fatal(t, problem)
			found := false
			for _, event := range events {
				if event.Type == "request.failed" {
					found = true
					if event.Payload["error_type"] != "local_package_upload_refused" || !strings.Contains(event.Payload["error"].(string), "filename refused") {
						t.Fatalf("refusal was hidden: %v", event.Payload)
					}
				}
			}
			if !found {
				t.Fatal("no request.failed event")
			}
			pod.mu.Lock()
			defer pod.mu.Unlock()
			if pod.uploadCalls != 1 || len(pod.uploads) != 0 || len(pod.localPrepares) != 0 || len(pod.offers) != 0 {
				t.Fatalf("refused header retried or reached preparation: calls=%d uploads=%d prepares=%d offers=%d", pod.uploadCalls, len(pod.uploads), len(pod.localPrepares), len(pod.offers))
			}
		})
	}
}

func TestLocalWheelTransportUnavailableRetriesAndReachesServing(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, uploadEnds: []codes.Code{codes.Unavailable}, serve: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	o := hostOwner(t, "upload-retry", rentalWiring(connection, private), func(opt *orchestrator.Options) { opt.Packages = localLauncher{revision: revision} })
	requestID := submitPrivateRental(t, o, revision, "upload-retry")
	waitUntil(t, "serving after transient upload failure", func() bool {
		// The fake pod has no idle heartbeat; simulate its normal reconciliation wake.
		o.c.WakeQueue()
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) != 0
	})
	row, problem := o.store.RequestRow(requestID)
	fatal(t, problem)
	if row.State == "failed" {
		t.Fatalf("transient upload failed request: %+v", row)
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if pod.uploadCalls != len(revision.Files)+1 || len(pod.uploads) != len(revision.Files) {
		t.Fatalf("upload retry calls=%d files=%d", pod.uploadCalls, len(pod.uploads))
	}
}
