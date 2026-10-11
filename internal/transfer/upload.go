// Package transfer writes objects to presigned object-storage URLs: bounded by bytes moving,
// retried at the storage edge, never holding a credential of ours.
package transfer

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// One object write or read at the storage edge is bounded by BYTES MOVING, not by a
// clock — see `mover`. The 30-minute constant that used to live here said in its own
// comment that it was standing in for a byte meter.

// attempts bounds transport retries at the storage edge. A hiccup on a residential
// uplink is not a protocol event: every condition is signed INTO the url, so
// re-sending the same body is the same write, and the no-clobber precondition means
// a partially-received first attempt cannot become half an object. Three tries, then
// the run stops and says so — and a stopped run resumes, because the objects that did
// land are already the hub's.
const attempts = 3

var storageIPv4Client = func() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	transport.DialContext = func(ctx context.Context, _, address string) (net.Conn, error) {
		return dialer.DialContext(ctx, "tcp4", address)
	}
	return &http.Client{Transport: transport}
}()

// send is the one storage-edge write in this binary. It talks to object storage,
// never to the hub: the URL is presigned and carries no credential of ours.
func send(ctx context.Context, method, url string, open func() (io.ReadCloser, error), length int64, headers map[string]string) (int, http.Header, []byte, int64, *exit.Error) {
	var status int
	var header http.Header
	var raw []byte
	var moved int64
	exhausted, err := retryStorage(ctx, func(attempt int) (bool, error) {
		status, header, raw = 0, nil, nil
		body, err := open()
		if err != nil {
			return false, exit.Internalf("the staged object is unreadable: %s", err)
		}
		rctx, cancel := ctx, func() {}
		var counted io.ReadCloser = http.NoBody
		var meter *mover
		if length == 0 {
			body.Close()
		} else {
			// The body is COUNTED and the context lives while the count moves: an upload
			// that is slow is not an upload that has stopped.
			meter = &mover{}
			rctx, cancel = meter.context(ctx)
			counted = meter.readCloser(body)
		}
		defer func() {
			if meter != nil {
				moved += meter.moved.Load()
			}
		}()
		req, err := http.NewRequestWithContext(rctx, method, url, counted)
		if err != nil {
			counted.Close()
			cancel()
			return false, exit.Internalf("the grant's URL is not usable: %s", err)
		}
		req.ContentLength = length
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		client := http.DefaultClient
		if attempt > 0 {
			client = storageIPv4Client
		}
		resp, err := client.Do(req)
		if err != nil {
			client.CloseIdleConnections()
			counted.Close()
			cancel()
			return true, err
		}
		raw, _ = io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		resp.Body.Close()
		counted.Close()
		cancel()
		status, header = resp.StatusCode, resp.Header
		if retryableStorageStatus(resp.StatusCode) {
			// The recorded answer stands if this was the last try: the caller renders
			// the storage layer's own refusal rather than a generic "unreachable".
			return true, fmt.Errorf("object storage answered %d", resp.StatusCode)
		}
		return false, nil
	})
	if err == nil || (exhausted && status != 0) {
		return status, header, raw, moved, nil
	}
	if !exhausted {
		return 0, nil, nil, moved, err.(*exit.Error)
	}
	return 0, nil, nil, moved, exit.Unavailablef("object storage is unreachable after %d attempts: %s", attempts, err).
		WithRemedy("re-run to resume from Tensorhub's durable transfer rows")
}

// UploadPresigned sends one local file to a release-scoped storage grant. Tensorhub
// reads the stored bytes and computes their identity during finalize. The byte
// count is the file length once any PUT attempt read those bytes, capped so
// retries do not count one immutable file twice. A successful PUT whose response
// was lost must not become 0 B when its retry correctly answers that the object
// already exists.
func UploadPresigned(ctx context.Context, subject, path, url string, headers map[string]string) (int64, *exit.Error) {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return 0, exit.Named(exit.Conflict, "upload.local_file_unreadable",
			"%s is no longer a regular file", path)
	}
	length := info.Size()
	status, _, body, moved, problem := send(ctx, http.MethodPut, url, opener(path, 0, length), length, headers)
	if problem != nil {
		return min(moved, length), problem
	}
	moved = min(moved, length) // retries do not make one immutable file larger
	switch status {
	case http.StatusOK, http.StatusCreated, http.StatusNoContent:
		return moved, nil
	case http.StatusPreconditionFailed:
		return moved, nil
	default:
		return moved, storageRefusal(status, subject, body)
	}
}

// storageRefusal renders what the storage layer said. It is not our vocabulary and
// is not translated: BadDigest and SignatureDoesNotMatch mean exactly what they say.
func storageRefusal(status int, subject string, body []byte) *exit.Error {
	said := strings.TrimSpace(string(body))
	if len(said) > 300 {
		said = said[:300] + "…"
	}
	code := exit.Failed
	if status == http.StatusForbidden || status == http.StatusUnauthorized {
		code = exit.Credential
	}
	return exit.Named(code, "storage.refused",
		"object storage answered %d writing %s: %s", status, subject, said).
		WithRemedy("the conditions are signed into the grant; a stripped condition is 403 and a wrong body is 400")
}

// opener hands `send` a FRESH reader per attempt. A retried upload that re-used a
// consumed reader would put an empty body under a signed digest — the exact failure
// retries exist to avoid, dressed up as a corrupt object.
type limitedReadCloser struct {
	io.Reader
	io.Closer
}

func opener(path string, offset, length int64) func() (io.ReadCloser, error) {
	return func() (io.ReadCloser, error) {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		if offset > 0 {
			if _, err := f.Seek(offset, io.SeekStart); err != nil {
				f.Close()
				return nil, err
			}
		}
		return &limitedReadCloser{Reader: io.LimitReader(f, length), Closer: f}, nil
	}
}

func retryableStorageStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooEarly, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}
