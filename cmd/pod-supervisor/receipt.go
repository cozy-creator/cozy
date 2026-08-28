package main

// THE READINESS RECEIPT, AND THE ONE-SHOT KEY THAT SIGNS IT.
//
// The adapter drops an OPAQUE payload on the filesystem; this process authenticates it
// under the attempt HMAC key and writes the envelope the media plane serves. This file
// does not read, parse, or believe the payload — it MACs bytes.
//
// THE KEY IS WIPED HERE (cl-036). Once the envelope is sealed the key has no further
// purpose, and this process is now also the one parsing attacker-influenced request bytes,
// so "no further purpose" is made a fact rather than a habit: `attemptKey` signs exactly
// once, zeroes its own bytes in the same call, and answers a second caller with an error.
// Its environment slot is already gone — config.go unsets it the instant it decodes it —
// so after `sealAndWipe` returns there is no copy of the key anywhere in the pod.

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/mediawire"
)

var receiptDomain = []byte("cozy.pod-readiness/1\x00")

// attemptKey is the readiness HMAC key and it is ONE-SHOT. The zero value is unusable and
// so is a used one; there is no accessor for the bytes.
type attemptKey struct{ raw []byte }

func newAttemptKey(raw []byte) *attemptKey { return &attemptKey{raw: raw} }

// sealAndWipe MACs one payload under the attempt key and destroys the key in the same
// call. A second call is an error, not a second signature: after the envelope is published
// nothing in this pod may sign anything again, and the merge made that worth enforcing
// rather than documenting.
func (k *attemptKey) sealAndWipe(payload []byte) (string, error) {
	if k == nil || k.raw == nil {
		return "", errors.New("the readiness key was wiped when the envelope was published " +
			"and this process signs nothing after that")
	}
	mac := hmac.New(sha256.New, k.raw)
	_, _ = mac.Write(receiptDomain)
	_, _ = mac.Write(payload)
	sum := hex.EncodeToString(mac.Sum(nil))
	for i := range k.raw {
		k.raw[i] = 0
	}
	k.raw = nil
	return sum, nil
}

// publishReceipt waits for the adapter's payload, seals it, and publishes the envelope by
// atomic rename. It gives up the moment any leg ends: a pod whose adapter or whose media
// plane has stopped will never become ready, and waiting out the frozen deadline would
// only delay the same answer.
func publishReceipt(ctx context.Context, key *attemptKey, legs []*leg) error {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		payload, found, err := readReceiptPayload()
		if err != nil {
			return err
		}
		if found {
			sum, err := key.sealAndWipe(payload)
			if err != nil {
				return err
			}
			envelope, err := json.Marshal(struct {
				Payload    []byte `json:"payload"`
				HMACSHA256 string `json:"hmac_sha256"`
			}{Payload: payload, HMACSHA256: sum})
			if err != nil {
				return fmt.Errorf("encode readiness envelope: %w", err)
			}
			if len(envelope) > mediawire.MaxReceiptBytes {
				return fmt.Errorf("readiness envelope exceeds the media plane's published ceiling")
			}
			return atomicWrite(receiptEnvelopePath, envelope, 0o400)
		}
		for _, l := range legs {
			if l.exited() {
				return legExit(l)
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func readReceiptPayload() ([]byte, bool, error) {
	return readStableRegularFile(receiptPayloadPath, mediawire.MaxReceiptBytes)
}

func readStableRegularFile(path string, maxBytes int64) ([]byte, bool, error) {
	pathInfo, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("inspect adapter readiness payload path: %w", err)
	}
	if !pathInfo.Mode().IsRegular() {
		return nil, false, fmt.Errorf("adapter readiness payload path is not one regular file")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, false, fmt.Errorf("open adapter readiness payload: %w", err)
	}
	defer file.Close()
	openInfo, err := file.Stat()
	if err != nil {
		return nil, false, fmt.Errorf("inspect opened adapter readiness payload: %w", err)
	}
	if !openInfo.Mode().IsRegular() || !os.SameFile(pathInfo, openInfo) {
		return nil, false, fmt.Errorf("adapter readiness payload path changed while it was opened")
	}
	payload, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, false, fmt.Errorf("read adapter readiness payload: %w", err)
	}
	if len(payload) == 0 || int64(len(payload)) > maxBytes {
		return nil, false, fmt.Errorf("adapter readiness payload is empty or exceeds its envelope bound")
	}
	return payload, true, nil
}
