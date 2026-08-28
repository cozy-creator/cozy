package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
)

// fetchExact is THE ONLY OUTBOUND NETWORK CALL IN THIS BINARY, and the fence keeps it
// that way. A pod supervisor that can be talked into fetching an arbitrary URL — or into
// attaching a credential to one — is an exfiltration channel with an extra step, so this
// program egresses exactly twice per boot: the provision spec and the provision bundle,
// each an EXACT GRANT.
//
// Exact means every degree of freedom is closed before a byte is trusted. The URL is
// credential-free HTTPS with no fragment and no userinfo (parseGrant). No proxy, no
// redirect, no content coding: identity bytes or nothing. The body is read under the
// declared length and must equal it, and its sha256 must equal the declared digest before
// the staged file is renamed into place read-only. No Authorization header is ever set —
// there is no credential here to set one with.

func fetchExact(ctx context.Context, grant artifactGrant, target string) error {
	transport := &http.Transport{
		Proxy:              nil,
		DisableCompression: true,
		TLSClientConfig:    &tls.Config{MinVersion: tls.VersionTLS12},
	}
	client := &http.Client{
		Transport: transport,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return errors.New("redirect refused")
		},
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, grant.url.String(), nil)
	if err != nil {
		return fmt.Errorf("construct exact %s request", grant.kind)
	}
	request.Header.Set("Accept-Encoding", "identity")
	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("fetch exact %s bytes: %w", grant.kind, redactNetworkError(err))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("fetch exact %s bytes: HTTP %s", grant.kind, response.Status)
	}
	if encoding := response.Header.Get("Content-Encoding"); encoding != "" && encoding != "identity" {
		return fmt.Errorf("fetch exact %s bytes: content encoding is not identity", grant.kind)
	}
	if response.ContentLength >= 0 && response.ContentLength != grant.length {
		return fmt.Errorf("fetch exact %s bytes: content length mismatch", grant.kind)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return fmt.Errorf("prepare exact %s target: %w", grant.kind, err)
	}
	temp, err := os.CreateTemp(filepath.Dir(target), ".artifact-*")
	if err != nil {
		return fmt.Errorf("stage exact %s bytes: %w", grant.kind, err)
	}
	tempName := temp.Name()
	defer os.Remove(tempName)
	hash := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(temp, hash), io.LimitReader(response.Body, grant.length+1))
	if copyErr != nil {
		temp.Close()
		return fmt.Errorf("read exact %s bytes: %w", grant.kind, copyErr)
	}
	if written != grant.length {
		temp.Close()
		return fmt.Errorf("fetch exact %s bytes: body length mismatch", grant.kind)
	}
	if !bytes.Equal(hash.Sum(nil), grant.want[:]) {
		temp.Close()
		return fmt.Errorf("fetch exact %s bytes: sha256 mismatch", grant.kind)
	}
	if err := temp.Chmod(0o400); err != nil {
		temp.Close()
		return fmt.Errorf("protect exact %s bytes: %w", grant.kind, err)
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return fmt.Errorf("flush exact %s bytes: %w", grant.kind, err)
	}
	if err := temp.Close(); err != nil {
		return fmt.Errorf("close exact %s bytes: %w", grant.kind, err)
	}
	if err := os.Rename(tempName, target); err != nil {
		return fmt.Errorf("publish exact %s bytes: %w", grant.kind, err)
	}
	return syncDirectory(filepath.Dir(target))
}

func redactNetworkError(err error) error {
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		return errors.New(urlErr.Err.Error())
	}
	return errors.New("network request failed")
}
