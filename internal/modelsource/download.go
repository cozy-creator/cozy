package modelsource

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

const maxCarrierHeader = int64(64 << 20)

type StagedFile struct {
	Member  string
	Path    string
	Carrier bool
}

func (r *Resolver) Stage(ctx context.Context, plan Plan, root string, headersOnly bool,
	progress func(string),
) ([]StagedFile, *exit.Error) {
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, exit.Internalf("cannot create model transfer staging: %s", err)
	}
	need := int64(0)
	if headersOnly {
		for _, file := range plan.Files {
			if len(file.Inline) > 0 {
				need += int64(len(file.Inline))
			} else {
				need += min(file.Length, maxCarrierHeader+8)
			}
		}
	} else {
		missing := int64(0)
		for _, file := range plan.Files {
			target := filepath.Join(root, filepath.FromSlash(file.Member))
			if exactFile(target, file) || len(file.Inline) > 0 {
				continue
			}
			landed := int64(0)
			if info, err := os.Stat(target + ".part"); err == nil && info.Mode().IsRegular() && info.Size() < file.Length {
				landed = info.Size()
			}
			missing += file.Length - landed
		}
		if plan.Bytes > (1<<63-1)-missing-(512<<20) {
			return nil, exit.Named(exit.Validation, "model_source_too_large", "model source is too large")
		}
		need = plan.Bytes + missing + 512<<20
	}
	if free, err := availableBytes(root); err == nil && uint64(need) > free {
		return nil, exit.Named(exit.Capacity, "model_transfer.disk_shortfall",
			"model transfer needs %d bytes of staging/store headroom; %d bytes are free", need, free).
			WithRemedy("free local disk space or authorize a rented upload")
	}
	staged := make([]StagedFile, 0, len(plan.Files))
	for _, file := range plan.Files {
		target := filepath.Join(root, filepath.FromSlash(file.Member))
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			return nil, exit.Internalf("cannot create model member directory: %s", err)
		}
		if len(file.Inline) > 0 {
			if problem := writeInline(target, file); problem != nil {
				return nil, problem
			}
		} else if headersOnly {
			if problem := r.stageHeader(ctx, file, target); problem != nil {
				return nil, problem
			}
		} else if problem := r.download(ctx, file, target, progress); problem != nil {
			return nil, problem
		}
		staged = append(staged, StagedFile{Member: file.Member, Path: target, Carrier: file.Carrier})
	}
	return staged, nil
}

func writeInline(target string, file File) *exit.Error {
	if int64(len(file.Inline)) != file.Length || digest(file.Inline) != file.SHA256 {
		return exit.Internalf("inline provider metadata changed after resolution")
	}
	if err := os.WriteFile(target, file.Inline, 0o600); err != nil {
		return exit.Internalf("cannot stage provider index: %s", err)
	}
	return nil
}

func (r *Resolver) stageHeader(ctx context.Context, file File, target string) *exit.Error {
	first, problem := r.rangeBytes(ctx, file, 0, 7)
	if problem != nil {
		return problem
	}
	if len(first) != 8 {
		return exit.Named(exit.Validation, "model_source_header_short", "%s returned a short header prefix", file.Member)
	}
	length := int64(binary.LittleEndian.Uint64(first))
	if length <= 1 || length > maxCarrierHeader || 8+length > file.Length {
		return exit.Named(exit.Validation, "model_source_header_invalid",
			"%s declares invalid header length %d", file.Member, length)
	}
	header, problem := r.rangeBytes(ctx, file, 8, 7+length)
	if problem != nil {
		return problem
	}
	if int64(len(header)) != length {
		return exit.Named(exit.Validation, "model_source_header_short", "%s returned a short header", file.Member)
	}
	temporary := target + ".header-part"
	f, err := os.OpenFile(temporary, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return exit.Internalf("cannot stage model header: %s", err)
	}
	_, writeErr := f.Write(first)
	if writeErr == nil {
		_, writeErr = f.Write(header)
	}
	if writeErr == nil {
		writeErr = f.Truncate(file.Length)
	}
	if writeErr == nil {
		writeErr = f.Sync()
	}
	closeErr := f.Close()
	if writeErr != nil || closeErr != nil {
		_ = os.Remove(temporary)
		return exit.Internalf("cannot finish sparse model header: %v %v", writeErr, closeErr)
	}
	if err := os.Rename(temporary, target); err != nil {
		_ = os.Remove(temporary)
		return exit.Internalf("cannot commit sparse model header: %s", err)
	}
	return nil
}

func (r *Resolver) rangeBytes(ctx context.Context, file File, first, last int64) ([]byte, *exit.Error) {
	headers := make(http.Header)
	headers.Set("Range", fmt.Sprintf("bytes=%d-%d", first, last))
	response, problem := r.request(ctx, http.MethodGet, file.URL, headers)
	if problem != nil {
		return nil, problem
	}
	defer response.Body.Close()
	if !ExactRange(response.StatusCode, response.ContentLength, response.Header.Get("Content-Range"),
		first, last, file.Length) {
		return nil, exit.Named(exit.Validation, "model_source_range_unsupported",
			"provider did not honor the exact bounded header range")
	}
	want := last - first + 1
	body, err := io.ReadAll(io.LimitReader(response.Body, want+1))
	if err != nil || int64(len(body)) != want {
		return nil, exit.Named(exit.Validation, "model_source_range_invalid",
			"provider returned %d bytes for a %d-byte range", len(body), want)
	}
	return body, nil
}

// ExactRange verifies that a provider returned precisely the requested byte
// interval from the already-resolved immutable object.
func ExactRange(status int, contentLength int64, contentRange string, first, last, total int64) bool {
	want := last - first + 1
	return status == http.StatusPartialContent &&
		contentRange == fmt.Sprintf("bytes %d-%d/%d", first, last, total) &&
		(contentLength < 0 || contentLength == want)
}

func (r *Resolver) download(ctx context.Context, file File, target string, progress func(string)) *exit.Error {
	if exactFile(target, file) {
		return nil
	}
	_ = os.Remove(target)
	part, etagPath := target+".part", target+".etag"
	info, err := os.Stat(part)
	offset := int64(0)
	if err == nil && info.Mode().IsRegular() && info.Size() == file.Length && exactFile(part, file) {
		if renameErr := os.Rename(part, target); renameErr == nil {
			_ = os.Remove(etagPath)
			return nil
		}
	} else if err == nil && info.Mode().IsRegular() && info.Size() < file.Length {
		offset = info.Size()
	} else if err == nil {
		_ = os.Remove(part)
		_ = os.Remove(etagPath)
	}
	etag, _ := os.ReadFile(etagPath)
	if offset > 0 && len(etag) == 0 {
		offset = 0
		_ = os.Remove(part)
	}
	headers := make(http.Header)
	if offset > 0 {
		headers.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		headers.Set("If-Range", strings.TrimSpace(string(etag)))
	}
	response, problem := r.request(ctx, http.MethodGet, file.URL, headers)
	if problem != nil {
		return problem
	}
	defer response.Body.Close()
	if offset > 0 && response.StatusCode == http.StatusOK {
		offset = 0
		_ = os.Remove(part)
	} else if offset > 0 && response.StatusCode == http.StatusPartialContent {
		wantPrefix := "bytes " + strconv.FormatInt(offset, 10) + "-"
		contentRange := response.Header.Get("Content-Range")
		_, total, found := strings.Cut(contentRange, "/")
		declared, parseErr := strconv.ParseInt(total, 10, 64)
		if !strings.HasPrefix(contentRange, wantPrefix) || !found || parseErr != nil || declared != file.Length {
			return exit.Named(exit.Validation, "model_source_resume_mismatch",
				"provider resumed %s at the wrong byte", file.Member)
		}
	} else if response.StatusCode != http.StatusOK {
		return exit.Named(exit.Validation, "model_source_download_refused",
			"provider download for %s answered HTTP %d", file.Member, response.StatusCode)
	}
	flags := os.O_CREATE | os.O_WRONLY
	if offset == 0 {
		flags |= os.O_TRUNC
	} else {
		flags |= os.O_APPEND
	}
	f, err := os.OpenFile(part, flags, 0o600)
	if err != nil {
		return exit.Internalf("cannot open model transfer part: %s", err)
	}
	if value := response.Header.Get("ETag"); value != "" && !strings.HasPrefix(value, "W/") {
		_ = os.WriteFile(etagPath, []byte(value), 0o600)
	}
	remaining := file.Length - offset
	n, copyErr := io.Copy(f, io.LimitReader(response.Body, remaining+1))
	syncErr := f.Sync()
	closeErr := f.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil {
		return exit.Unavailablef("model download stopped after %d bytes: %v %v %v", offset+n, copyErr, syncErr, closeErr)
	}
	if n != remaining {
		return exit.Named(exit.Validation, "model_source_length_mismatch",
			"%s delivered %d of %d remaining bytes", file.Member, n, remaining)
	}
	if progress != nil {
		progress(fmt.Sprintf("downloaded %s (%d bytes)", file.Member, file.Length))
	}
	if !exactFile(part, file) {
		_ = os.Remove(part)
		_ = os.Remove(etagPath)
		return exit.Named(exit.Validation, "model_source_digest_mismatch",
			"%s did not match its declared SHA-256/length", file.Member)
	}
	if err := os.Rename(part, target); err != nil {
		return exit.Internalf("cannot commit downloaded model member: %s", err)
	}
	_ = os.Remove(etagPath)
	return nil
}

func exactFile(path string, file File) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() != file.Length {
		return false
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, f); err != nil {
		return false
	}
	return hex.EncodeToString(hash.Sum(nil)) == file.SHA256
}
