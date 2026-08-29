// Package upload owns bounded content-addressed bytes received by the localhost API.
package upload

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
)

const MaxBytes int64 = 64 << 20

type Stored struct {
	ID        string `json:"upload_id"`
	Digest    string `json:"digest"`
	Length    int64  `json:"length"`
	MediaType string `json:"media_type"`
	URL       string `json:"url"`
}

func Put(layout home.Layout, source io.Reader, contentLength int64, mediaType string) (Stored, *exit.Error) {
	if contentLength > MaxBytes {
		return Stored{}, tooLarge(contentLength)
	}
	normalized, _, err := mime.ParseMediaType(strings.TrimSpace(mediaType))
	if err != nil || normalized == "" {
		return Stored{}, exit.Usagef("upload Content-Type %q is not a media type", mediaType)
	}
	temporary, err := os.CreateTemp(layout.Uploads, ".upload-*")
	if err != nil {
		return Stored{}, exit.Internalf("cannot create upload staging: %s", err)
	}
	temporaryPath := temporary.Name()
	keep := false
	defer func() {
		_ = temporary.Close()
		if !keep {
			_ = os.Remove(temporaryPath)
		}
	}()

	hash := sha256.New()
	written, err := io.Copy(io.MultiWriter(temporary, hash), io.LimitReader(source, MaxBytes+1))
	if err != nil {
		return Stored{}, exit.Internalf("cannot receive upload bytes: %s", err)
	}
	if written > MaxBytes {
		return Stored{}, tooLarge(written)
	}
	if err := temporary.Sync(); err != nil {
		return Stored{}, exit.Internalf("cannot sync upload bytes: %s", err)
	}
	if err := temporary.Close(); err != nil {
		return Stored{}, exit.Internalf("cannot close upload staging: %s", err)
	}

	hexDigest := hex.EncodeToString(hash.Sum(nil))
	id := "upl-" + hexDigest
	destination := filepath.Join(layout.Uploads, hexDigest)
	if info, err := os.Stat(destination); err == nil {
		if info.Size() != written {
			return Stored{}, exit.Internalf("upload %s exists with another length", id)
		}
	} else if !os.IsNotExist(err) {
		return Stored{}, exit.Internalf("cannot inspect upload destination: %s", err)
	} else if err := os.Rename(temporaryPath, destination); err != nil {
		return Stored{}, exit.Internalf("cannot publish upload: %s", err)
	} else {
		keep = true
		_ = os.Chmod(destination, 0o600)
	}
	return Stored{
		ID: id, Digest: "sha256:" + hexDigest, Length: written,
		MediaType: normalized, URL: "/v1/uploads/" + id,
	}, nil
}

func Open(layout home.Layout, id string) (*os.File, string, *exit.Error) {
	hexDigest, ok := strings.CutPrefix(id, "upl-")
	if !ok || len(hexDigest) != 64 {
		return nil, "", notFound()
	}
	if _, err := hex.DecodeString(hexDigest); err != nil || strings.ToLower(hexDigest) != hexDigest {
		return nil, "", notFound()
	}
	file, err := os.Open(filepath.Join(layout.Uploads, hexDigest))
	if os.IsNotExist(err) {
		return nil, "", notFound()
	}
	if err != nil {
		return nil, "", exit.Internalf("cannot open upload %s: %s", id, err)
	}
	return file, "sha256:" + hexDigest, nil
}

func tooLarge(length int64) *exit.Error {
	return exit.Named(exit.Structural, "upload_too_large",
		"upload is %d bytes; the localhost boundary admits at most %d", length, MaxBytes)
}

func notFound() *exit.Error {
	return exit.Named(exit.NotFound, "upload_not_found", "no upload by that opaque id")
}
