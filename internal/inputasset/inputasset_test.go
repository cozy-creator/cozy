package inputasset

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestOpaqueBinaryKeepsOctetStreamAcrossProbeFingerprintAndBind(t *testing.T) {
	data := bytes.Repeat([]byte{0, 255, 127, 128}, 131072/4)
	path := filepath.Join(t.TempDir(), "opaque.bin")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	length, media, problem := Probe(path)
	if problem != nil || length != int64(len(data)) || media != "application/octet-stream" {
		t.Fatalf("probe: %d %q %v", length, media, problem)
	}
	facts, problem := Fingerprint(path, int64(len(data)))
	if problem != nil {
		t.Fatal(problem)
	}
	digest := sha256.Sum256(data)
	if facts.Digest != "sha256:"+hex.EncodeToString(digest[:]) || facts.MediaType != media {
		t.Fatalf("byte identity/type changed: %+v", facts)
	}
	bound, problem := Bind(records.AssetBinding{FieldPath: "latent", LocalPath: path, Digest: facts.Digest, Length: facts.Length, ModTime: facts.ModTime}, int64(len(data)))
	if problem != nil || bound.MediaType != media || bound.Digest != facts.Digest {
		t.Fatalf("binding: %+v %v", bound, problem)
	}
	if problem := Verify(bound, int64(len(data))); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := Fingerprint(path, int64(len(data)-1)); problem == nil || problem.ErrName() != "input_over_cap" {
		t.Fatalf("octet stream bypassed byte bound: %v", problem)
	}
}

func TestKnownImageContentIsNotRelabeledByItsBinaryExtension(t *testing.T) {
	var data bytes.Buffer
	if err := png.Encode(&data, image.NewRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "actual-image.bin")
	if err := os.WriteFile(path, data.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	_, media, problem := Probe(path)
	if problem != nil || media != "image/png" {
		t.Fatalf("known bytes became generic: %q %v", media, problem)
	}
	facts, problem := Fingerprint(path, int64(data.Len()))
	if problem != nil || facts.MediaType != media {
		t.Fatalf("fingerprint changed known type: %+v %v", facts, problem)
	}
}
