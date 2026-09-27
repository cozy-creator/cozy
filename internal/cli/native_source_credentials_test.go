package cli

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/secret"
)

// A gated Civitai or Hugging Face source fetched by a rented ingest answered 401 on the pod
// because upload_* exports received no credential.
func TestProviderUploadsReceiveTheProviderCredential(t *testing.T) {
	r := NewResolver(nil, config.Config{
		HuggingFaceToken: secret.New("hf-token"),
		CivitaiToken:     secret.New("civitai-token"),
	}, nil)
	for operation, want := range map[string]string{
		"download_huggingface": "bearer hf-token",
		"upload_huggingface":   "bearer hf-token",
		"download_civitai":     "bearer civitai-token",
		"upload_civitai":       "bearer civitai-token",
		"convert_cozytensors":  "",
		"source_files":         "",
	} {
		if got := r.NativeSourceCredential(operation); got != want {
			t.Errorf("%s: credential %q, want %q", operation, got, want)
		}
	}
}
