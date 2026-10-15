package producttest

import (
	"net/http"
	"testing"
)

func TestRunMediaRequiresLocalCredentialBeforePathRead(t *testing.T) {
	root := t.TempDir()
	calcifer := startCalciferProcess(t, root)
	response := calcifer.call(t, http.MethodPost, "/v1/requests", map[string]any{
		"package": "local/proof", "function": "score", "input": map[string]any{},
		"local_assets": []map[string]any{{"field_path": "image", "local_path": "/not-a-granted-file", "digest": childDigest("8"), "length": 1}},
	}, "Authorization", "")
	if response.Status != http.StatusUnauthorized {
		t.Fatalf("untrusted media reached file resolution: %s", response.brief())
	}
	if requests := listInvocations(t, root); len(requests) != 0 {
		t.Fatalf("unauthorized media created requests: %+v", requests)
	}
}
