package producttest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
)

func childDigest(letter string) string { return "sha256:" + strings.Repeat(letter, 64) }

func offerChildParent(t *testing.T, store *records.Store, parent records.Request) records.Request {
	t.Helper()
	_, problem := store.Dispatch(records.Attempt{RequestID: parent.ID, InstanceID: "private-worker", SessionID: "private-boot", InvocationDigest: childDigest("1"), InvocationCanonical: []byte(`{}`)})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(parent.ID, 1, "private-boot"))
	current, problem := store.RequestRow(parent.ID)
	fatal(t, problem)
	return *current
}

func assessmentJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func assessmentDocument(t *testing.T, value map[string]any) []byte {
	t.Helper()
	raw, err := fixtureBytes(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func assessmentDigest(raw []byte) string {
	value, _ := canonical.Spell(canonical.Digest(raw))
	return value
}
