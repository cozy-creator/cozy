package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// retainedFailure is a run's failure before it started, as a refused preparation records it.
func retainedFailure(code, detail string) map[string]any {
	return records.QueuedFailure(exit.Named(exit.Failed, code, "%s", detail))
}

func recordPrivateTransaction(t *testing.T, store *records.Store, label, rentalID string) records.Request {
	t.Helper()
	request := records.Request{
		ID: "req-private-" + label, IdemKey: "idem-private-" + label,
		BodyDigest: "sha256:" + strings.Repeat("a", 64),
		Package:    "local/private-proof", Entrypoint: "prepare",
		Release: "1.0.0", PlanID: "sha256:" + strings.Repeat("b", 64),
		LocalInstallationID: "sha256:" + strings.Repeat("c", 64),
		Payload:             []byte(`{"source_revision":"frozen","seed":17,"quality_bar":0.95}`),
		Worker:              rentalID, Rental: rentalID != "", RentalRequired: rentalID != "",
	}
	_, fresh, problem := store.Submit(request)
	fatal(t, problem)
	if !fresh {
		t.Fatal("private transaction fixture was not newly recorded")
	}
	row, problem := store.RequestByReference(request.ID)
	fatal(t, problem)
	return *row
}

func assertPrivateTransactionIdentity(t *testing.T, store *records.Store, before records.Request, state string) {
	t.Helper()
	after, problem := store.RequestByReference(before.ID)
	fatal(t, problem)
	if after == nil || after.State != state || after.ID != before.ID || after.Number != before.Number ||
		after.IdemKey != before.IdemKey || after.BodyDigest != before.BodyDigest || after.CreatedAt != before.CreatedAt ||
		after.Package != before.Package || after.Entrypoint != before.Entrypoint || after.Release != before.Release ||
		after.PlanID != before.PlanID || after.LocalInstallationID != before.LocalInstallationID ||
		after.InstallationID != before.InstallationID || !bytes.Equal(after.Payload, before.Payload) ||
		after.Worker != before.Worker || after.Ordinal != before.Ordinal {
		t.Fatalf("transaction identity or state changed: before=%+v after=%+v want-state=%s", before, after, state)
	}
}
