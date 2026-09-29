package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"slices"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// publicationAuthorization binds a run's consented repositories to the machine it runs on:
// a rental or this computer's machine, each by its pinned Host leaf and its hub identity.
func (m *machineRuns) publicationAuthorization(ctx context.Context, request, machine string, connection *machineConnection) (string, *exit.Error) {
	names, problem := m.store.RequestPublicationRepositories(request)
	if problem != nil || len(names) == 0 {
		return "", problem
	}
	machineID := connection.HubID()
	if (!connection.Owned() && machineID == "") || len(connection.CertificateDER) == 0 || len(connection.CertificateDigest) != sha256.Size {
		return "", exit.Named(exit.Structural, "publication.machine_identity_required", "publication authority requires the machine's pinned certificate identity")
	}
	account := connection.Account()
	if account == nil {
		return "", exit.Named(exit.Credential, "publication.account_unavailable", "publication authority requires a Tensorhub login for machine %s", machine)
	}
	raw, problem := m.store.MachinePublicationIntent(request)
	if problem != nil {
		return "", problem
	}
	var intent hub.MachinePublicationGrantIntent
	if len(raw) == 0 {
		if !connection.Owned() {
			// A rental grants only while the Hub says it is this exact ready worker.
			selected, problem := account.Rental(ctx, machineID)
			if problem != nil {
				return "", problem
			}
			if selected.ID != machineID || selected.State != hub.RentalReady ||
				selected.WorkerID != connection.Claim.WorkerId || selected.WorkerBootID != connection.Claim.WorkerBootId {
				return "", exit.New(exit.Conflict, "publication authority rental readback differs from the authenticated machine")
			}
		}
		leaf, err := x509.ParseCertificate(connection.CertificateDER)
		if err != nil {
			return "", exit.New(exit.Conflict, "publication authority machine certificate is invalid")
		}
		now := time.Now()
		expires := now.Add(7 * 24 * time.Hour)
		if leaf.NotAfter.Before(expires) {
			expires = leaf.NotAfter
		}
		intent, problem = hub.PrepareMachinePublicationGrant(machine, machineID, connection.CertificateDER, names, now, expires)
		if problem != nil {
			return "", problem
		}
		raw, _ = json.Marshal(intent)
	} else if json.Unmarshal(raw, &intent) != nil {
		return "", exit.Internalf("recorded publication authorization is unreadable")
	}
	certificate, err := base64.RawURLEncoding.DecodeString(intent.CertificateDER)
	digest := sha256.Sum256(certificate)
	consented := make([]string, 0, len(intent.Repositories))
	for _, repository := range intent.Repositories {
		consented = append(consented, repository.Org+"/"+repository.Name)
	}
	if err != nil || intent.Machine != machine || intent.MachineID != machineID || !bytes.Equal(digest[:], connection.CertificateDigest) ||
		!slices.Equal(consented, names) || !slices.Equal(intent.Permissions, []string{"assessment", "checkpoint", "release"}) {
		return "", exit.New(exit.Conflict, "publication authorization differs from recorded consent or the pinned machine certificate")
	}
	if problem := m.store.RecordMachinePublicationIntent(request, raw); problem != nil {
		return "", problem
	}
	if problem := account.AuthorizeMachinePublication(ctx, intent); problem != nil {
		return "", problem
	}
	return intent.AuthorizationID, nil
}
