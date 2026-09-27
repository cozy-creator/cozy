package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"slices"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

func (m *machineRuns) publicationAuthorization(ctx context.Context, request, machine string, connection *machineConnection) (string, *exit.Error) {
	names, problem := m.store.RequestPublicationRepositories(request)
	if problem != nil || len(names) == 0 {
		return "", problem
	}
	if machine == "local" || len(connection.certificateDigest) != sha256.Size {
		return "", exit.Named(exit.Structural, "publication.machine_identity_required", "publication authority requires the rented machine's pinned certificate identity")
	}
	raw, problem := m.store.MachinePublicationIntent(request)
	if problem != nil {
		return "", problem
	}
	var intent hub.MachinePublicationGrantIntent
	account := client(m.fleet.atRental(machine))
	if len(raw) == 0 {
		selected, problem := account.Rental(ctx, machine)
		if problem != nil {
			return "", problem
		}
		if selected.ID != machine || selected.WorkerID != connection.claim.WorkerId || selected.WorkerBootID != connection.claim.WorkerBootId {
			return "", exit.New(exit.Conflict, "publication authority rental readback differs from the authenticated machine")
		}
		block, remaining := pem.Decode([]byte(selected.CertPEM))
		if block == nil || block.Type != "CERTIFICATE" || len(bytes.TrimSpace(remaining)) != 0 {
			return "", exit.New(exit.Conflict, "publication authority has no exact machine certificate")
		}
		certificate, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return "", exit.New(exit.Conflict, "publication authority machine certificate is invalid")
		}
		now := time.Now()
		expires := now.Add(7 * 24 * time.Hour)
		if expires.After(certificate.NotAfter) {
			expires = certificate.NotAfter
		}
		intent, problem = hub.PrepareMachinePublicationGrant(selected, names, now, expires)
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
	if err != nil || intent.RentalID != machine || !bytes.Equal(digest[:], connection.certificateDigest) ||
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
