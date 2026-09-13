package hub

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"net/http"
	"slices"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/google/uuid"
)

type PublicationRepository struct {
	Org  string `json:"org"`
	Name string `json:"name"`
}

// MachinePublicationGrantIntent is frozen in the client submission record before
// its first AuthKit call. A retry reuses this exact value, including ID and expiry.
// It contains public certificate/scope metadata, never a device key or token.
type MachinePublicationGrantIntent struct {
	AuthorizationID string                  `json:"authorization_id"`
	RentalID        string                  `json:"rental_id"`
	Repositories    []PublicationRepository `json:"repositories"`
	Permissions     []string                `json:"permissions"`
	ExpiresAtUnix   int64                   `json:"expires_at_unix"`
	CertificateDER  string                  `json:"certificate_der_b64url"`
}

func PrepareMachinePublicationGrant(rental Rental, repositories []string, now, expires time.Time) (MachinePublicationGrantIntent, *exit.Error) {
	if rental.State != RentalReady || rental.ID == "" || rental.WorkerID == "" {
		return MachinePublicationGrantIntent{}, exit.New(exit.Conflict, "publication authority requires the current ready rental")
	}
	block, remaining := pem.Decode([]byte(rental.CertPEM))
	if block == nil || block.Type != "CERTIFICATE" || len(bytes.TrimSpace(remaining)) != 0 {
		return MachinePublicationGrantIntent{}, exit.New(exit.Validation, "rental has no exact pinned certificate")
	}
	certificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil || now.Before(certificate.NotBefore) || !now.Before(certificate.NotAfter) {
		return MachinePublicationGrantIntent{}, exit.New(exit.Validation, "rental certificate is not currently valid")
	}
	expires = expires.UTC().Truncate(time.Second)
	if !expires.After(now.Add(time.Second)) || expires.After(now.Add(7*24*time.Hour)) || expires.After(certificate.NotAfter) {
		return MachinePublicationGrantIntent{}, exit.New(exit.Validation, "publication permission exceeds its certificate or seven-day window")
	}
	names := slices.Clone(repositories)
	slices.Sort(names)
	names = slices.Compact(names)
	if len(names) < 1 || len(names) > 16 {
		return MachinePublicationGrantIntent{}, exit.New(exit.Validation, "publication permission needs 1..16 explicit model repositories")
	}
	wanted := make([]PublicationRepository, 0, len(names))
	for _, name := range names {
		ref, problem := ParseRef(name)
		if problem != nil {
			return MachinePublicationGrantIntent{}, problem
		}
		if ref.Org == "local" {
			return MachinePublicationGrantIntent{}, exit.New(exit.Validation, "publication permission needs public model repository names")
		}
		wanted = append(wanted, PublicationRepository{Org: ref.Org, Name: ref.Name})
	}
	id, err := uuid.NewRandom()
	if err != nil {
		return MachinePublicationGrantIntent{}, exit.Internalf("cannot create publication authorization identity")
	}
	return MachinePublicationGrantIntent{AuthorizationID: id.String(), RentalID: rental.ID, Repositories: wanted,
		Permissions: []string{"assessment", "checkpoint", "release"}, ExpiresAtUnix: expires.Unix(), CertificateDER: base64.RawURLEncoding.EncodeToString(block.Bytes)}, nil
}

// AuthorizeMachinePublication uses the client's ordinary AuthKit credential once.
// Runtime later renews with its own certificate and the returned grant ID; the
// initial short token is deliberately discarded here and never sent to Python.
func (c *Client) AuthorizeMachinePublication(ctx context.Context, intent MachinePublicationGrantIntent) *exit.Error {
	id, err := uuid.Parse(intent.AuthorizationID)
	if err != nil || id == uuid.Nil || id.String() != intent.AuthorizationID || intent.RentalID == "" {
		return exit.New(exit.Validation, "publication authorization intent is malformed")
	}
	ttl := min(int64(900), intent.ExpiresAtUnix-time.Now().Unix()-1)
	if ttl < 1 {
		return exit.Named(exit.Credential, "publication.authority_expired", "publication authorization has expired; start a new explicitly authorized run")
	}
	var out struct {
		Token     string    `json:"token"`
		ExpiresAt time.Time `json:"expires_at"`
	}
	problem := c.do(ctx, call{method: http.MethodPost, path: "/v1/auth/delegated/token", auth: true,
		body: map[string]any{"ttl_seconds": ttl, "delegate_certificate_der_b64url": intent.CertificateDER,
			"requested_grant": map[string]any{"authorization_id": intent.AuthorizationID, "rental_id": intent.RentalID,
				"repositories": intent.Repositories, "permissions": intent.Permissions, "expires_at_unix": intent.ExpiresAtUnix}},
	}, &out)
	if problem != nil {
		return problem
	}
	if out.Token == "" || out.ExpiresAt.IsZero() {
		return exit.Named(exit.Conflict, "publication.authority_response_invalid", "AuthKit did not acknowledge the requested publication authority")
	}
	return nil
}
