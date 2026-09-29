package hub

import (
	"context"
	"crypto/x509"
	"encoding/base64"
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
// its first Hub authorization call. A retry reuses this exact value, including ID and expiry.
// It contains public certificate/scope metadata, never a device key or token. MachineID is a
// rental id (pr-…) or an owned machine's id (om-…); the certificate is its Host leaf.
type MachinePublicationGrantIntent struct {
	AuthorizationID string `json:"authorization_id"`
	// Machine is this host's name for the machine (a rental id or "local"); MachineID is the
	// hub's identity for it (the rental id, or an owned machine's om-… id).
	Machine        string                  `json:"machine"`
	MachineID      string                  `json:"machine_id"`
	Repositories   []PublicationRepository `json:"repositories"`
	Permissions    []string                `json:"permissions"`
	ExpiresAtUnix  int64                   `json:"expires_at_unix"`
	CertificateDER string                  `json:"certificate_der_b64url"`
}

// PrepareMachinePublicationGrant binds explicit repositories to one machine's exact Host
// leaf for at most seven days and never past the leaf's own validity.
func PrepareMachinePublicationGrant(machine, machineID string, leaf []byte, repositories []string, now, expires time.Time) (MachinePublicationGrantIntent, *exit.Error) {
	if machine == "" {
		return MachinePublicationGrantIntent{}, exit.New(exit.Conflict, "publication authority requires one machine identity")
	}
	certificate, err := x509.ParseCertificate(leaf)
	if err != nil || now.Before(certificate.NotBefore) || !now.Before(certificate.NotAfter) {
		return MachinePublicationGrantIntent{}, exit.New(exit.Validation, "machine certificate is not currently valid")
	}
	expires = expires.UTC().Truncate(time.Second)
	if !expires.After(now.Add(time.Second)) || expires.After(now.Add(7*24*time.Hour)) || expires.After(certificate.NotAfter) {
		return MachinePublicationGrantIntent{}, exit.New(exit.Validation, "publication permission exceeds its certificate or seven-day window")
	}
	names, problem := NormalizePublicationRepositories(repositories)
	if problem != nil {
		return MachinePublicationGrantIntent{}, problem
	}
	wanted := make([]PublicationRepository, 0, len(names))
	for _, name := range names {
		ref, _ := ParseRef(name)
		wanted = append(wanted, PublicationRepository{Org: ref.Org, Name: ref.Name})
	}
	id, err := uuid.NewRandom()
	if err != nil {
		return MachinePublicationGrantIntent{}, exit.Internalf("cannot create publication authorization identity")
	}
	return MachinePublicationGrantIntent{AuthorizationID: id.String(), Machine: machine, MachineID: machineID, Repositories: wanted,
		Permissions: []string{"assessment", "checkpoint", "release"}, ExpiresAtUnix: expires.Unix(), CertificateDER: base64.RawURLEncoding.EncodeToString(leaf)}, nil
}

// NormalizePublicationRepositories validates explicit consent before selecting or
// buying a machine. The order and duplicates do not change request identity.
func NormalizePublicationRepositories(repositories []string) ([]string, *exit.Error) {
	names := slices.Clone(repositories)
	slices.Sort(names)
	names = slices.Compact(names)
	if len(names) < 1 || len(names) > 16 {
		return nil, exit.New(exit.Validation, "publication permission needs 1..16 explicit model repositories")
	}
	for _, name := range names {
		ref, problem := ParseRef(name)
		if problem != nil {
			return nil, problem
		}
		if ref.Org == "local" {
			return nil, exit.New(exit.Validation, "publication permission needs public model repository names")
		}
	}
	return names, nil
}

// AuthorizeMachinePublication uses the client's ordinary AuthKit credential once.
// Runtime later renews with its independent worker capability and the grant ID; the
// initial short token is deliberately discarded here and never sent to Python.
func (c *Client) AuthorizeMachinePublication(ctx context.Context, intent MachinePublicationGrantIntent) *exit.Error {
	id, err := uuid.Parse(intent.AuthorizationID)
	if err != nil || id == uuid.Nil || id.String() != intent.AuthorizationID || intent.Machine == "" {
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
	problem := c.do(ctx, call{method: http.MethodPost, path: "/v1/machine-authorizations", auth: true,
		body: map[string]any{"ttl_seconds": ttl, "delegate_certificate_der_b64url": intent.CertificateDER,
			"requested_grant": map[string]any{"authorization_id": intent.AuthorizationID, "machine_id": intent.MachineID,
				"repositories": intent.Repositories, "permissions": intent.Permissions, "expires_at_unix": intent.ExpiresAtUnix}},
	}, &out)
	if problem != nil {
		return problem
	}
	if out.Token == "" || out.ExpiresAt.IsZero() {
		return exit.Named(exit.Conflict, "publication.authority_response_invalid", "Tensorhub did not acknowledge the requested publication authority")
	}
	return nil
}
