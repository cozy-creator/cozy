package machines

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

type executionAccess struct {
	Origin      string            `json:"origin"`
	Token       string            `json:"token"`
	ExpiresAt   int64             `json:"expires_at"`
	Environment map[string]string `json:"environment"`
	CA          string            `json:"ca_der_b64url,omitempty"`
	Certificate string            `json:"certificate_sha256"`
}

// attachAccess runs after authenticated readiness. Cached grants are scoped to a Hub,
// leaf and expiry, and replayed into the agent after a restart. Account credentials never
// cross the machine boundary. The cache is private and is never rendered as run data.
func (h *Host) attachAccess(ctx context.Context, launch *Launch, origin string, account *hub.Client) (string, *exit.Error) {
	all := map[string]executionAccess{}
	raw, err := os.ReadFile(h.path("execution-access.json"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", exit.Internalf("cannot read execution access cache: %s", err)
	}
	if err == nil && json.Unmarshal(raw, &all) != nil {
		return "", exit.New(exit.Conflict, "execution access cache is unreadable")
	}
	digest := sha256.Sum256(launch.Leaf)
	certificate := hex.EncodeToString(digest[:])
	grant := all[origin]
	if grant.Certificate != certificate || grant.ExpiresAt <= time.Now().Add(time.Minute).Unix() || grant.Token == "" {
		if account == nil {
			return "", exit.Named(exit.Credential, "machine.execution_access_required", "running work from %s requires account-authorized execution access", origin)
		}
		access, problem := account.AuthorizeExecutionAccess(ctx, launch.Leaf)
		if problem != nil {
			return "", problem
		}
		grant = executionAccess{Origin: access.Environment["TENSORHUB_ORIGIN"], Token: access.Token, ExpiresAt: access.ExpiresAt.Unix(), Environment: access.Environment, Certificate: certificate}
		if len(access.TrustRoot) > 0 {
			grant.CA = base64.RawURLEncoding.EncodeToString(access.TrustRoot)
		}
		all[origin] = grant
		raw, _ = json.Marshal(all)
		if err := writePrivate(h.path("execution-access.json"), raw); err != nil {
			return "", exit.Internalf("cannot retain execution access: %s", err)
		}
	}
	owner, problem := h.Owner()
	if problem != nil {
		return "", problem
	}
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	if err != nil || len(public) != ed25519.PublicKeySize {
		return "", exit.New(exit.Credential, "the machine owner key is unreadable")
	}
	token, err := capability.MintSigned(public, owner.Sign, capability.Grant{Machine: launch.WorkerID, Action: "hub-access", Expires: time.Now().Add(5 * time.Minute).Unix()})
	if err != nil {
		return "", exit.Internalf("cannot authorize machine execution access: %s", err)
	}
	// Do not send cache-only identity metadata as part of the access API.
	body, _ := json.Marshal(map[string]any{"origin": grant.Origin, "token": grant.Token, "expires_at": grant.ExpiresAt, "environment": grant.Environment, "ca_der_b64url": grant.CA})
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://"+launch.Addr+"/v1/hubs/access", bytes.NewReader(body))
	if err != nil {
		return "", exit.Internalf("cannot address machine access API: %s", err)
	}
	request.Header.Set("Authorization", "Cozy-Cap "+token)
	request.Header.Set("Content-Type", "application/json")
	pin, problem := h.Pin()
	if problem != nil {
		return "", problem
	}
	transport := &http.Transport{TLSClientConfig: pin.TLSConfig()}
	defer transport.CloseIdleConnections()
	response, err := (&http.Client{Transport: transport, Timeout: 30 * time.Second}).Do(request)
	if err != nil {
		return "", exit.Unavailablef("machine execution access could not be attached: %s", err)
	}
	defer response.Body.Close()
	// Refusals are typed and never echo a credential-bearing request or response body.
	raw, err = io.ReadAll(io.LimitReader(response.Body, 64<<10))
	if err != nil {
		return "", exit.Unavailablef("machine execution access response was interrupted")
	}
	if response.StatusCode/100 != 2 {
		return "", exit.Named(exit.Credential, "machine.execution_access_refused", "the machine refused execution access (HTTP %d)", response.StatusCode)
	}
	var reply struct {
		Origin    string `json:"origin"`
		ExpiresAt int64  `json:"expires_at"`
	}
	if json.Unmarshal(raw, &reply) != nil || reply.Origin != grant.Origin || reply.ExpiresAt != grant.ExpiresAt {
		return "", exit.New(exit.Conflict, "the machine did not acknowledge execution access")
	}
	return grant.Environment["TENSORHUB_ORIGIN"], nil
}
