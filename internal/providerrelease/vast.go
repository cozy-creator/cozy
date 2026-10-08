// Package providerrelease owns the explicit development escape hatch for ending
// a provider resource with no Hub rental record. It cannot search or rent machines.
package providerrelease

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/secret"
)

const VastOrigin = "https://console.vast.ai"

type Result struct {
	Provider       string `json:"provider"`
	ResourceID     int64  `json:"provider_resource_id"`
	Label          string `json:"label"`
	State          string `json:"state"`
	Changed        bool   `json:"changed"`
	ProviderAbsent bool   `json:"provider_absent"`
}

type vastInstance struct {
	ID    int64  `json:"id"`
	Label string `json:"label"`
}

// Vast destroys exactly the requested instance after verifying its provider label.
// A stopped instance is still present; only an authenticated 404 proves absence.
func Vast(ctx context.Context, origin string, id int64, expectedLabel string, token secret.Value) (Result, *exit.Error) {
	result := Result{Provider: "vast", ResourceID: id, Label: expectedLabel}
	if id <= 0 || expectedLabel == "" || !token.Present() {
		return result, exit.New(exit.Validation, "name a positive provider resource id, its expected label, and a provider credential")
	}
	base, problem := config.HubOrigin(origin)
	if problem != nil {
		return result, exit.New(exit.Validation, "invalid provider API origin")
	}
	u, err := url.Parse(base)
	if err != nil || u.Path != "" {
		return result, exit.New(exit.Validation, "provider API origin must not contain a path")
	}
	if u.Scheme != "https" && (net.ParseIP(u.Hostname()) == nil || !net.ParseIP(u.Hostname()).IsLoopback()) {
		return result, exit.New(exit.Validation, "provider API requires HTTPS except at a literal loopback address")
	}
	client := &http.Client{Timeout: 30 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	endpoint := base + "/api/v0/instances/" + strconv.FormatInt(id, 10) + "/"
	request := func(method string) (int, []byte, *exit.Error) {
		req, err := http.NewRequestWithContext(ctx, method, endpoint, nil)
		if err != nil {
			return 0, nil, exit.Internalf("cannot construct provider request")
		}
		req.Header.Set("Authorization", "Bearer "+token.Reveal())
		req.Header.Set("Accept", "application/json")
		req.Header.Set("User-Agent", "cozy-provider-release/1")
		response, err := client.Do(req)
		if err != nil {
			return 0, nil, exit.New(exit.Unavailable, "provider %s could not be observed for instance %d", method, id)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(io.LimitReader(response.Body, (1<<20)+1))
		if err != nil || len(body) > 1<<20 {
			return response.StatusCode, nil, exit.New(exit.Unavailable, "provider response is unreadable")
		}
		if response.StatusCode == http.StatusNotFound {
			return response.StatusCode, nil, nil
		}
		if response.StatusCode == http.StatusUnauthorized || response.StatusCode == http.StatusForbidden {
			return response.StatusCode, nil, exit.New(exit.Credential, "provider credential was refused")
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			return response.StatusCode, nil, exit.New(exit.Unavailable, "provider %s returned HTTP %d for instance %d", method, response.StatusCode, id)
		}
		var envelope struct {
			Success *bool `json:"success"`
		}
		if json.Unmarshal(body, &envelope) == nil && envelope.Success != nil && !*envelope.Success {
			return response.StatusCode, nil, exit.New(exit.Failed, "provider refused %s for instance %d", method, id)
		}
		return response.StatusCode, body, nil
	}
	read := func() (bool, *exit.Error) {
		status, body, problem := request(http.MethodGet)
		if problem != nil {
			return false, problem
		}
		if status == http.StatusNotFound {
			return true, nil
		}
		var envelope struct {
			Instance *vastInstance `json:"instances"`
		}
		if json.Unmarshal(body, &envelope) != nil || envelope.Instance == nil || envelope.Instance.ID != id {
			return false, exit.New(exit.Unavailable, "provider did not identify requested instance %d", id)
		}
		if envelope.Instance.Label != expectedLabel {
			return false, exit.Named(exit.Conflict, "provider.resource_identity_changed", "provider instance %d does not have the expected label", id)
		}
		return false, nil
	}
	absent, problem := read()
	if problem != nil {
		return result, problem
	}
	if !absent {
		status, _, problem := request(http.MethodDelete)
		if problem != nil {
			return result, problem.WithRemedy("the release may be unconfirmed; retry the same provider resource id, never create a replacement")
		}
		result.Changed = status != http.StatusNotFound
		for {
			absent, problem = read()
			if problem != nil {
				return result, problem
			}
			if absent {
				break
			}
			select {
			case <-ctx.Done():
				return result, exit.Named(exit.Unavailable, "provider.release_unconfirmed", "release of provider instance %d was requested but absence is not confirmed", id).WithRemedy("repeat the same command to reconcile this instance")
			case <-time.After(time.Second):
			}
		}
	}
	result.State, result.ProviderAbsent = "absent", true
	return result, nil
}

// Token reads only explicitly supplied standard input, never repository dotenv or
// ambient provider configuration. Error text never contains the credential.
func Token(input io.Reader) (secret.Value, *exit.Error) {
	raw, err := io.ReadAll(io.LimitReader(input, 8193))
	if err != nil || len(raw) > 8192 {
		return secret.Value{}, exit.New(exit.Credential, "provider credential input is unreadable or exceeds 8192 bytes")
	}
	value := strings.TrimSpace(string(raw))
	if value == "" || strings.ContainsAny(value, " \t\r\n") {
		return secret.Value{}, exit.New(exit.Credential, "supply exactly one nonempty provider API key on standard input")
	}
	return secret.New(value), nil
}
