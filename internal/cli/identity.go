package cli

import (
	"runtime"
	"runtime/debug"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
)

// localInvocationIdentity mints the two per-daemon identity digests every LOCAL
// InvocationSpec rides (cl-022's guard): the execution environment's and the evaluated
// configuration's. They used to be the orchestrator Options' empty strings, so every
// local InvocationSpec froze `environment_spec_digest: ""` — an UNDER-SPECIFIED identity
// persisted forever under the request's digest.
//
// These are digest preimages, not stored documents. A closed `kind` value separates the
// two meanings without inventing two versioned file formats. They are frozen per daemon
// run, never per request — a request cannot choose the environment it runs under — and
// the config's one secret enters as its DIGEST, never raw.
func localInvocationIdentity(cfg config.Config) (environmentSpec, configDigest string) {
	build := "unknown"
	if info, ok := debug.ReadBuildInfo(); ok {
		if info.Main.Version != "" && info.Main.Version != "(devel)" {
			build = info.Main.Version
		}
		for _, s := range info.Settings {
			if s.Key == "vcs.revision" && s.Value != "" {
				build = s.Value
			}
		}
	}
	environmentSpec = spellOf(map[string]canonical.Value{
		"kind":          "execution_environment",
		"os":            runtime.GOOS,
		"arch":          runtime.GOARCH,
		"service_build": build,
	})
	configDigest = spellOf(map[string]canonical.Value{
		"kind":             "evaluated_config",
		"home":             cfg.Home,
		"port":             int64(cfg.Port),
		"yield":            cfg.Yield,
		"hub_url":          cfg.HubURL,
		"hub_url_source":   cfg.HubURLSource,
		"hub_token":        cfg.HubToken.Digest(),
		"hub_token_source": cfg.HubTokenSource,
		// The tfs VALUE stays out: an executable path is resolution rather than identity.
		// Its provenance is the config fact frozen into the child environment.
		"tfs_source":                    cfg.TfsSource,
		"local_rate_micro_usd_per_hour": cfg.LocalRateMicroUSDPerHour,
	})
	return environmentSpec, configDigest
}

func spellOf(doc map[string]canonical.Value) string {
	data, err := canonical.Write(doc)
	if err != nil {
		// Unreachable by construction: every value above is a string or an int64.
		panic("the local identity document cannot be canonicalized: " + err.Error())
	}
	spelled, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		panic("the local identity digest cannot be spelled: " + err.Error())
	}
	return spelled
}
