package cli

import (
	"runtime/debug"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
)

// localConfigDigest freezes the evaluated local config used by InvocationSpec.
// Environment identity comes from the exact Hub-selected PlacementSet.
func localConfigDigest(cfg config.Config) string {
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
	return spellOf(map[string]canonical.Value{
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
		"service_build":                 build,
	})
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
