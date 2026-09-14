package cli

import (
	"bytes"
	"net"
	"net/url"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Default selection is an intake fact. No tensor payload is acquired here, and
// unavailable defaults leave both unused imports and explicit overrides usable.
func (r *Resolver) captureMachineModelDefaults(capture localpackage.ExecutionCapture, rental bool) (localpackage.ExecutionCapture, *exit.Error) {
	document := &pb.MachineExecutionCapture{}
	if err := canonical.Unmarshal(capture.Canonical, document); err != nil {
		return capture, exit.New(exit.Conflict, "captured model defaults have no exact code inventory")
	}
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return capture, problem
	}
	for _, revision := range capture.Revisions {
		iface, problem := launch.ReadPackageInterface(filepath.Join(layout.LocalPackages,
			strings.TrimPrefix(revision.Digest, "sha256:"), launch.PackageInterfaceFile), revision.PackageInterfaceDigest)
		if problem != nil {
			return capture, problem
		}
		digest, err := canonical.Raw(revision.Digest)
		if err != nil {
			return capture, exit.New(exit.Conflict, "captured default revision changed")
		}
		r.captureDefaultRows(document, revision.Package, digest, iface, rental)
	}
	var err error
	capture.Canonical, capture.Digest, err = canonical.Identity(document)
	if err != nil || len(capture.Canonical) > 1<<20 {
		return capture, exit.New(exit.Validation, "captured Model defaults exceed the machine capture bound")
	}
	return capture, nil
}

func (r *Resolver) captureDefaultRows(document *pb.MachineExecutionCapture, pkg string, revision []byte, iface *launch.PackageInterface, rental bool) {
	entries := map[string]bool{}
	for _, binding := range document.Bindings {
		if bytes.Equal(binding.CalleeRevisionDigest, revision) {
			entries[binding.Entrypoint] = true
		}
	}
	for _, entry := range iface.Entrypoints {
		if !entries[entry.Name] {
			continue
		}
		for _, slot := range entry.Models {
			row := &pb.MachineModelDefault{CalleeRevisionDigest: revision, Entrypoint: entry.Name, Parameter: slot.Param}
			row.PublicOrigin, row.Rungs, row.UnavailableCode = r.captureDefaultLadder(pkg, entry.Name, slot, rental)
			document.ModelDefaults = append(document.ModelDefaults, row)
		}
	}
	sort.Slice(document.ModelDefaults, func(i, j int) bool {
		a, b := document.ModelDefaults[i], document.ModelDefaults[j]
		if compared := bytes.Compare(a.CalleeRevisionDigest, b.CalleeRevisionDigest); compared != 0 {
			return compared < 0
		}
		if a.Entrypoint != b.Entrypoint {
			return a.Entrypoint < b.Entrypoint
		}
		return a.Parameter < b.Parameter
	})
}

func (r *Resolver) captureDefaultLadder(pkg, entrypoint string, slot launch.Slot, rental bool) (string, []*pb.MachineModelDefaultRung, string) {
	selected, problem := r.childModelLadder(pkg, entrypoint, slot)
	if problem != nil {
		code := "model_default_unavailable"
		if problem.ErrName() == "child.model_unbound" {
			code = "model_default_unbound"
		}
		return "", nil, code
	}
	origin := strings.TrimRight(r.cfg.HubURL, "/")
	parsed, err := url.Parse(origin)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" ||
		(parsed.Scheme != "https" && (rental || parsed.Scheme != "http" || !(parsed.Hostname() == "localhost" || net.ParseIP(parsed.Hostname()).IsLoopback()))) {
		return "", nil, "model_default_origin_unsupported"
	}
	// The read probe deliberately has no configured token or machine-key provider.
	// Public card metadata alone does not establish anonymous byte-read authority.
	config := r.cfg
	config.HubToken, config.HubTokenSource = secret.New(""), "unset"
	public := hub.New(config, "cozy-captured-model-defaults")
	ref, problem := hub.ParseRef(selected.Model)
	if problem != nil {
		return "", nil, "model_default_unavailable"
	}
	ctx, cancel := hub.Context()
	defer cancel()
	rungs := make([]*pb.MachineModelDefaultRung, 0, len(selected.Ladder))
	for _, rung := range selected.Ladder {
		resolved, problem := public.ResolveModel(ctx, selected.Model+"@"+rung.Manifest, "")
		if problem != nil || resolved.Model != selected.Model || resolved.ManifestID != rung.Manifest || resolved.ManifestLength <= 0 || resolved.HeaderID == "" {
			return "", nil, "model_default_unavailable"
		}
		reads, problem := public.CheckpointReads(ctx, ref, rung.Manifest, []string{resolved.HeaderID})
		if problem != nil || len(reads) != 1 || reads[0].ObjectID != resolved.HeaderID || reads[0].Length <= 0 || reads[0].URL == "" {
			return "", nil, "model_default_unavailable"
		}
		digest, err := canonical.Raw(rung.Manifest)
		if err != nil {
			return "", nil, "model_default_unavailable"
		}
		rungs = append(rungs, &pb.MachineModelDefaultRung{Gpu: rung.GPU, Repository: selected.Model,
			Manifest: &pb.Ref{Digest: digest, Length: uint64(resolved.ManifestLength)}})
	}
	if len(rungs) == 0 {
		return "", nil, "model_default_unavailable"
	}
	return origin, rungs, ""
}
