package cli

import (
	"net"
	"net/url"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Default selection is an intake fact. No tensor payload is acquired here, and
// unavailable defaults leave both unused imports and explicit overrides usable.
func (r *Resolver) captureMachineModelDefaults(capture localpackage.ExecutionCapture, request records.Request, publicOrigin string) (localpackage.ExecutionCapture, *exit.Error) {
	document := &pb.MachineExecutionCapture{}
	if err := canonical.Unmarshal(capture.Canonical, document); err != nil {
		return capture, exit.New(exit.Conflict, "captured model defaults have no exact code inventory")
	}
	for _, installation := range capture.Installations {
		iface, problem := launch.DecodePackageInterface(installation.PackageInterface)
		if problem != nil {
			return capture, problem
		}
		r.captureDefaultRows(document, installation.Package, installation.ID, iface, request, publicOrigin)
	}
	var err error
	capture.Canonical, capture.Digest, err = canonical.Identity(document)
	if err != nil || len(capture.Canonical) > 1<<20 {
		return capture, exit.New(exit.Validation, "captured Model defaults exceed the machine capture bound")
	}
	return capture, nil
}

func (r *Resolver) captureDefaultRows(document *pb.MachineExecutionCapture, pkg string, revision string, iface *launch.PackageInterface, request records.Request, publicOrigin string) {
	entries := map[string]bool{}
	for _, binding := range document.Bindings {
		if binding.CalleeInstallationId == revision {
			entries[binding.Entrypoint] = true
		}
	}
	for _, entry := range iface.Entrypoints {
		if !entries[entry.Name] {
			continue
		}
		for _, slot := range entry.Models {
			row := &pb.MachineModelDefault{CalleeInstallationId: revision, Entrypoint: entry.Name, Parameter: slot.Param}
			row.PublicOrigin, row.Rungs, row.UnavailableCode = r.captureDefaultLadder(pkg, entry.Name, slot, request, publicOrigin)
			document.ModelDefaults = append(document.ModelDefaults, row)
		}
	}
	sort.Slice(document.ModelDefaults, func(i, j int) bool {
		a, b := document.ModelDefaults[i], document.ModelDefaults[j]
		if a.CalleeInstallationId != b.CalleeInstallationId {
			return a.CalleeInstallationId < b.CalleeInstallationId
		}
		if a.Entrypoint != b.Entrypoint {
			return a.Entrypoint < b.Entrypoint
		}
		return a.Parameter < b.Parameter
	})
}

func (r *Resolver) captureDefaultLadder(pkg, entrypoint string, slot launch.Slot, request records.Request, publicOrigin string) (string, []*pb.MachineModelDefaultRung, string) {
	selected, problem := r.childModelLadder(request.Hub, pkg, entrypoint, slot)
	if problem != nil {
		code := "model_default_unavailable"
		if problem.ErrName() == "child.model_unbound" {
			code = "model_default_unbound"
		}
		return "", nil, code
	}
	origin := strings.TrimRight(r.cfg.ForHub(request.Hub).HubURL, "/")
	if publicOrigin != "" {
		origin = publicOrigin
	}
	parsed, err := url.Parse(origin)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" ||
		(parsed.Scheme != "https" && (request.Rental || parsed.Scheme != "http" || !(parsed.Hostname() == "localhost" || net.ParseIP(parsed.Hostname()).IsLoopback()))) {
		return "", nil, "model_default_origin_unsupported"
	}
	// The read probe deliberately has no configured token or machine-key provider.
	// Public card metadata alone does not establish anonymous byte-read authority.
	config := r.cfg
	config.HubURL = origin
	config.HubToken, config.HubTokenSource = secret.New(""), "unset"
	public := hub.New(config, "cozy-captured-model-defaults")
	ref, problem := hub.ParseRef(selected.Model)
	if problem != nil {
		return "", nil, "model_default_unavailable"
	}
	ctx, cancel := hub.Context()
	defer cancel()
	count := 0
	for _, model := range request.Models {
		if model.Package == pkg && model.BindingSlot() == slot.Path {
			count = model.GPUs
			break
		}
	}
	rungs := make([]*pb.MachineModelDefaultRung, 0, len(selected.Ladder))
	for _, rung := range selected.Ladder {
		if uint64(rung.GPUs) > uint64(^uint32(0)) {
			return "", nil, "model_default_unavailable"
		} // protobuf uint32 representation must not wrap
		if count > 0 && rung.GPUs > 0 && rung.GPUs != count {
			continue
		}
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
		rungs = append(rungs, &pb.MachineModelDefaultRung{Gpu: rung.GPU, Gpus: uint32(rung.GPUs), Repository: selected.Model,
			Manifest: &pb.Ref{Digest: digest, Length: uint64(resolved.ManifestLength)}})
	}
	if len(rungs) == 0 {
		return "", nil, "model_default_unavailable"
	}
	return origin, rungs, ""
}
