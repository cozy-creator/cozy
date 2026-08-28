// Package remotecontrol validates Tensorhub's persisted acquisition-attempt
// control snapshot and projects only the facts Creator needs to own the remote
// worker. It never reads a local install, invokes Runtime, or re-renders a plan.
package remotecontrol

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/hub"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

const (
	format          = "tensorhub.rental_control_snapshot/1"
	bindingFormat   = "tensorhub.endpoint_binding_release/1"
	planFormat      = "cozy.endpoint.EntrypointBindingPlan/1"
	rmbFormat       = "cozy.endpoint.ResolvedModelBinding/1"
	bundleFormat    = "tensorhub.endpoint_bundle/1"
	maxSnapshotSize = 64 << 20
)

type bindingDocument struct {
	CanonicalBytes []byte `json:"canonical_bytes"`
	Digest         string `json:"digest"`
	Kind           string `json:"kind"`
	Length         int64  `json:"length"`
}

type snapshot struct {
	AcquisitionAttemptID        string                   `json:"acquisition_attempt_id"`
	ArtifactObjectSet           hub.ExactControlDocument `json:"artifact_object_set"`
	BindingDocuments            []bindingDocument        `json:"binding_documents"`
	BindingRelease              hub.ExactControlDocument `json:"binding_release"`
	Descriptor                  hub.ExactControlDocument `json:"descriptor"`
	EndpointBundle              hub.ExactControlDocument `json:"endpoint_bundle"`
	EndpointExecutionDigest     string                   `json:"endpoint_execution_digest"`
	EnvironmentSpec             hub.ExactControlDocument `json:"environment_spec"`
	EvaluatedConfig             hub.ExactControlDocument `json:"evaluated_config"`
	Format                      string                   `json:"format"`
	InstalledEnvironmentReceipt hub.ExactControlDocument `json:"installed_environment_receipt"`
	PlacementSet                hub.ExactControlDocument `json:"placement_set"`
}

// Facts is the verified remote-only projection. Descriptor is used for payload
// typing; Placement is handed directly to the orchestrator.
type Facts struct {
	Descriptor              *launch.Descriptor
	Placement               orchestrator.DesiredPlacement
	EndpointExecutionDigest string
	ArtifactObjectSetDigest string
	ModelRootDigests        []string
}

type exactRef struct {
	Digest string
	Length int64
}

func validateExact(name string, document hub.ExactControlDocument) *exit.Error {
	if document.Length <= 0 || document.Length > canonical.DocMax ||
		int64(len(document.CanonicalBytes)) != document.Length {
		return invalid("%s length is %d for %d bytes", name, document.Length, len(document.CanonicalBytes))
	}
	sum := sha256.Sum256(document.CanonicalBytes)
	if document.Digest != "sha256:"+hex.EncodeToString(sum[:]) {
		return invalid("%s bytes do not match digest %s", name, document.Digest)
	}
	return nil
}

func invalid(format string, args ...any) *exit.Error {
	return exit.Named(exit.Conflict, "rental.control_snapshot_invalid", format, args...).
		WithRemedy("release this rental; Creator will not rebuild remote control truth from a local install")
}

func decodeSnapshot(exact hub.ExactControlDocument) (snapshot, *exit.Error) {
	var out snapshot
	if exact.Length <= 0 || exact.Length > maxSnapshotSize || int64(len(exact.CanonicalBytes)) != exact.Length {
		return out, invalid("control snapshot length is %d for %d bytes", exact.Length, len(exact.CanonicalBytes))
	}
	sum := sha256.Sum256(exact.CanonicalBytes)
	if exact.Digest != "sha256:"+hex.EncodeToString(sum[:]) {
		return out, invalid("control snapshot bytes do not match digest %s", exact.Digest)
	}
	decoder := json.NewDecoder(bytes.NewReader(exact.CanonicalBytes))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&out); err != nil {
		return out, invalid("control snapshot is not the closed document: %s", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return out, invalid("control snapshot has trailing JSON")
	}
	again, err := json.Marshal(out)
	if err != nil || !bytes.Equal(again, exact.CanonicalBytes) {
		return out, invalid("control snapshot is not its exact canonical encoding")
	}
	if out.Format != format || out.AcquisitionAttemptID == "" || out.EndpointExecutionDigest == "" ||
		len(out.BindingDocuments) == 0 {
		return out, invalid("control snapshot header is incomplete")
	}
	return out, nil
}

func requireKeys(doc canonical.Doc, names ...string) error {
	want := map[string]bool{}
	for _, name := range names {
		want[name] = true
	}
	for name := range doc {
		if !want[name] {
			return fmt.Errorf("unknown field %q", name)
		}
		delete(want, name)
	}
	if len(want) != 0 {
		missing := make([]string, 0, len(want))
		for name := range want {
			missing = append(missing, name)
		}
		sort.Strings(missing)
		return fmt.Errorf("missing fields %s", strings.Join(missing, ","))
	}
	return nil
}

func refOf(doc canonical.Doc) (exactRef, error) {
	if err := requireKeys(doc, "digest", "length"); err != nil {
		return exactRef{}, err
	}
	ref := exactRef{Digest: doc.Str("digest"), Length: doc.Int("length")}
	if ref.Digest == "" || ref.Length <= 0 {
		return exactRef{}, fmt.Errorf("incomplete ref")
	}
	return ref, nil
}

func sameRef(ref exactRef, document hub.ExactControlDocument) bool {
	return ref.Digest == document.Digest && ref.Length == document.Length
}

func exactOf(document bindingDocument) hub.ExactControlDocument {
	return hub.ExactControlDocument{
		CanonicalBytes: document.CanonicalBytes, Digest: document.Digest, Length: document.Length,
	}
}

// Decode verifies the complete snapshot closure and returns the remote
// placement. endpointRef is the provider-neutral product ref persisted with the
// rental, not a local installation key.
func Decode(control hub.ExactControlDocument, endpointRef string) (Facts, *exit.Error) {
	var facts Facts
	s, e := decodeSnapshot(control)
	if e != nil {
		return facts, e
	}
	for name, document := range map[string]hub.ExactControlDocument{
		"artifact_object_set": s.ArtifactObjectSet,
		"binding_release":     s.BindingRelease, "descriptor": s.Descriptor,
		"endpoint_bundle": s.EndpointBundle, "environment_spec": s.EnvironmentSpec,
		"evaluated_config":              s.EvaluatedConfig,
		"installed_environment_receipt": s.InstalledEnvironmentReceipt,
		"placement_set":                 s.PlacementSet,
	} {
		if e := validateExact(name, document); e != nil {
			return facts, e
		}
	}

	documents := map[string]bindingDocument{}
	priorKind, priorDigest := "", ""
	for i, document := range s.BindingDocuments {
		if !validKind(document.Kind) || documents[document.Digest].Digest != "" ||
			priorKind > document.Kind || priorKind == document.Kind && priorDigest >= document.Digest {
			return facts, invalid("binding document %d is unknown, duplicated, or unsorted", i)
		}
		if e := validateExact("binding document "+document.Digest, exactOf(document)); e != nil {
			return facts, e
		}
		documents[document.Digest] = document
		priorKind, priorDigest = document.Kind, document.Digest
	}

	release, err := canonical.ReadObject(s.BindingRelease.CanonicalBytes)
	if err != nil || requireKeys(release, "descriptor", "documents", "endpoint_execution_digest",
		"environment_spec", "format", "installed_environment_receipt", "object_set_digest") != nil ||
		release.Str("format") != bindingFormat || release.Str("endpoint_execution_digest") != s.EndpointExecutionDigest {
		return facts, invalid("binding release is not the exact closed release: %v", err)
	}
	descriptorRef, err := refOf(release.Sub("descriptor"))
	if err != nil || !sameRef(descriptorRef, s.Descriptor) {
		return facts, invalid("binding release descriptor does not match the snapshot descriptor")
	}
	environmentRef, err := refOf(release.Sub("environment_spec"))
	if err != nil || !sameRef(environmentRef, s.EnvironmentSpec) {
		return facts, invalid("binding release environment does not match the snapshot environment")
	}
	receiptRef, err := refOf(release.Sub("installed_environment_receipt"))
	if err != nil || !sameRef(receiptRef, s.InstalledEnvironmentReceipt) {
		return facts, invalid("binding release receipt does not match the snapshot receipt")
	}
	if release.Str("object_set_digest") != s.ArtifactObjectSet.Digest {
		return facts, invalid("binding release object set does not match the snapshot object set")
	}
	objectSet, err := canonical.ReadObject(s.ArtifactObjectSet.CanonicalBytes)
	if err != nil || requireKeys(objectSet, "kind", "roots") != nil ||
		objectSet.Str("kind") != "tensorhub.resolved_object_set/1" {
		return facts, invalid("artifact object set is not the closed canonical document: %v", err)
	}
	for _, root := range objectSet.List("roots") {
		if err := requireKeys(root, "objects", "root"); err != nil {
			return facts, invalid("artifact object set root is not closed: %v", err)
		}
		ref, err := refOf(root.Sub("root"))
		if err != nil {
			return facts, invalid("artifact object set root is malformed: %v", err)
		}
		facts.ModelRootDigests = append(facts.ModelRootDigests, ref.Digest)
	}

	releaseKinds := map[string]string{}
	for _, row := range release.List("documents") {
		if err := requireKeys(row, "kind", "ref"); err != nil {
			return facts, invalid("binding release document row is not closed: %s", err)
		}
		ref, err := refOf(row.Sub("ref"))
		stored := documents[ref.Digest]
		if err != nil || releaseKinds[ref.Digest] != "" || stored.Digest == "" ||
			stored.Kind != row.Str("kind") ||
			stored.Length != ref.Length {
			return facts, invalid("binding release document %s is absent, changed, or duplicated", ref.Digest)
		}
		releaseKinds[ref.Digest] = stored.Kind
	}
	if len(releaseKinds) != len(documents) {
		return facts, invalid("snapshot binding closure has extra documents")
	}

	set, err := canonical.Read(s.PlacementSet.CanonicalBytes, &pb.PlacementSet{})
	if err != nil || len(set.List("placements")) != 1 {
		return facts, invalid("placement set is not one exact canonical placement: %v", err)
	}
	placement := set.List("placements")[0]
	if placement.Str("placement_id") != s.AcquisitionAttemptID {
		return facts, invalid("placement id %q does not equal acquisition attempt %q",
			placement.Str("placement_id"), s.AcquisitionAttemptID)
	}
	placementSpec := placement.Sub("spec")
	if placementSpec.Str("descriptor_digest") != s.Descriptor.Digest ||
		placementSpec.Str("environment_spec_digest") != s.EnvironmentSpec.Digest ||
		placementSpec.Str("installed_environment_receipt_digest") != s.InstalledEnvironmentReceipt.Digest {
		return facts, invalid("placement fixed digests disagree with the exact snapshot documents")
	}

	planSubjects := map[string]int64{}
	for _, subject := range placementSpec.List("binding_plans") {
		if subject.Str("kind") != "plan" || subject.Str("subject_id") != subject.Str("digest") ||
			subject.Str("digest") == "" || subject.Int("length") <= 0 ||
			planSubjects[subject.Str("digest")] != 0 {
			return facts, invalid("placement carries a malformed binding-plan subject")
		}
		planSubjects[subject.Str("digest")] = subject.Int("length")
	}
	if len(planSubjects) == 0 {
		return facts, invalid("placement carries no binding plan")
	}
	for digest, kind := range releaseKinds {
		if kind == "entrypoint_binding_plan" {
			if planSubjects[digest] != documents[digest].Length {
				return facts, invalid("placement plan set differs from the binding release")
			}
			delete(planSubjects, digest)
		}
	}
	if len(planSubjects) != 0 {
		return facts, invalid("placement reaches a plan outside the binding release")
	}

	bundle, err := canonical.ReadObject(s.EndpointBundle.CanonicalBytes)
	if err != nil || requireKeys(bundle, "descriptor", "endpoint_release_id", "evaluated_config", "format",
		"project_wheel", "resolved_wheel_set", "wheelhouse_manifest") != nil ||
		bundle.Str("format") != bundleFormat || bundle.Str("endpoint_release_id") != placementSpec.Str("endpoint_release_id") {
		return facts, invalid("endpoint bundle disagrees with the placement: %v", err)
	}
	bundleDescriptor, err := refOf(bundle.Sub("descriptor"))
	if err != nil || !sameRef(bundleDescriptor, s.Descriptor) {
		return facts, invalid("endpoint bundle descriptor differs from the exact descriptor")
	}
	configRef, err := refOf(bundle.Sub("evaluated_config"))
	if err != nil || !sameRef(configRef, s.EvaluatedConfig) {
		return facts, invalid("endpoint bundle evaluated config differs from the exact config")
	}

	environment, err := canonical.Read(s.EnvironmentSpec.CanonicalBytes, &pb.EndpointEnvironmentSpec{})
	if err != nil || environment.Str("endpoint_bundle_digest") != s.EndpointBundle.Digest {
		return facts, invalid("environment spec does not close the endpoint bundle: %v", err)
	}
	receipt, err := canonical.ReadObject(s.InstalledEnvironmentReceipt.CanonicalBytes)
	if err != nil || requireKeys(receipt, "distributions", "endpoint_bundle_digest",
		"environment_spec_digest", "format", "platform_target", "project_wheel_digest",
		"wheelhouse_manifest_digest") != nil ||
		receipt.Str("format") != "cozy.worker.v1.InstalledEnvironmentReceipt/1" ||
		receipt.Str("endpoint_bundle_digest") != s.EndpointBundle.Digest ||
		receipt.Str("environment_spec_digest") != s.EnvironmentSpec.Digest ||
		receipt.Str("project_wheel_digest") != environment.Str("project_wheel_digest") ||
		receipt.Str("wheelhouse_manifest_digest") != environment.Str("wheelhouse_manifest_digest") {
		return facts, invalid("installed-environment receipt does not close the selected environment: %v", err)
	}

	descriptor, problem := launch.DecodeDescriptor(s.Descriptor.CanonicalBytes)
	if problem != nil || descriptor.Digest != s.Descriptor.Digest {
		return facts, invalid("endpoint descriptor is unreadable or has the wrong digest: %v", problem)
	}
	visible := map[string]*launch.Entrypoint{}
	for i := range descriptor.Entrypoints {
		ep := &descriptor.Entrypoints[i]
		if !ep.Hidden {
			if visible[ep.Name] != nil {
				return facts, invalid("descriptor repeats visible entrypoint %q", ep.Name)
			}
			visible[ep.Name] = ep
		}
	}
	reachable := map[string]bool{}
	referencedRMB := map[string]bool{}
	boundEntrypoints := map[string]bool{}
	bindings := []*orchestrator.Binding{}
	for digest, document := range documents {
		if document.Kind != "entrypoint_binding_plan" {
			continue
		}
		plan, err := canonical.ReadObject(document.CanonicalBytes)
		if err != nil || requireKeys(plan, "bindings", "descriptor", "entrypoint", "format") != nil ||
			plan.Str("format") != planFormat {
			return facts, invalid("binding plan %s is not the closed canonical plan: %v", digest, err)
		}
		planDescriptor, err := refOf(plan.Sub("descriptor"))
		entrypoint := plan.Str("entrypoint")
		ep := visible[entrypoint]
		if err != nil || !sameRef(planDescriptor, s.Descriptor) || ep == nil ||
			boundEntrypoints[entrypoint] {
			return facts, invalid("binding plan %s disagrees with the descriptor", digest)
		}
		boundEntrypoints[entrypoint] = true
		slots := map[string]bool{}
		for _, slot := range plan.List("bindings") {
			slotName := slot.Str("slot")
			if err := requireKeys(slot, "binding", "slot"); err != nil || slotName == "" ||
				slots[slotName] {
				return facts, invalid("binding plan %s has a malformed slot", digest)
			}
			slots[slotName] = true
			ref, err := refOf(slot.Sub("binding"))
			if err != nil || releaseKinds[ref.Digest] != "resolved_model_binding" ||
				documents[ref.Digest].Length != ref.Length {
				return facts, invalid("binding plan %s reaches an absent resolved binding", digest)
			}
			referencedRMB[ref.Digest] = true
		}
		reachable[digest] = true
		bindings = append(bindings, &orchestrator.Binding{
			Entrypoint: entrypoint, Outputs: launch.AssetPaths(ep.Result),
			RuntimePlan: &orchestrator.BindingPlanSubject{
				SubjectID: digest, Kind: "plan", Digest: digest, Length: uint64(document.Length),
				CanonicalBytes: append([]byte(nil), document.CanonicalBytes...),
			},
		})
	}
	if len(bindings) != len(visible) {
		return facts, invalid("binding plans cover %d visible entrypoints, descriptor has %d", len(bindings), len(visible))
	}
	for digest := range referencedRMB {
		document := documents[digest]
		rmb, err := canonical.ReadObject(document.CanonicalBytes)
		if err != nil || requireKeys(rmb, "checkpoint", "config", "execution_layout", "format",
			"model_construction_contract", "stamps") != nil || rmb.Str("format") != rmbFormat {
			return facts, invalid("resolved binding %s is not closed canonical control: %v", digest, err)
		}
		mcc, err := refOf(rmb.Sub("model_construction_contract"))
		config, configErr := refOf(rmb.Sub("config").Sub("document"))
		if err != nil || configErr != nil || releaseKinds[mcc.Digest] != "model_construction_contract" ||
			releaseKinds[config.Digest] != "model_config" || documents[mcc.Digest].Length != mcc.Length ||
			documents[config.Digest].Length != config.Length {
			return facts, invalid("resolved binding %s reaches an absent MCC or config", digest)
		}
		reachable[digest], reachable[mcc.Digest], reachable[config.Digest] = true, true, true
	}
	if len(reachable) != len(documents) {
		return facts, invalid("binding release contains control documents no plan reaches")
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].Entrypoint < bindings[j].Entrypoint })

	parts := strings.Split(strings.TrimSpace(endpointRef), "/")
	major := uint64(0)
	var majorErr error
	if len(parts) == 4 && strings.HasPrefix(parts[2], "v") {
		major, majorErr = strconv.ParseUint(strings.TrimPrefix(parts[2], "v"), 10, 32)
	}
	repo, repoErr := hub.ParseRef(strings.Join(parts[:min(len(parts), 2)], "/"))
	if len(parts) != 4 || repoErr != nil || majorErr != nil || major == 0 ||
		parts[2] != fmt.Sprintf("v%d", major) || parts[3] == "" || visible[parts[3]] == nil {
		return facts, invalid("rental endpoint_ref %q is not covered by the exact descriptor", endpointRef)
	}
	facts.Descriptor = descriptor
	facts.EndpointExecutionDigest = s.EndpointExecutionDigest
	facts.ArtifactObjectSetDigest = s.ArtifactObjectSet.Digest
	facts.Placement = orchestrator.DesiredPlacement{
		Endpoint: repo.String(), ReleaseID: placementSpec.Str("endpoint_release_id"),
		// No InstallID: a remote placement resolves from Tensorhub's attempt, never a
		// local install row; the attempt id rides PlacementIDValue.
		DescriptorDigest: s.Descriptor.Digest,
		Bindings:         bindings, EnvironmentSpecDigest: s.EnvironmentSpec.Digest,
		InstalledEnvironmentReceiptDigest: s.InstalledEnvironmentReceipt.Digest,
		ConfigDigest:                      s.EvaluatedConfig.Digest, ExactPlacementSetDigest: s.PlacementSet.Digest,
		ExactPlacementSetBytes: append([]byte(nil), s.PlacementSet.CanonicalBytes...),
		PlacementIDValue:       s.AcquisitionAttemptID,
	}
	return facts, nil
}

func validKind(kind string) bool {
	switch kind {
	case "entrypoint_binding_plan", "resolved_model_binding", "model_construction_contract", "model_config":
		return true
	default:
		return false
	}
}
