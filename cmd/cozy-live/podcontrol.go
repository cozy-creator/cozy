package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/launch"
	"github.com/cozy-creator/cozy-creator/internal/records"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

// podControl is the stand-in hub's half of Tensorhub's ACQUISITION-ATTEMPT CONTROL
// SNAPSHOT (`tensorhub.rental_control_snapshot/1`). A real hub authors it before provider
// Create, from the release it installed on the pod; this stand-in authors the same closed
// document from the release the DRIVER installed, so the client's strict verifier
// (internal/remotecontrol) is exercised over real bytes and real digests rather than a
// stub. Every sub-document is the exact canonical bytes the snapshot names, and every
// cross-reference the verifier checks is satisfied by construction.
//
// The exact documents that have no local source (environment spec, installed-environment
// receipt, endpoint bundle, evaluated config, wheels) are MINIMAL but real: canonical bytes,
// sha256 identities, and the digest joins between them are all genuine.
type podControl struct {
	endpoint string // org/name
	release  string // endpoint release id the pod serves
	// entrypoints -> canonical EntrypointBindingPlan bytes, as the installed runtime
	// itself authors them for a weightless release (verified against `bindings`).
	plans map[string][]byte

	descriptor      hub.ExactControlDocument
	environmentSpec hub.ExactControlDocument
	receipt         hub.ExactControlDocument
	bundle          hub.ExactControlDocument
	config          hub.ExactControlDocument
	objectSet       hub.ExactControlDocument
	bindingRelease  hub.ExactControlDocument
	executionDigest string
	documents       []podBindingDocument
}

// The snapshot's own wire shape. Field order and tags mirror remotecontrol's closed
// decoder exactly: it re-marshals and demands byte equality.
type podBindingDocument struct {
	CanonicalBytes []byte `json:"canonical_bytes"`
	Digest         string `json:"digest"`
	Kind           string `json:"kind"`
	Length         int64  `json:"length"`
}

type podSnapshot struct {
	AcquisitionAttemptID        string                   `json:"acquisition_attempt_id"`
	ArtifactObjectSet           hub.ExactControlDocument `json:"artifact_object_set"`
	BindingDocuments            []podBindingDocument     `json:"binding_documents"`
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

// defaultPodControl is what every stand-in hub in a section provisions against unless its
// spec names another. A section sets it once from the driver root's own install.
var defaultPodControl *podControl

func exactOf(data []byte) hub.ExactControlDocument {
	sum := sha256.Sum256(data)
	return hub.ExactControlDocument{
		CanonicalBytes: data, Digest: "sha256:" + hex.EncodeToString(sum[:]), Length: int64(len(data)),
	}
}

func refOf(doc hub.ExactControlDocument) map[string]canonical.Value {
	return map[string]canonical.Value{"digest": doc.Digest, "length": doc.Length}
}

func rawDigest(doc hub.ExactControlDocument) []byte {
	raw, err := canonical.Raw(doc.Digest)
	must("spelling "+doc.Digest, err)
	return raw
}

func writeCanonical(what string, v canonical.Value) hub.ExactControlDocument {
	data, err := canonical.Write(v)
	must("authoring "+what, err)
	return exactOf(data)
}

// localPodControl reads the driver root's ACTIVE install of one endpoint and authors the
// fixed half of the control closure from it: the verified descriptor and the runtime's own
// canonical weightless plans. It runs in-process; the driver's env fence is COZY_HOME.
func localPodControl(root, endpointRef string) *podControl {
	parts := strings.Split(endpointRef, "/")
	must("endpoint ref", func() error {
		if len(parts) != 4 || !strings.HasPrefix(parts[2], "v") {
			return fmt.Errorf("%q is not org/name/vN/entrypoint", endpointRef)
		}
		return nil
	}())
	major, err := strconv.Atoi(strings.TrimPrefix(parts[2], "v"))
	must("endpoint major", err)
	endpoint := parts[0] + "/" + parts[1]

	must("setting COZY_HOME for the one env reader", os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	must("config", errOf(e))
	l, e := home.Open(cfg.Home)
	must("layout", errOf(e))
	st, e := records.Open(l.DB)
	must("records", errOf(e))
	defer st.Close()
	_, gen, e := st.ActivePin(endpoint, major)
	must("the active install of "+endpoint, errOf(e))
	if gen == nil {
		must("the active install of "+endpoint, fmt.Errorf("no active pin for %s v%d", endpoint, major))
	}
	facts, e := launch.Read(*gen, cfg.Home, cfg.Tool())
	must("reading the install", errOf(e))
	_, weightless, e := facts.RuntimeCLI.Bindings()
	must("the runtime's bindings", errOf(e))
	reported := map[string]launch.WeightlessPlan{}
	for _, p := range weightless {
		reported[p.Entrypoint] = p
	}

	c := &podControl{endpoint: endpoint, release: launch.ReleaseID(*gen), plans: map[string][]byte{}}
	c.descriptor = exactOf(facts.Descriptor.Raw)
	if c.descriptor.Digest != facts.Descriptor.Digest || c.descriptor.Digest != gen.Descriptor {
		must("the descriptor identity", fmt.Errorf("%s / %s / install %s disagree",
			c.descriptor.Digest, facts.Descriptor.Digest, gen.Descriptor))
	}
	// THE PLANS ARE THE RUNTIME'S. The document is authored here in the runtime's exact
	// shape and then CHECKED against the identity the installed runtime reported for the
	// same entrypoint, so a drift between the two writers is a refusal and never a green
	// snapshot carrying a plan the pod would not recognise.
	for _, ep := range facts.Descriptor.Entrypoints {
		if ep.Hidden {
			continue
		}
		plan := writeCanonical("plan "+ep.Name, map[string]canonical.Value{
			"bindings":   []canonical.Value{},
			"descriptor": refOf(c.descriptor),
			"entrypoint": ep.Name,
			"format":     "cozy.endpoint.EntrypointBindingPlan/1",
		})
		got, ok := reported[ep.Name]
		if !ok || got.Digest != plan.Digest || got.SubjectID != plan.Digest || int64(got.Length) != plan.Length {
			must("plan identity for "+ep.Name, fmt.Errorf("authored %s (%d B), runtime reported %s/%s (%d B)",
				plan.Digest, plan.Length, got.SubjectID, got.Digest, got.Length))
		}
		c.plans[ep.Name] = plan.CanonicalBytes
	}
	c.authorClosure()
	return c
}

// authorClosure mints the attempt-independent documents in dependency order. Each digest
// below is the sha256 of the bytes that sit beside it in the snapshot.
func (c *podControl) authorClosure() {
	platform := &pb.PlatformTarget{
		OsArch: "linux/amd64", Libc: "glibc", PythonAbi: "cp312",
		AcceleratorBackend: "cuda", AcceleratorAbi: "cu124",
	}
	platformDoc := map[string]canonical.Value{
		"os_arch": platform.OsArch, "libc": platform.Libc, "python_abi": platform.PythonAbi,
		"accelerator_backend": platform.AcceleratorBackend, "accelerator_abi": platform.AcceleratorAbi,
	}
	c.config = writeCanonical("evaluated config", map[string]canonical.Value{
		"format": "tensorhub.evaluated_config/1", "values": map[string]canonical.Value{},
	})
	projectWheel := exactOf([]byte("cozy-live stand-in project wheel for " + c.release))
	wheelhouse := writeCanonical("wheelhouse manifest", map[string]canonical.Value{
		"format": "tensorhub.wheelhouse_manifest/1", "wheels": []canonical.Value{},
	})
	resolvedWheels := writeCanonical("resolved wheel set", map[string]canonical.Value{
		"format": "tensorhub.resolved_wheel_set/1", "wheels": []canonical.Value{},
	})
	c.bundle = writeCanonical("endpoint bundle", map[string]canonical.Value{
		"descriptor":          refOf(c.descriptor),
		"endpoint_release_id": c.release,
		"evaluated_config":    refOf(c.config),
		"format":              "tensorhub.endpoint_bundle/1",
		"project_wheel":       refOf(projectWheel),
		"resolved_wheel_set":  refOf(resolvedWheels),
		"wheelhouse_manifest": refOf(wheelhouse),
	})
	envBytes, err := canonical.Bytes(&pb.EndpointEnvironmentSpec{
		PlatformTarget: platform, EndpointBundleDigest: rawDigest(c.bundle),
		ProjectWheelDigest: rawDigest(projectWheel), WheelhouseManifestDigest: rawDigest(wheelhouse),
	})
	must("authoring the environment spec", err)
	c.environmentSpec = exactOf(envBytes)
	c.receipt = writeCanonical("installed-environment receipt", map[string]canonical.Value{
		"distributions":              []canonical.Value{},
		"endpoint_bundle_digest":     c.bundle.Digest,
		"environment_spec_digest":    c.environmentSpec.Digest,
		"format":                     "cozy.worker.v1.InstalledEnvironmentReceipt/1",
		"platform_target":            platformDoc,
		"project_wheel_digest":       projectWheel.Digest,
		"wheelhouse_manifest_digest": wheelhouse.Digest,
	})
	c.objectSet = writeCanonical("artifact object set", map[string]canonical.Value{
		"kind": "tensorhub.resolved_object_set/1", "roots": []canonical.Value{},
	})

	names := make([]string, 0, len(c.plans))
	for name := range c.plans {
		names = append(names, name)
	}
	sort.Strings(names)
	c.documents = nil
	for _, name := range names {
		doc := exactOf(c.plans[name])
		c.documents = append(c.documents, podBindingDocument{
			CanonicalBytes: doc.CanonicalBytes, Digest: doc.Digest,
			Kind: "entrypoint_binding_plan", Length: doc.Length,
		})
	}
	sort.Slice(c.documents, func(i, j int) bool {
		if c.documents[i].Kind != c.documents[j].Kind {
			return c.documents[i].Kind < c.documents[j].Kind
		}
		return c.documents[i].Digest < c.documents[j].Digest
	})
	// The execution digest fences the selected closure: descriptor, environment, receipt,
	// object set and every plan, in a fixed order.
	var fence strings.Builder
	for _, d := range []string{c.descriptor.Digest, c.environmentSpec.Digest,
		c.receipt.Digest, c.objectSet.Digest} {
		fence.WriteString(d + "\n")
	}
	for _, d := range c.documents {
		fence.WriteString(d.Digest + "\n")
	}
	c.executionDigest = exactOf([]byte(fence.String())).Digest

	rows := make([]canonical.Value, 0, len(c.documents))
	for _, d := range c.documents {
		rows = append(rows, map[string]canonical.Value{
			"kind": d.Kind, "ref": map[string]canonical.Value{"digest": d.Digest, "length": d.Length},
		})
	}
	c.bindingRelease = writeCanonical("binding release", map[string]canonical.Value{
		"descriptor":                    refOf(c.descriptor),
		"documents":                     rows,
		"endpoint_execution_digest":     c.executionDigest,
		"environment_spec":              refOf(c.environmentSpec),
		"format":                        "tensorhub.endpoint_binding_release/1",
		"installed_environment_receipt": refOf(c.receipt),
		"object_set_digest":             c.objectSet.Digest,
	})
}

// snapshot authors the attempt-bound half: ONE placement whose id IS the acquisition
// attempt, and the closed snapshot document around it.
func (c *podControl) snapshot(attemptID string) *hub.ExactControlDocument {
	subjects := make([]*pb.ArtifactSubject, 0, len(c.documents))
	for _, d := range c.documents {
		raw, err := canonical.Raw(d.Digest)
		must("plan digest", err)
		subjects = append(subjects, &pb.ArtifactSubject{
			Digest: raw, SubjectId: d.Digest, Kind: "plan", Length: uint64(d.Length),
		})
	}
	setBytes, err := canonical.Bytes(&pb.PlacementSet{Placements: []*pb.Placement{{
		PlacementId: attemptID,
		Spec: &pb.PlacementSpec{
			EndpointReleaseId:                 c.release,
			EnvironmentSpecDigest:             rawDigest(c.environmentSpec),
			InstalledEnvironmentReceiptDigest: rawDigest(c.receipt),
			DescriptorDigest:                  rawDigest(c.descriptor),
			BindingPlans:                      subjects,
		},
	}}})
	must("authoring the placement set", err)
	doc := podSnapshot{
		AcquisitionAttemptID: attemptID,
		ArtifactObjectSet:    c.objectSet, BindingDocuments: c.documents,
		BindingRelease: c.bindingRelease, Descriptor: c.descriptor, EndpointBundle: c.bundle,
		EndpointExecutionDigest: c.executionDigest, EnvironmentSpec: c.environmentSpec,
		EvaluatedConfig: c.config, Format: "tensorhub.rental_control_snapshot/1",
		InstalledEnvironmentReceipt: c.receipt, PlacementSet: exactOf(setBytes),
	}
	data, err := json.Marshal(doc)
	must("encoding the control snapshot", err)
	out := exactOf(data)
	return &out
}
