// Package managedinstall materializes a published package over one exact, already
// qualified managed-local base. It never resolves packages, installs Torch/CUDA, runs
// a native build, or accepts a base path from Tensorhub.
package managedinstall

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/install"
	"github.com/cozy-creator/cozy-creator/internal/launch"
	"github.com/cozy-creator/cozy-creator/internal/packageprofile"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/transfer"
	"github.com/cozy-creator/cozy-creator/internal/wheel"

	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

const managedBaseFormat = "cozy.local.ManagedBaseReceipt/1"

type Request struct {
	Package     string
	Release     string
	Major       int
	Profile     string
	DeviceIndex *int
	Force       bool
	Grant       hub.LocalQualificationMaterials
	Config      config.Config
}

type Result struct {
	Install    records.PackageInstall
	Facts      records.ManagedProfileInstall
	Superseded string
	Idempotent bool
}

type baseReceipt struct {
	EnvironmentProof   string `json:"environment_proof"`
	Format             string `json:"format"`
	GenerationIdentity string `json:"generation_identity"`
	Profile            string `json:"profile"`
	Python             string `json:"python"`
	PythonABI          string `json:"python_abi"`
	WheelhouseManifest string `json:"wheelhouse_manifest_digest"`
}

type proofOutput struct {
	Digest     string `json:"digest"`
	Generation string `json:"generation"`
	Length     int64  `json:"length"`
	Reused     bool   `json:"reused"`
}

func Run(ctx context.Context, layout home.Layout, store *records.Store, request Request) (*Result, *exit.Error) {
	grant := request.Grant
	if request.Package == "" || request.Release == "" || request.Major <= 0 ||
		grant.CandidateID == "" || grant.Profile != request.Profile ||
		grant.BaseRealization.Kind != "managed-local" || !digest(grant.BaseRealization.Digest) ||
		grant.LeaseID == "" || grant.LeaseExpiresAt == "" {
		return nil, exit.Named(exit.Structural, "managed_install_grant_invalid",
			"Tensorhub returned incomplete managed-local qualification materials")
	}
	leaseExpiry, err := time.Parse(time.RFC3339, grant.LeaseExpiresAt)
	if err != nil || !leaseExpiry.After(time.Now()) {
		return nil, exit.Named(exit.Conflict, "managed_install_lease_expired",
			"managed-local qualification-materials lease %s is absent, malformed, or expired", grant.LeaseID).
			WithRemedy("request fresh local-qualification materials for the same non-refused candidate")
	}
	profiles, problem := packageprofile.NormalizeSet([]string{request.Profile})
	if problem != nil {
		return nil, problem
	}
	profile := profiles[0]
	base, receipt, problem := openManagedBase(layout, grant, profile)
	if problem != nil {
		return nil, problem
	}

	_, priorInstall, problem := store.ActivePin(request.Package, request.Major)
	if problem != nil {
		return nil, problem
	}
	if priorInstall != nil && priorInstall.SourceKind == "published-profile" && !request.Force {
		facts, problem := store.ManagedInstall(priorInstall.ID)
		if problem != nil {
			return nil, problem
		}
		if facts != nil && facts.CandidateID == grant.CandidateID &&
			facts.BaseRealizationDigest == grant.BaseRealization.Digest {
			return &Result{Install: *priorInstall, Facts: *facts, Idempotent: true}, nil
		}
		return nil, exit.New(exit.Conflict, "%s@v%d is already pinned to managed candidate %s",
			request.Package, request.Major, facts.CandidateID).
			WithRemedy("use --force to materialize and atomically replace it")
	}

	id := records.NewID("install")
	genDir := layout.GenerationDir(id)
	if err := os.Mkdir(genDir, 0o700); err != nil {
		return nil, exit.Internalf("cannot create managed generation %s: %s", id, err)
	}
	fail := func(problem *exit.Error) (*Result, *exit.Error) {
		_ = os.RemoveAll(genDir)
		return nil, problem
	}
	docDir, wheelDir := filepath.Join(genDir, "documents"), filepath.Join(genDir, "wheels")
	if err := os.MkdirAll(docDir, 0o700); err != nil {
		return fail(exit.Internalf("cannot create managed document root: %s", err))
	}
	documents := []struct {
		name   string
		doc    hub.ExactDocument
		format string
		typed  bool
	}{
		{"package-environment-spec.json", grant.PackageEnvironmentSpec, "cozy.worker.v1.PackageEnvironmentSpec/2", true},
		{"package-bundle.json", grant.PackageBundle, "tensorhub.package_bundle/2", false},
		{"resolved-wheel-set.json", grant.ResolvedWheelSet, "ResolvedWheelSet/3", false},
		{"wheelhouse-manifest.json", grant.WheelhouseManifest, "WheelhouseManifest/3", false},
		{"resolution-lock.json", grant.ResolutionLock, "tensorhub.resolution_lock/1", false},
	}
	docPaths := map[string]string{}
	for _, item := range documents {
		path := filepath.Join(docDir, item.name)
		if problem := writeExactDocument(path, item.doc, item.format, item.typed); problem != nil {
			return fail(problem)
		}
		docPaths[item.name] = path
	}
	descriptorPath := filepath.Join(docDir, "descriptor.json")
	if problem := writeSemanticDocument(descriptorPath, grant.PackageDescriptor); problem != nil {
		return fail(problem)
	}
	docPaths["descriptor.json"] = descriptorPath
	wheelFacts, problem := releaseWheelFacts(grant.PackageBundle.CanonicalBytes,
		grant.ResolvedWheelSet.CanonicalBytes)
	if problem != nil {
		return fail(problem)
	}
	baseManifest := filepath.Join(base, "WheelhouseManifest.json")
	if info, err := os.Lstat(baseManifest); err != nil || !info.Mode().IsRegular() {
		return fail(exit.Named(exit.Structural, "managed_base_wheelhouse_mismatch",
			"managed base WheelhouseManifest is absent, symlinked, or non-regular"))
	}
	baseBytes, err := os.ReadFile(baseManifest)
	if err != nil || !bytes.Equal(baseBytes, grant.WheelhouseManifest.CanonicalBytes) {
		return fail(exit.Named(exit.Structural, "managed_base_wheelhouse_mismatch",
			"managed base %s does not carry the exact granted WheelhouseManifest", base).
			WithRemedy("install the qualified managed base realization before this package"))
	}

	if err := os.MkdirAll(wheelDir, 0o700); err != nil {
		return fail(exit.Internalf("cannot create managed wheel root: %s", err))
	}
	wheelFiles := map[string]string{}
	seenRoles := map[string]bool{}
	nativeCustom := false
	for _, download := range grant.Downloads {
		downloadExpiry, expiryErr := time.Parse(time.RFC3339, download.ExpiresAt)
		if seenRoles[download.Role] || download.Role == "" || !digest(download.Ref.Digest) ||
			download.Ref.Length <= 0 || download.URL == "" || expiryErr != nil ||
			!downloadExpiry.After(time.Now()) {
			return fail(exit.Named(exit.Structural, "managed_install_download_invalid",
				"local execution has a malformed or duplicate wheel download role %q", download.Role))
		}
		fact, ok := wheelFacts[download.Role]
		if !ok || fact.Digest != download.Ref.Digest || fact.Length != download.Ref.Length {
			return fail(exit.Named(exit.Structural, "managed_install_download_invalid",
				"local execution returned an unknown or changed wheel role %q", download.Role))
		}
		seenRoles[download.Role] = true
		target := filepath.Join(wheelDir, fact.Filename)
		if problem := transfer.DownloadExact(ctx, download.Role, download.URL,
			target, download.Ref.Digest, download.Ref.Length); problem != nil {
			return fail(problem)
		}
		class := wheel.CustomWheel
		if download.Role == "project_wheel" {
			class = wheel.ProjectWheel
		}
		inspected, problem := wheel.Inspect(target, class)
		if problem != nil || !sameWheelFact(fact, inspected) {
			if problem != nil {
				return fail(problem)
			}
			return fail(exit.Named(exit.Structural, "managed_install_wheel_fact_mismatch",
				"downloaded role %s disagrees with its canonical WheelFact", download.Role))
		}
		nativeCustom = nativeCustom || class == wheel.CustomWheel && inspected.Native
		wheelFiles[download.Ref.Digest] = target
	}
	if !seenRoles["project_wheel"] || len(seenRoles) != len(wheelFacts) {
		return fail(exit.Named(exit.Structural, "managed_install_project_wheel_absent",
			"local execution did not grant every-and-only project/custom wheel"))
	}

	environmentRoot := filepath.Join(genDir, "environment")
	requestPath, receiptPath := filepath.Join(genDir, "environment-proof-request.json"), filepath.Join(genDir, "installed-receipt.json")
	proofRequest := map[string]any{
		"format":                   "cozy.runtime.EnvironmentProofRequest/1",
		"package_environment_spec": docPaths["package-environment-spec.json"],
		"package_bundle":           docPaths["package-bundle.json"],
		"resolved_wheel_set":       docPaths["resolved-wheel-set.json"],
		"wheelhouse_manifest":      docPaths["wheelhouse-manifest.json"],
		"wheel_files":              wheelFiles, "python": filepath.Join(base, receipt.Python),
		"environment_root": environmentRoot,
	}
	if problem := writeJCS(requestPath, proofRequest); problem != nil {
		return fail(problem)
	}
	toolEnv := request.Config.Tool("COZY_HOME=" + request.Config.Home)
	proof, problem := runProof(filepath.Join(base, receipt.EnvironmentProof), requestPath,
		receiptPath, toolEnv)
	if problem != nil {
		return fail(problem)
	}
	if problem := verifyFile(receiptPath, proof.Digest, proof.Length); problem != nil {
		return fail(problem)
	}
	if proof.Reused {
		return fail(exit.Named(exit.Structural, "managed_environment_reused_fresh_root",
			"Runtime reported a reused overlay inside a fresh managed generation root"))
	}
	overlayContentDigest, problem := verifyOverlayReceipt(receiptPath, grant, wheelFacts)
	if problem != nil {
		return fail(problem)
	}
	projectDir := filepath.Join(proof.Generation, "site-packages")
	expectedGeneration := filepath.Join(environmentRoot, "contents",
		strings.TrimPrefix(overlayContentDigest, "sha256:"))
	if !inside(environmentRoot, proof.Generation) || filepath.Clean(proof.Generation) != expectedGeneration {
		return fail(exit.Named(exit.Structural, "managed_install_generation_escape",
			"Runtime published overlay generation outside its exact content-addressed path"))
	}
	runtimeBin := filepath.Join(base, "bin", "cozy-runtime")
	if problem := regularExecutable(runtimeBin, "managed base cozy-runtime"); problem != nil {
		return fail(problem)
	}
	hostEvidencePath := filepath.Join(genDir, "host-evidence.json")
	hostDigest, problem := observeHost(runtimeBin, projectDir, hostEvidencePath, profile,
		toolEnv)
	if problem != nil {
		return fail(problem)
	}
	nativeDigest := ""
	if nativeCustom {
		if grant.NativeWheelProof == nil {
			return fail(exit.Named(exit.Structural, "managed_native_proof_absent",
				"native custom wheel has no banked operator fixture/result proof request"))
		}
		nativeEvidence := filepath.Join(genDir, "native-wheel-qualification.json")
		nativeDigest, problem = runNativeProof(base, grant, profile, request.DeviceIndex, environmentRoot,
			proof.Generation, proof.Digest, overlayContentDigest, receiptPath,
			docPaths["wheelhouse-manifest.json"], nativeEvidence, toolEnv)
		if problem != nil {
			return fail(problem)
		}
	} else if grant.NativeWheelProof != nil {
		return fail(exit.Named(exit.Structural, "managed_native_proof_unexpected",
			"pure overlay received a native-wheel proof request"))
	}
	derivedDescriptor, problem := describe(runtimeBin, projectDir, toolEnv)
	if problem != nil {
		return fail(problem)
	}
	descriptorDigest := derivedDescriptor.Digest
	if descriptorDigest != grant.PackageDescriptor.Digest ||
		!bytes.Equal(derivedDescriptor.Raw, grant.PackageDescriptor.CanonicalBytes) {
		return fail(exit.Named(exit.Structural, "managed_descriptor_mismatch",
			"installed package derives descriptor %s; the published release grants %s with different bytes",
			descriptorDigest, grant.PackageDescriptor.Digest).
			WithRemedy("refuse this candidate; its qualified documents and installed project wheel do not describe the same package"))
	}

	gen := records.PackageInstall{
		ID: id, Package: request.Package, Major: request.Major, Version: request.Release,
		SourceKind: "published-profile", SourceRef: request.Package + "@" + request.Release + "#" + profile,
		SourceDigest: grant.PackageBundle.Digest, Verified: true, Dir: genDir,
		Python: filepath.Join(base, receipt.Python), Runtime: runtimeBin, ProjectDir: projectDir,
		LockDigest: grant.ResolutionLock.Digest, Platform: profile, LinkMode: "overlay",
		Packages: len(wheelFiles), Closure: proof.Digest, PackageDescriptor: descriptorDigest,
	}
	gen.BytesExcl, gen.BytesShared = install.Disk(genDir)
	facts := records.ManagedProfileInstall{
		PackageReleaseID: request.Release, Profile: profile, CandidateID: grant.CandidateID,
		BaseRealizationDigest:    grant.BaseRealization.Digest,
		WheelhouseManifestDigest: grant.WheelhouseManifest.Digest,
		EnvironmentSpecDigest:    grant.PackageEnvironmentSpec.Digest,
		PackageBundleDigest:      grant.PackageBundle.Digest,
		ResolvedWheelSetDigest:   grant.ResolvedWheelSet.Digest,
		ResolutionLockDigest:     grant.ResolutionLock.Digest,
		InstalledReceiptDigest:   proof.Digest, InstalledReceiptLength: proof.Length,
		HostEvidenceDigest: hostDigest, NativeEvidenceDigest: nativeDigest,
		LeaseID: grant.LeaseID, LeaseExpiresAt: grant.LeaseExpiresAt,
	}
	superseded, problem := store.ActivateManaged(gen, facts)
	if problem != nil {
		return fail(problem)
	}
	return &Result{Install: gen, Facts: facts, Superseded: superseded}, nil
}

func releaseWheelFacts(bundleBytes, resolvedBytes []byte) (map[string]wheel.Fact, *exit.Error) {
	var bundle struct {
		ProjectWheel wheel.Fact `json:"project_wheel"`
	}
	var resolved struct {
		Wheels []wheel.Fact `json:"wheels"`
	}
	if json.Unmarshal(bundleBytes, &bundle) != nil || json.Unmarshal(resolvedBytes, &resolved) != nil ||
		!digest(bundle.ProjectWheel.Digest) || bundle.ProjectWheel.Filename == "" ||
		bundle.ProjectWheel.Distribution == "" || bundle.ProjectWheel.Version == "" ||
		bundle.ProjectWheel.Length <= 0 {
		return nil, exit.Named(exit.Structural, "managed_install_wheel_facts_invalid",
			"PackageBundle/ResolvedWheelSet carry no complete project/custom WheelFacts")
	}
	out := map[string]wheel.Fact{"project_wheel": bundle.ProjectWheel}
	seenDigests := map[string]bool{bundle.ProjectWheel.Digest: true}
	seenDistributions := map[string]bool{bundle.ProjectWheel.Distribution: true}
	for _, fact := range resolved.Wheels {
		role := "custom_wheel:" + fact.Distribution + ":" + strings.TrimPrefix(fact.Digest, "sha256:")
		if _, duplicate := out[role]; duplicate || !digest(fact.Digest) || fact.Filename == "" ||
			fact.Distribution == "" || fact.Version == "" || fact.Length <= 0 ||
			seenDigests[fact.Digest] || seenDistributions[fact.Distribution] {
			return nil, exit.Named(exit.Structural, "managed_install_wheel_facts_invalid",
				"ResolvedWheelSet has malformed or duplicate custom WheelFacts")
		}
		out[role] = fact
		seenDigests[fact.Digest] = true
		seenDistributions[fact.Distribution] = true
	}
	return out, nil
}

func sameWheelFact(a, b wheel.Fact) bool {
	if a.Digest != b.Digest || a.Distribution != b.Distribution || a.Filename != b.Filename ||
		a.Length != b.Length || a.Version != b.Version || len(a.ImportRoots) != len(b.ImportRoots) ||
		len(a.Tags) != len(b.Tags) {
		return false
	}
	for i := range a.ImportRoots {
		if a.ImportRoots[i] != b.ImportRoots[i] {
			return false
		}
	}
	for i := range a.Tags {
		if a.Tags[i] != b.Tags[i] {
			return false
		}
	}
	return true
}

type overlayWheelReceipt struct {
	Digest       string `json:"digest"`
	Distribution string `json:"distribution"`
	Owner        string `json:"owner"`
	Version      string `json:"version"`
}

type overlayReceipt struct {
	BaseFamilyDigest   string                `json:"base_family_digest"`
	EnvironmentSpec    string                `json:"environment_spec_digest"`
	Format             string                `json:"format"`
	OverlayWheels      []overlayWheelReceipt `json:"overlay_wheels"`
	ProjectWheelDigest string                `json:"project_wheel_digest"`
}

type nativeDistribution struct {
	Distribution string `json:"distribution"`
	Version      string `json:"version"`
}

type nativeSeat struct {
	CUDAAvailable  bool                 `json:"cuda_available"`
	Distributions  []nativeDistribution `json:"distributions"`
	Implementation string               `json:"implementation"`
	LibcName       string               `json:"libc_name"`
	LibcVersion    string               `json:"libc_version"`
	Machine        string               `json:"machine"`
	PythonABI      string               `json:"python_abi"`
	System         string               `json:"system"`
	TorchCUDA      string               `json:"torch_cuda"`
	TorchVersion   string               `json:"torch_version"`
}

type nativeGPU struct {
	DeviceIndex   int    `json:"device_index"`
	DriverVersion string `json:"driver_version"`
	Name          string `json:"name"`
	SM            int    `json:"sm"`
}

type nativeInspectionMember struct {
	CubinSMs              []int    `json:"cubin_sms"`
	CUDASections          []string `json:"cuda_sections"`
	DefinedSymbolsDigest  string   `json:"defined_symbols_digest"`
	Member                string   `json:"member"`
	Needed                []string `json:"needed"`
	PTXCompute            []int    `json:"ptx_compute"`
	RequiredSymbolsDigest string   `json:"required_symbols_digest"`
	Runpaths              []string `json:"runpaths"`
	Soname                string   `json:"soname"`
}

type nativeInspection struct {
	Format  string                   `json:"format"`
	Members []nativeInspectionMember `json:"members"`
}

type nativeHostQualification struct {
	Format           string `json:"format"`
	GPUSM            *int   `json:"gpu_sm"`
	HostLibraryCount int    `json:"host_library_count"`
	InspectionDigest string `json:"inspection_digest"`
	Qualification    string `json:"qualification"`
}

type nativeOperatorObservation struct {
	ExpectedResultDigest string `json:"expected_result_digest"`
	Fixture              string `json:"fixture"`
	Format               string `json:"format"`
	ResultDigest         string `json:"result_digest"`
}

type nativeQualification struct {
	BaseWorkerProfile            string `json:"base_worker_profile"`
	PackageEnvironmentSpecDigest string `json:"package_environment_spec_digest"`
	Format                       string `json:"format"`
	HostCapability               struct {
		GPU  *nativeGPU `json:"gpu"`
		Seat nativeSeat `json:"seat"`
	} `json:"host_capability"`
	HostQualification                 nativeHostQualification   `json:"host_qualification"`
	InstalledEnvironmentReceiptDigest string                    `json:"installed_environment_receipt_digest"`
	OperatorObservation               nativeOperatorObservation `json:"operator_observation"`
	OverlayContentDigest              string                    `json:"overlay_content_digest"`
	StaticInspection                  nativeInspection          `json:"static_inspection"`
	WheelhouseManifestDigest          string                    `json:"wheelhouse_manifest_digest"`
}

func verifyOverlayReceipt(path string, grant hub.LocalQualificationMaterials,
	wheelFacts map[string]wheel.Fact,
) (string, *exit.Error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", exit.Internalf("cannot reread Runtime overlay receipt: %s", err)
	}
	canonicalBytes, err := canonical.NormalizeJCS(raw)
	if err != nil || !bytes.Equal(canonicalBytes, raw) {
		return "", exit.Named(exit.Structural, "managed_environment_receipt_invalid",
			"Runtime overlay receipt is not exact canonical JCS: %v", err)
	}
	var receipt overlayReceipt
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&receipt); err != nil || receipt.Format != "cozy.runtime.PackageOverlayReceipt/1" ||
		receipt.BaseFamilyDigest != grant.WheelhouseManifest.Digest ||
		receipt.EnvironmentSpec != grant.PackageEnvironmentSpec.Digest {
		return "", exit.Named(exit.Structural, "managed_environment_receipt_invalid",
			"Runtime overlay receipt does not bind the granted base/environment: %v", err)
	}
	project := wheelFacts["project_wheel"]
	if receipt.ProjectWheelDigest != project.Digest {
		return "", exit.Named(exit.Structural, "managed_environment_receipt_invalid",
			"Runtime overlay receipt does not bind the granted project wheel")
	}
	expected := make([]overlayWheelReceipt, 0, len(wheelFacts))
	for role, fact := range wheelFacts {
		owner := "custom"
		if role == "project_wheel" {
			owner = "project"
		}
		expected = append(expected, overlayWheelReceipt{Digest: fact.Digest,
			Distribution: fact.Distribution, Owner: owner, Version: fact.Version})
	}
	sort.Slice(expected, func(i, j int) bool { return expected[i].Distribution < expected[j].Distribution })
	if len(receipt.OverlayWheels) != len(expected) {
		return "", exit.Named(exit.Structural, "managed_environment_receipt_invalid",
			"Runtime overlay receipt names %d wheels; the granted set has %d",
			len(receipt.OverlayWheels), len(expected))
	}
	for i := range expected {
		if receipt.OverlayWheels[i] != expected[i] {
			return "", exit.Named(exit.Structural, "managed_environment_receipt_invalid",
				"Runtime overlay receipt wheel row %d disagrees with the granted inventory", i)
		}
	}
	wheelDigests := make([]string, 0, len(receipt.OverlayWheels))
	for _, row := range receipt.OverlayWheels {
		wheelDigests = append(wheelDigests, row.Digest)
	}
	sort.Strings(wheelDigests)
	contentBytes, err := json.Marshal(map[string]any{"base_family_digest": receipt.BaseFamilyDigest,
		"format": "cozy.runtime.PackageOverlayContent/1", "wheel_digests": wheelDigests})
	if err != nil {
		return "", exit.Internalf("cannot encode overlay content identity: %s", err)
	}
	contentBytes, err = canonical.NormalizeJCS(contentBytes)
	if err != nil {
		return "", exit.Internalf("cannot canonicalize overlay content identity: %s", err)
	}
	return hash(contentBytes), nil
}

func runNativeProof(base string, grant hub.LocalQualificationMaterials, profile string, deviceIndex *int,
	environmentRoot, generation, receiptDigest, overlayContentDigest, receiptPath, wheelhouse,
	output string, env []string,
) (string, *exit.Error) {
	spec := grant.NativeWheelProof
	if spec == nil || !digest(spec.ExpectedResultDigest) || spec.Fixture == "" ||
		(deviceIndex != nil && (*deviceIndex < 0 || *deviceIndex > 63)) {
		return "", exit.Named(exit.Structural, "managed_native_proof_invalid",
			"native-wheel proof fixture, result digest, or device index is invalid")
	}
	binary := filepath.Join(base, "bin", "cozy-native-wheel-proof")
	if problem := regularExecutable(binary, "managed base native-wheel proof"); problem != nil {
		return "", problem
	}
	request := map[string]any{
		"format": "cozy.runtime.NativeWheelProofRequest/1", "device_index": deviceIndex,
		"package_environment_spec_digest": grant.PackageEnvironmentSpec.Digest,
		"environment_root":                environmentRoot, "expected_result_digest": spec.ExpectedResultDigest,
		"fixture": spec.Fixture, "installed_environment_receipt_digest": receiptDigest,
		"python": filepath.Join(base, "bin", "python"), "wheelhouse_manifest": wheelhouse,
	}
	requestPath := strings.TrimSuffix(output, filepath.Ext(output)) + "-request.json"
	if problem := writeJCS(requestPath, request); problem != nil {
		return "", problem
	}
	cmd := exec.Command(binary, requestPath, output)
	cmd.Env = env
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", exit.Named(exit.Structural, "managed_native_proof_refused", "%s", condense(stderr.String()))
	}
	var result struct {
		Digest string `json:"digest"`
		Length int64  `json:"length"`
	}
	if json.Unmarshal(stdout.Bytes(), &result) != nil || verifyFile(output, result.Digest, result.Length) != nil {
		return "", exit.Internalf("cozy-native-wheel-proof returned incomplete evidence")
	}
	if problem := validateNativeEvidence(output, grant, profile, deviceIndex, generation,
		receiptDigest, overlayContentDigest, receiptPath, wheelhouse); problem != nil {
		return "", problem
	}
	return result.Digest, nil
}

func validateNativeEvidence(path string, grant hub.LocalQualificationMaterials, profile string,
	deviceIndex *int, generation, receiptDigest, overlayContentDigest, receiptPath,
	wheelhouse string,
) *exit.Error {
	refuse := func(format string, args ...any) *exit.Error {
		return exit.Named(exit.Structural, "managed_native_evidence_invalid", format, args...)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return refuse("cannot read native qualification evidence: %v", err)
	}
	canonicalBytes, err := canonical.NormalizeJCS(raw)
	if err != nil || !bytes.Equal(canonicalBytes, raw) {
		return refuse("native qualification evidence is not exact canonical JCS: %v", err)
	}
	var evidence nativeQualification
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&evidence); err != nil {
		return refuse("native qualification evidence is not closed: %v", err)
	}
	if evidence.Format != "cozy.runtime.NativeWheelQualification/1" ||
		evidence.BaseWorkerProfile != profile ||
		evidence.PackageEnvironmentSpecDigest != grant.PackageEnvironmentSpec.Digest ||
		evidence.WheelhouseManifestDigest != grant.WheelhouseManifest.Digest ||
		evidence.InstalledEnvironmentReceiptDigest != receiptDigest ||
		evidence.OverlayContentDigest != overlayContentDigest ||
		filepath.Base(generation) != strings.TrimPrefix(overlayContentDigest, "sha256:") {
		return refuse("native qualification evidence does not join the exact profile/environment/receipt/content")
	}
	if _, err := os.Stat(receiptPath); err != nil {
		return refuse("installed Runtime receipt disappeared before native qualification joined it")
	}

	inspection := evidence.StaticInspection
	if inspection.Format != "cozy.runtime.NativeWheelInspection/1" || len(inspection.Members) == 0 {
		return refuse("native qualification has no closed static inspection")
	}
	priorMember := ""
	for i, member := range inspection.Members {
		if member.Member == "" || member.Member <= priorMember ||
			!digest(member.DefinedSymbolsDigest) || !digest(member.RequiredSymbolsDigest) ||
			member.CubinSMs == nil || member.CUDASections == nil || member.Needed == nil ||
			member.PTXCompute == nil || member.Runpaths == nil {
			return refuse("native static inspection member %d is malformed or unsorted", i)
		}
		priorMember = member.Member
	}
	inspectionBytes, err := json.Marshal(inspection)
	if err != nil {
		return exit.Internalf("cannot encode native inspection evidence: %s", err)
	}
	inspectionBytes, err = canonical.NormalizeJCS(inspectionBytes)
	if err != nil {
		return exit.Internalf("cannot canonicalize native inspection evidence: %s", err)
	}
	host := evidence.HostQualification
	if host.Format != "cozy.runtime.NativeHostQualification/1" || host.HostLibraryCount < 0 ||
		host.InspectionDigest != hash(inspectionBytes) || host.Qualification != "static-host-compatible" {
		return refuse("native host qualification does not join the measured static inspection")
	}
	operator := evidence.OperatorObservation
	if grant.NativeWheelProof == nil || operator.Format != "cozy.runtime.NativeOperatorObservation/1" ||
		operator.Fixture != grant.NativeWheelProof.Fixture ||
		operator.ExpectedResultDigest != grant.NativeWheelProof.ExpectedResultDigest ||
		operator.ResultDigest != operator.ExpectedResultDigest || !digest(operator.ResultDigest) {
		return refuse("native operator observation does not reproduce the banked fixture/result")
	}

	baseBytes, err := os.ReadFile(wheelhouse)
	if err != nil {
		return refuse("cannot reread exact WheelhouseManifest: %v", err)
	}
	var base struct {
		BaseDistributions    []nativeDistribution `json:"base_distributions"`
		CompatibilityProfile struct {
			TorchRelease     string `json:"torch_release"`
			AcceleratorBuild string `json:"accelerator_build"`
			PythonABI        string `json:"python_abi"`
			OSCPU            string `json:"os_cpu"`
		} `json:"compatibility_profile"`
	}
	if json.Unmarshal(baseBytes, &base) != nil || len(base.BaseDistributions) == 0 {
		return refuse("WheelhouseManifest carries no base inventory/profile")
	}
	baseProfile := "torch" + base.CompatibilityProfile.TorchRelease + "-" +
		base.CompatibilityProfile.AcceleratorBuild + "-" + base.CompatibilityProfile.PythonABI + "-" +
		base.CompatibilityProfile.OSCPU
	seat := evidence.HostCapability.Seat
	if baseProfile != profile || seat.Implementation != "cpython" || seat.System != "linux" ||
		(seat.Machine != "x86_64" && seat.Machine != "amd64") ||
		seat.PythonABI != pythonABI(profile) || seat.LibcName == "" || seat.LibcVersion == "" ||
		seat.TorchCUDA != acceleratorVersion(profile) ||
		(seat.TorchVersion != base.CompatibilityProfile.TorchRelease &&
			!strings.HasPrefix(seat.TorchVersion, base.CompatibilityProfile.TorchRelease+"+")) ||
		!sameDistributions(seat.Distributions, base.BaseDistributions) {
		return refuse("native proof seat does not match the exact managed base/profile inventory")
	}
	gpu := evidence.HostCapability.GPU
	if deviceIndex == nil {
		if gpu != nil || host.GPUSM != nil {
			return refuse("CPU native proof unexpectedly carries GPU evidence")
		}
	} else if gpu == nil || host.GPUSM == nil || gpu.DeviceIndex != *deviceIndex ||
		gpu.SM <= 0 || *host.GPUSM != gpu.SM || gpu.Name == "" || gpu.DriverVersion == "" ||
		!seat.CUDAAvailable {
		return refuse("native proof GPU evidence does not match the selected device/profile")
	}
	return nil
}

func sameDistributions(a, b []nativeDistribution) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] || a[i].Distribution == "" || a[i].Version == "" ||
			(i > 0 && (a[i-1].Distribution > a[i].Distribution ||
				a[i-1].Distribution == a[i].Distribution)) {
			return false
		}
	}
	return true
}

func openManagedBase(layout home.Layout, grant hub.LocalQualificationMaterials,
	profile string,
) (string, baseReceipt, *exit.Error) {
	var receipt baseReceipt
	base := layout.ManagedBase(grant.BaseRealization.Digest)
	path := filepath.Join(base, "ManagedBaseReceipt.json")
	if info, err := os.Lstat(base); err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", receipt, exit.Named(exit.Structural, "managed_base_absent",
			"no regular exact managed base directory at %s", base).
			WithRemedy("install the qualified %s managed base realization first", profile)
	}
	if info, err := os.Lstat(path); err != nil || !info.Mode().IsRegular() {
		return "", receipt, exit.Named(exit.Structural, "managed_base_receipt_invalid",
			"ManagedBaseReceipt is absent, symlinked, or non-regular at %s", path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", receipt, exit.Named(exit.Structural, "managed_base_absent",
			"no exact managed base at %s", base).
			WithRemedy("install the qualified %s managed base realization first", profile)
	}
	if hash(raw) != grant.BaseRealization.Digest {
		return "", receipt, exit.Named(exit.Structural, "managed_base_receipt_mismatch",
			"ManagedBaseReceipt stored bytes do not hash to %s", grant.BaseRealization.Digest)
	}
	canonicalBytes, err := canonical.NormalizeJCS(raw)
	if err != nil || !bytes.Equal(canonicalBytes, raw) {
		return "", receipt, exit.Named(exit.Structural, "managed_base_receipt_invalid",
			"ManagedBaseReceipt is not canonical: %v", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&receipt); err != nil || receipt.Format != managedBaseFormat ||
		receipt.Profile != profile || receipt.WheelhouseManifest != grant.WheelhouseManifest.Digest ||
		receipt.PythonABI != pythonABI(profile) || receipt.Python != "bin/python" ||
		receipt.EnvironmentProof != "bin/cozy-environment-proof" || !digest(receipt.GenerationIdentity) {
		return "", receipt, exit.Named(exit.Structural, "managed_base_receipt_invalid",
			"ManagedBaseReceipt fields disagree with the granted profile/base: %v", err)
	}
	for name, rel := range map[string]string{"python": receipt.Python,
		"environment proof": receipt.EnvironmentProof} {
		if problem := regularExecutable(filepath.Join(base, rel), "managed base "+name); problem != nil {
			return "", receipt, problem
		}
	}
	return base, receipt, nil
}

func writeExactDocument(path string, doc hub.ExactDocument, format string, typed bool) *exit.Error {
	if problem := exactBytes(doc.Digest, doc.Length, doc.CanonicalBytes); problem != nil {
		return problem
	}
	if typed {
		if _, err := canonical.Read(doc.CanonicalBytes, &pb.PackageEnvironmentSpec{}); err != nil {
			return exit.Named(exit.Structural, "managed_install_document_invalid", "%s: %v", format, err)
		}
	} else {
		value, err := canonical.ReadObject(doc.CanonicalBytes)
		if err != nil || value["format"] != format {
			return exit.Named(exit.Structural, "managed_install_document_invalid", "%s: %v", format, err)
		}
	}
	if err := os.WriteFile(path, doc.CanonicalBytes, 0o600); err != nil {
		return exit.Internalf("cannot stage %s: %s", format, err)
	}
	return nil
}

func writeSemanticDocument(path string, doc hub.ExactDocument) *exit.Error {
	if problem := exactBytes(doc.Digest, doc.Length, doc.CanonicalBytes); problem != nil {
		return problem
	}
	normalized, err := canonical.NormalizeJCS(doc.CanonicalBytes)
	if err != nil || !bytes.Equal(normalized, doc.CanonicalBytes) {
		return exit.Named(exit.Structural, "managed_install_document_invalid",
			"semantic package document is not exact canonical JCS: %v", err)
	}
	if err := os.WriteFile(path, doc.CanonicalBytes, 0o600); err != nil {
		return exit.Internalf("cannot stage package semantic document: %s", err)
	}
	return nil
}

func runProof(binary, request, receipt string, env []string) (proofOutput, *exit.Error) {
	var out proofOutput
	cmd := exec.Command(binary, request, receipt)
	cmd.Env = env
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return out, exit.Named(exit.Structural, "managed_environment_proof_missing", "%s: %v", binary, err)
	}
	if cmd.ProcessState.ExitCode() != 0 {
		return out, exit.Named(exit.Structural, "managed_environment_proof_refused",
			"cozy-environment-proof exited %d: %s", cmd.ProcessState.ExitCode(), condense(stderr.String()))
	}
	if json.Unmarshal(stdout.Bytes(), &out) != nil || !digest(out.Digest) || out.Length <= 0 ||
		out.Generation == "" {
		return out, exit.Internalf("cozy-environment-proof returned an incomplete result: %s", stdout.String())
	}
	return out, nil
}

func describe(runtime, project string, env []string) (*launch.PackageDescriptor, *exit.Error) {
	cmd := exec.Command(runtime, "--json", "--dir", project, "describe")
	cmd.Env = env
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, exit.Named(exit.Structural, "managed_descriptor_refused", "%s", condense(stderr.String()))
	}
	descriptor, problem := launch.DecodeDescriptor(stdout.Bytes())
	if problem != nil {
		return nil, exit.Named(exit.Structural, "managed_descriptor_refused",
			"managed cozy-runtime returned an invalid descriptor: %s", problem.Message)
	}
	return descriptor, nil
}

func observeHost(runtime, project, output, profile string, env []string) (string, *exit.Error) {
	cmd := exec.Command(runtime, "--json", "--dir", project, "doctor")
	cmd.Env = env
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", exit.Named(exit.Structural, "managed_host_observation_refused", "%s", condense(stderr.String()))
	}
	canonicalBytes, err := canonical.NormalizeJCS(stdout.Bytes())
	if err != nil {
		return "", exit.Named(exit.Structural, "managed_host_observation_invalid", "%v", err)
	}
	var observed struct {
		Device struct {
			Name        string `json:"name"`
			State       string `json:"state"`
			CUDAVersion string `json:"cuda_version"`
		} `json:"device"`
	}
	if json.Unmarshal(canonicalBytes, &observed) != nil || observed.Device.Name == "" ||
		observed.Device.State != "present" || !strings.HasPrefix(observed.Device.CUDAVersion, acceleratorVersion(profile)) {
		return "", exit.Named(exit.Structural, "managed_host_profile_mismatch",
			"current host does not measure the qualified GPU/CUDA class for %s", profile)
	}
	if err := os.WriteFile(output, canonicalBytes, 0o600); err != nil {
		return "", exit.Internalf("cannot store managed host evidence: %s", err)
	}
	return hash(canonicalBytes), nil
}

func writeJCS(path string, value any) *exit.Error {
	body, err := json.Marshal(value)
	if err != nil {
		return exit.Internalf("cannot encode Runtime request: %s", err)
	}
	canonicalBytes, err := canonical.NormalizeJCS(body)
	if err != nil {
		return exit.Internalf("Runtime request is not canonical: %s", err)
	}
	if err := os.WriteFile(path, canonicalBytes, 0o600); err != nil {
		return exit.Internalf("cannot store Runtime request: %s", err)
	}
	return nil
}

func exactBytes(digestValue string, length int64, body []byte) *exit.Error {
	if !digest(digestValue) || length <= 0 || int64(len(body)) != length || hash(body) != digestValue {
		return exit.Named(exit.Structural, "managed_install_document_identity_mismatch",
			"exact document does not match its digest/length ref")
	}
	return nil
}

func verifyFile(path, digestValue string, length int64) *exit.Error {
	body, err := os.ReadFile(path)
	if err != nil {
		return exit.Internalf("cannot read Runtime receipt: %s", err)
	}
	return exactBytes(digestValue, length, body)
}

func regularExecutable(path, name string) *exit.Error {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&0o111 == 0 {
		return exit.Named(exit.Structural, "managed_base_file_invalid",
			"%s at %s is absent, symlinked, non-regular, or non-executable", name, path)
	}
	return nil
}

func inside(root, child string) bool {
	rel, err := filepath.Rel(root, child)
	return err == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

func hash(body []byte) string {
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func digest(value string) bool {
	if len(value) != 71 || !strings.HasPrefix(value, "sha256:") || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil
}

func pythonABI(profile string) string {
	for _, part := range strings.Split(profile, "-") {
		if strings.HasPrefix(part, "cp") {
			return part
		}
	}
	return ""
}

func acceleratorVersion(profile string) string {
	parts := strings.Split(profile, "-")
	if len(parts) < 2 || !strings.HasPrefix(parts[1], "cu") || len(parts[1]) < 4 {
		return ""
	}
	digits := strings.TrimPrefix(parts[1], "cu")
	return digits[:len(digits)-1] + "." + digits[len(digits)-1:]
}

func condense(value string) string {
	value = strings.Join(strings.Fields(value), " ")
	if len(value) > 400 {
		return value[:400] + "…"
	}
	return value
}
