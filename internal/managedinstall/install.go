// Package managedinstall materializes a published endpoint over one exact, already
// qualified managed-local base. It never resolves packages, installs Torch/CUDA, runs
// a native build, or accepts a base path from Tensorhub.
package managedinstall

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/install"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/transfer"
	"github.com/cozy-creator/cozy-creator/internal/wheel"

	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

const managedBaseFormat = "cozy.local.ManagedBaseReceipt/1"

type Request struct {
	Endpoint string
	Release  string
	Major    int
	Profile  string
	Force    bool
	Grant    hub.LocalExecutionGrant
	Config   config.Config
}

type Result struct {
	Install    records.EndpointInstall
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

func Run(layout home.Layout, store *records.Store, request Request) (*Result, *exit.Error) {
	grant := request.Grant
	if request.Endpoint == "" || request.Release == "" || request.Major <= 0 ||
		grant.CandidateID == "" || grant.Profile != request.Profile ||
		grant.BaseRealization.Kind != "managed-local" || !digest(grant.BaseRealization.Digest) ||
		grant.LeaseID == "" || grant.LeaseExpiresAt == "" {
		return nil, exit.Named(exit.Structural, "managed_install_grant_invalid",
			"Tensorhub returned an incomplete or non-managed-local execution grant")
	}
	profiles, problem := endpointprofile.NormalizeSet([]string{request.Profile})
	if problem != nil {
		return nil, problem
	}
	profile := profiles[0]
	base, receipt, problem := openManagedBase(layout, grant, profile)
	if problem != nil {
		return nil, problem
	}

	_, priorInstall, problem := store.ActivePin(request.Endpoint, request.Major)
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
			request.Endpoint, request.Major, facts.CandidateID).
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
		{"endpoint-environment-spec.json", grant.EndpointEnvironmentSpec, "cozy.worker.v1.EndpointEnvironmentSpec/2", true},
		{"endpoint-bundle.json", grant.EndpointBundle, "tensorhub.endpoint_bundle/2", false},
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
	wheelFacts, problem := releaseWheelFacts(grant.EndpointBundle.CanonicalBytes,
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
			WithRemedy("install the qualified managed base realization before this endpoint"))
	}

	if err := os.MkdirAll(wheelDir, 0o700); err != nil {
		return fail(exit.Internalf("cannot create managed wheel root: %s", err))
	}
	wheelFiles := map[string]string{}
	seenRoles := map[string]bool{}
	nativeCustom := false
	for _, download := range grant.Downloads {
		if seenRoles[download.Role] || download.Role == "" || !digest(download.Ref.Digest) ||
			download.Ref.Length <= 0 || download.URL == "" {
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
		if problem := transfer.DownloadExact(context.Background(), download.Role, download.URL,
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
		"format":                    "cozy.runtime.EnvironmentProofRequest/1",
		"endpoint_environment_spec": docPaths["endpoint-environment-spec.json"],
		"endpoint_bundle":           docPaths["endpoint-bundle.json"],
		"resolved_wheel_set":        docPaths["resolved-wheel-set.json"],
		"wheelhouse_manifest":       docPaths["wheelhouse-manifest.json"],
		"wheel_files":               wheelFiles, "python": filepath.Join(base, receipt.Python),
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
	projectDir := filepath.Join(proof.Generation, "site-packages")
	if !inside(environmentRoot, proof.Generation) {
		return fail(exit.Named(exit.Structural, "managed_install_generation_escape",
			"Runtime published overlay generation outside its fresh environment root"))
	}
	runtimeBin := filepath.Join(base, "bin", "cozy-runtime")
	if problem := regularExecutable(runtimeBin, "managed base cozy-runtime"); problem != nil {
		return fail(problem)
	}
	descriptorDigest, problem := describe(runtimeBin, projectDir, toolEnv)
	if problem != nil {
		return fail(problem)
	}
	hostEvidencePath := filepath.Join(genDir, "host-evidence.json")
	hostDigest, problem := observeHost(runtimeBin, projectDir, hostEvidencePath, profile,
		toolEnv)
	if problem != nil {
		return fail(problem)
	}
	if nativeCustom {
		if grant.NativeWheelProof == nil {
			return fail(exit.Named(exit.Structural, "managed_native_proof_absent",
				"native custom wheel has no banked operator fixture/result proof request"))
		}
		nativeEvidence := filepath.Join(genDir, "native-wheel-qualification.json")
		hostDigest, problem = runNativeProof(base, grant, profile, environmentRoot,
			proof.Digest, docPaths["wheelhouse-manifest.json"], nativeEvidence, toolEnv)
		if problem != nil {
			return fail(problem)
		}
	} else if grant.NativeWheelProof != nil {
		return fail(exit.Named(exit.Structural, "managed_native_proof_unexpected",
			"pure overlay received a native-wheel proof request"))
	}

	gen := records.EndpointInstall{
		ID: id, Endpoint: request.Endpoint, Major: request.Major, Version: request.Release,
		SourceKind: "published-profile", SourceRef: request.Endpoint + "@" + request.Release + "#" + profile,
		SourceDigest: grant.EndpointBundle.Digest, Verified: true, Dir: genDir,
		Python: filepath.Join(base, receipt.Python), Runtime: runtimeBin, ProjectDir: projectDir,
		LockDigest: grant.ResolutionLock.Digest, Platform: profile, LinkMode: "overlay",
		Packages: len(wheelFiles), Closure: proof.Digest, Descriptor: descriptorDigest,
	}
	gen.BytesExcl, gen.BytesShared = install.Disk(genDir)
	facts := records.ManagedProfileInstall{
		ReleaseID: request.Release, Profile: profile, CandidateID: grant.CandidateID,
		BaseRealizationDigest:    grant.BaseRealization.Digest,
		BaseWorkerImageDigest:    grant.BaseWorkerImageDigest,
		WheelhouseManifestDigest: grant.WheelhouseManifest.Digest,
		EnvironmentSpecDigest:    grant.EndpointEnvironmentSpec.Digest,
		EndpointBundleDigest:     grant.EndpointBundle.Digest,
		ResolvedWheelSetDigest:   grant.ResolvedWheelSet.Digest,
		ResolutionLockDigest:     grant.ResolutionLock.Digest,
		InstalledReceiptDigest:   proof.Digest, InstalledReceiptLength: proof.Length,
		HostEvidenceDigest: hostDigest, LeaseID: grant.LeaseID, LeaseExpiresAt: grant.LeaseExpiresAt,
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
		!digest(bundle.ProjectWheel.Digest) || bundle.ProjectWheel.Filename == "" {
		return nil, exit.Named(exit.Structural, "managed_install_wheel_facts_invalid",
			"EndpointBundle/ResolvedWheelSet carry no complete project/custom WheelFacts")
	}
	out := map[string]wheel.Fact{"project_wheel": bundle.ProjectWheel}
	for _, fact := range resolved.Wheels {
		role := "custom_wheel:" + fact.Distribution + ":" + strings.TrimPrefix(fact.Digest, "sha256:")
		if _, duplicate := out[role]; duplicate || !digest(fact.Digest) || fact.Filename == "" {
			return nil, exit.Named(exit.Structural, "managed_install_wheel_facts_invalid",
				"ResolvedWheelSet has malformed or duplicate custom WheelFacts")
		}
		out[role] = fact
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

func runNativeProof(base string, grant hub.LocalExecutionGrant, profile, environmentRoot,
	receiptDigest, wheelhouse, output string, env []string,
) (string, *exit.Error) {
	spec := grant.NativeWheelProof
	if spec == nil || !digest(spec.ExpectedResultDigest) || spec.Fixture == "" ||
		(spec.DeviceIndex != nil && (*spec.DeviceIndex < 0 || *spec.DeviceIndex > 63)) {
		return "", exit.Named(exit.Structural, "managed_native_proof_invalid",
			"native-wheel proof fixture, result digest, or device index is invalid")
	}
	binary := filepath.Join(base, "bin", "cozy-native-wheel-proof")
	if problem := regularExecutable(binary, "managed base native-wheel proof"); problem != nil {
		return "", problem
	}
	request := map[string]any{
		"format":                  "cozy.runtime.NativeWheelProofRequest/1",
		"base_realization_digest": grant.BaseRealization.Digest,
		"base_realization_kind":   grant.BaseRealization.Kind,
		"base_worker_profile":     profile, "device_index": spec.DeviceIndex,
		"endpoint_environment_spec_digest": grant.EndpointEnvironmentSpec.Digest,
		"environment_root":                 environmentRoot, "expected_result_digest": spec.ExpectedResultDigest,
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
	return result.Digest, nil
}

func openManagedBase(layout home.Layout, grant hub.LocalExecutionGrant,
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
		if _, err := canonical.Read(doc.CanonicalBytes, &pb.EndpointEnvironmentSpec{}); err != nil {
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

func describe(runtime, project string, env []string) (string, *exit.Error) {
	cmd := exec.Command(runtime, "--json", "--dir", project, "describe", "--check")
	cmd.Env = env
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", exit.Named(exit.Structural, "managed_descriptor_refused", "%s", condense(stderr.String()))
	}
	var out struct {
		DescriptorDigest string `json:"descriptor_digest"`
	}
	if json.Unmarshal(stdout.Bytes(), &out) != nil || !digest(out.DescriptorDigest) {
		return "", exit.Internalf("managed cozy-runtime describe returned no descriptor digest")
	}
	return out.DescriptorDigest, nil
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

func SortedDownloads(rows []hub.DownloadGrant) []hub.DownloadGrant {
	out := append([]hub.DownloadGrant{}, rows...)
	sort.Slice(out, func(i, j int) bool { return out[i].Role < out[j].Role })
	return out
}

func (r Result) String() string {
	return fmt.Sprintf("%s %s %s", r.Install.Endpoint, r.Facts.Profile, r.Facts.InstalledReceiptDigest)
}
