// Package managedbase owns Creator's package-independent local Runtime base.
// Tensorhub selects an exact profile and WheelhouseManifest; this package turns
// that selection into one immutable, locally runnable OCI generation and an
// atomic active receipt under Cozy home. Package names and releases never enter
// the base identity.
package managedbase

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
)

const (
	RecordFormat     = "managed_base"
	SupportedProfile = "torch2.13.0-cu130-cp312-linux-x86"
	imageRepository  = "docker.io/tensorhub/worker"
)

var commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)

type ExactDocument struct {
	Bytes  []byte
	Digest string
	Length int64
}

type Input struct {
	Profile  string
	Manifest ExactDocument
}

type Ref struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

type Wheel struct {
	Digest       string   `json:"digest"`
	Distribution string   `json:"distribution"`
	Filename     string   `json:"filename"`
	ImportRoots  []string `json:"import_roots"`
	Length       int64    `json:"length"`
	Tags         []string `json:"tags"`
	Version      string   `json:"version"`
}

type distribution struct {
	Distribution string `json:"distribution"`
	Version      string `json:"version"`
}

type compatibilityProfile struct {
	AcceleratorBuild string `json:"accelerator_build"`
	OSCPU            string `json:"os_cpu"`
	PythonABI        string `json:"python_abi"`
	TorchRelease     string `json:"torch_release"`
}

type wheelhouseManifest struct {
	Format               string               `json:"format"`
	BaseDistributions    []distribution       `json:"base_distributions"`
	BaseImportRoots      []json.RawMessage    `json:"base_import_roots"`
	CompatibilityProfile compatibilityProfile `json:"compatibility_profile"`
	Wheels               []Wheel              `json:"wheels"`
}

type Identity struct {
	Profile            string `json:"profile"`
	WheelhouseManifest Ref    `json:"wheelhouse_manifest"`
	Image              string `json:"image"`
	ImageConfigDigest  string `json:"image_config_digest"`
	PythonABI          string `json:"python_abi"`
	PythonVersion      string `json:"python_version"`
	TorchVersion       string `json:"torch_version"`
	CUDA               string `json:"cuda"`
	Runtime            Wheel  `json:"runtime"`
	TensorFS           Wheel  `json:"tensorfs"`
	BaseImage          string `json:"base_image"`
	SourceRepository   string `json:"source_repository"`
	SourceRevision     string `json:"source_revision"`
}

type Record struct {
	Format        string   `json:"record"`
	BaseID        string   `json:"base_id"`
	InstalledRoot string   `json:"installed_root"`
	Identity      Identity `json:"identity"`
}

type dockerInspect struct {
	ID          string   `json:"Id"`
	RepoDigests []string `json:"RepoDigests"`
	Config      struct {
		Labels map[string]string `json:"Labels"`
	} `json:"Config"`
}

type imageProbe struct {
	ManifestSHA256 string `json:"manifest_sha256"`
	Python         string `json:"python"`
	Runtime        string `json:"runtime"`
	TensorFS       string `json:"tensorfs"`
	Torch          string `json:"torch"`
	TorchCUDA      string `json:"torch_cuda"`
	Worker         bool   `json:"worker"`
}

// Ensure verifies or produces the selected package-independent base, then
// atomically activates its receipt. A failed refresh never moves the active file.
func Ensure(root string, input Input) (*Record, *exit.Error) {
	manifest, runtimeWheel, tensorFSWheel, problem := validateInput(input)
	if problem != nil {
		return nil, problem
	}
	docker, err := exec.LookPath("docker")
	if err != nil {
		return nil, exit.Named(exit.Structural, "managed_base.docker_missing",
			"the exact local Runtime base requires Docker, but no docker client is installed").
			WithRemedy("install Docker with NVIDIA Container Toolkit support, then retry the same package install")
	}

	// A selected manifest has one deterministic directory. If it was already
	// produced, its immutable receipt tells us the exact image to restore by
	// digest; a mutable profile tag is never consulted on this path.
	dir := generationDir(root, input.Profile, input.Manifest.Digest)
	if raw, err := os.ReadFile(filepath.Join(dir, "receipt.json")); err == nil {
		var record Record
		storedManifest, manifestErr := os.ReadFile(filepath.Join(dir, "WheelhouseManifest.json"))
		if json.Unmarshal(raw, &record) != nil || validateRecord(&record, dir, input,
			runtimeWheel, tensorFSWheel) != nil || manifestErr != nil || !bytes.Equal(storedManifest, input.Manifest.Bytes) {
			return nil, exit.Named(exit.Conflict, "managed_base.receipt_invalid",
				"the managed base receipt at %s is corrupt or names another selection", dir).
				WithRemedy("move that one managed-base generation aside and retry the package install")
		}
		imageInspect, inspectErr := inspectImage(docker, record.Identity.Image)
		if inspectErr != nil {
			if pullErr := run(docker, "pull", record.Identity.Image); pullErr != nil {
				return nil, exit.Named(exit.Unavailable, "managed_base.image_unavailable",
					"the exact managed base image %s is absent and could not be restored", record.Identity.Image).
					WithRemedy("Docker said: %s", condense(pullErr.Error()))
			}
			imageInspect, inspectErr = inspectImage(docker, record.Identity.Image)
		}
		if inspectErr != nil || !hasRepoDigest(imageInspect.RepoDigests, record.Identity.Image) ||
			labelsMatch(imageInspect.Config.Labels, input, runtimeWheel) != nil ||
			imageInspect.ID != record.Identity.ImageConfigDigest {
			return nil, exit.Named(exit.Conflict, "managed_base.image_mismatch",
				"the installed managed base image no longer matches its exact receipt")
		}
		if problem := activate(root, &record); problem != nil {
			return nil, problem
		}
		return &record, nil
	}

	tag := imageRepository + ":" + input.Profile
	// A new local generation begins with registry readback of the public profile
	// alias. The alias is only discovery: after verification, the receipt and all
	// launches retain the returned immutable repository digest.
	if pullErr := run(docker, "pull", tag); pullErr != nil {
		return nil, exit.Named(exit.Unavailable, "managed_base.image_unavailable",
			"Docker could not fetch the selected local Runtime base %s", tag).
			WithRemedy("Docker said: %s", condense(pullErr.Error()))
	}
	inspected, inspectErr := inspectImage(docker, tag)
	if inspectErr != nil {
		return nil, exit.Named(exit.Structural, "managed_base.image_invalid",
			"Docker cannot inspect the selected local Runtime base %s: %s", tag, inspectErr)
	}
	labels := inspected.Config.Labels
	if err := labelsMatch(labels, input, runtimeWheel); err != nil {
		return nil, exit.Named(exit.Conflict, "managed_base.image_mismatch",
			"the published local Runtime base does not match Tensorhub's selected profile: %s", err).
			WithRemedy("retry after the exact %s image is published", input.Profile)
	}
	image, err := immutableImage(inspected.RepoDigests)
	if err != nil {
		return nil, exit.Named(exit.Conflict, "managed_base.image_identity_missing", "%s", err)
	}
	probe, err := probeImage(docker, image)
	if err != nil {
		return nil, exit.Named(exit.Structural, "managed_base.proof_failed",
			"the selected local Runtime base failed its installed-environment proof: %s", err)
	}
	if err := verifyProbe(probe, input, labels); err != nil {
		return nil, exit.Named(exit.Conflict, "managed_base.proof_mismatch",
			"the selected local Runtime base disagrees with its exact record: %s", err)
	}

	identity := Identity{
		Profile:            input.Profile,
		WheelhouseManifest: Ref{Digest: input.Manifest.Digest, Length: input.Manifest.Length},
		Image:              image, ImageConfigDigest: inspected.ID,
		PythonABI:     manifest.CompatibilityProfile.PythonABI,
		PythonVersion: labels["cozy.python.version"], TorchVersion: labels["cozy.torch.version"],
		CUDA: labels["cozy.cuda.version"], Runtime: runtimeWheel, TensorFS: tensorFSWheel,
		BaseImage:        labels["cozy.base_image.source"],
		SourceRepository: labels["org.opencontainers.image.source"],
		SourceRevision:   labels["org.opencontainers.image.revision"],
	}
	identityBytes, _ := json.Marshal(identity)
	sum := sha256.Sum256(identityBytes)
	baseID := "sha256:" + hex.EncodeToString(sum[:])
	record := &Record{Format: RecordFormat, BaseID: baseID, InstalledRoot: dir, Identity: identity}
	if problem := persist(dir, record, input.Manifest.Bytes); problem != nil {
		return nil, problem
	}
	if problem := activate(root, record); problem != nil {
		return nil, problem
	}
	return record, nil
}

// Open reads the exact managed base selected by a package's retained profile
// and WheelhouseManifest. It performs no fallback to PATH or to an active base.
func Open(root, profile, manifestDigest string) (*Record, *exit.Error) {
	if profile != SupportedProfile {
		return nil, exit.Named(exit.Unavailable, "managed_base.profile_unsupported",
			"Creator can run only %s locally; this install selected %q", SupportedProfile, profile)
	}
	if _, err := canonical.Raw(manifestDigest); err != nil {
		return nil, exit.Named(exit.Conflict, "managed_base.manifest_identity_mismatch",
			"the installed package names an invalid WheelhouseManifest digest")
	}
	dir := generationDir(root, profile, manifestDigest)
	raw, err := os.ReadFile(filepath.Join(dir, "receipt.json"))
	if err != nil {
		return nil, exit.Named(exit.Structural, "managed_base.missing",
			"the exact local Runtime base for %s (%s) is not installed", profile, manifestDigest).
			WithRemedy("reinstall the package so Creator can produce its selected managed base")
	}
	var record Record
	if json.Unmarshal(raw, &record) != nil || record.Format != RecordFormat ||
		record.InstalledRoot != dir || record.Identity.Profile != profile ||
		record.Identity.WheelhouseManifest.Digest != manifestDigest ||
		record.Identity.PythonABI != "cp312" || record.Identity.PythonVersion != "3.12.3" ||
		record.Identity.TorchVersion != "2.13.0+cu130" || record.Identity.CUDA != "13.0" ||
		record.Identity.Runtime.Version != "0.0.11" || record.Identity.TensorFS.Version != "0.0.3" ||
		!strings.HasPrefix(record.Identity.Image, imageRepository+"@sha256:") {
		return nil, exit.Named(exit.Conflict, "managed_base.receipt_invalid",
			"the managed base receipt at %s is corrupt or names another selection", dir)
	}
	identityBytes, _ := json.Marshal(record.Identity)
	sum := sha256.Sum256(identityBytes)
	if record.BaseID != "sha256:"+hex.EncodeToString(sum[:]) {
		return nil, exit.Named(exit.Conflict, "managed_base.receipt_invalid",
			"the managed base receipt at %s has a changed identity", dir)
	}
	return &record, nil
}

func validateInput(input Input) (wheelhouseManifest, Wheel, Wheel, *exit.Error) {
	var manifest wheelhouseManifest
	if input.Profile != SupportedProfile {
		return manifest, Wheel{}, Wheel{}, exit.Named(exit.Unavailable, "managed_base.profile_unsupported",
			"Creator can run only %s locally; Tensorhub selected %q", SupportedProfile, input.Profile)
	}
	rawDigest, err := canonical.Raw(input.Manifest.Digest)
	if err != nil || input.Manifest.Length != int64(len(input.Manifest.Bytes)) ||
		!bytes.Equal(canonical.Digest(input.Manifest.Bytes), rawDigest) {
		return manifest, Wheel{}, Wheel{}, exit.Named(exit.Conflict, "managed_base.manifest_identity_mismatch",
			"Tensorhub's selected WheelhouseManifest bytes do not match their digest and length")
	}
	if _, err := canonical.ReadObject(input.Manifest.Bytes); err != nil || json.Unmarshal(input.Manifest.Bytes, &manifest) != nil {
		return manifest, Wheel{}, Wheel{}, exit.Named(exit.Conflict, "managed_base.manifest_invalid",
			"Tensorhub's selected WheelhouseManifest is not exact canonical JSON")
	}
	profile := manifest.CompatibilityProfile
	if manifest.Format != "WheelhouseManifest/1" || profile.PythonABI != "cp312" ||
		profile.TorchRelease != "2.13.0" || profile.AcceleratorBuild != "cu130" || profile.OSCPU != "linux-x86" {
		return manifest, Wheel{}, Wheel{}, exit.Named(exit.Conflict, "managed_base.manifest_profile_mismatch",
			"the selected WheelhouseManifest does not describe CPython 3.12 / Torch 2.13 / CUDA 13 / Linux x86")
	}
	runtimeWheel, err := exactWheel(manifest, "cozy-runtime", "0.0.11") //cozy:allow exact managed-base distribution identity
	if err != nil {
		return manifest, Wheel{}, Wheel{}, exit.Named(exit.Conflict, "managed_base.runtime_invalid", "%s", err)
	}
	tensorFSWheel, err := exactWheel(manifest, "tensorfs", "0.0.3")
	if err != nil {
		return manifest, Wheel{}, Wheel{}, exit.Named(exit.Conflict, "managed_base.tensorfs_invalid", "%s", err)
	}
	if version(manifest.BaseDistributions, "torch") != "2.13.0+cu130" {
		return manifest, Wheel{}, Wheel{}, exit.Named(exit.Conflict, "managed_base.torch_invalid",
			"the selected base does not own exact Torch 2.13.0+cu130")
	}
	return manifest, runtimeWheel, tensorFSWheel, nil
}

func exactWheel(manifest wheelhouseManifest, name, wantVersion string) (Wheel, error) {
	var found []Wheel
	for _, wheel := range manifest.Wheels {
		if wheel.Distribution == name {
			found = append(found, wheel)
		}
	}
	if len(found) != 1 || found[0].Version != wantVersion || found[0].Length <= 0 {
		return Wheel{}, fmt.Errorf("selected base must carry one exact %s %s wheel", name, wantVersion)
	}
	if _, err := canonical.Raw(found[0].Digest); err != nil {
		return Wheel{}, fmt.Errorf("selected %s wheel digest is invalid", name)
	}
	if version(manifest.BaseDistributions, name) != wantVersion {
		return Wheel{}, fmt.Errorf("selected %s wheel and installed distribution disagree", name)
	}
	return found[0], nil
}

func version(rows []distribution, name string) string {
	value := ""
	for _, row := range rows {
		if row.Distribution == name {
			if value != "" {
				return ""
			}
			value = row.Version
		}
	}
	return value
}

func labelsMatch(labels map[string]string, input Input, runtimeWheel Wheel) error {
	want := map[string]string{
		"cozy.base_worker.profile":        input.Profile,
		"cozy.wheelhouse_manifest_digest": input.Manifest.Digest,
		"cozy.python.version":             "3.12.3",
		"cozy.torch.version":              "2.13.0+cu130",
		"cozy.cuda.version":               "13.0",
		"cozy.control_runtime_digest":     runtimeWheel.Digest,
		"cozy.control_runtime_length":     strconv.FormatInt(runtimeWheel.Length, 10),
		"cozy.runtime_worker.entrypoint":  "/opt/cozy/bin/cozy-runtime-worker",
	}
	for name, value := range want {
		if labels[name] != value {
			return fmt.Errorf("image label %s is %q, want %q", name, labels[name], value)
		}
	}
	if !strings.Contains(labels["cozy.base_image.source"], "@sha256:") {
		return fmt.Errorf("base image source is not digest-pinned")
	}
	if !strings.HasPrefix(labels["org.opencontainers.image.source"], "https://github.com/") {
		return fmt.Errorf("image source repository is absent")
	}
	if !commitRE.MatchString(labels["org.opencontainers.image.revision"]) {
		return fmt.Errorf("image source revision is not an exact Git commit")
	}
	return nil
}

func inspectImage(docker, ref string) (dockerInspect, error) {
	var out []dockerInspect
	cmd := exec.Command(docker, "image", "inspect", ref)
	cmd.Env = config.Frozen().Tool()
	raw, err := cmd.Output()
	if err != nil {
		return dockerInspect{}, err
	}
	if json.Unmarshal(raw, &out) != nil || len(out) != 1 || out[0].ID == "" {
		return dockerInspect{}, fmt.Errorf("docker returned an invalid image inspection")
	}
	return out[0], nil
}

func immutableImage(repoDigests []string) (string, error) {
	for _, value := range repoDigests {
		if strings.HasPrefix(value, "tensorhub/worker@sha256:") {
			return "docker.io/" + value, nil
		}
		if strings.HasPrefix(value, imageRepository+"@sha256:") {
			return value, nil
		}
	}
	return "", fmt.Errorf("the selected image has no immutable %s repository digest", imageRepository)
}

func hasRepoDigest(repoDigests []string, image string) bool {
	short := strings.TrimPrefix(image, "docker.io/")
	for _, value := range repoDigests {
		if value == image || value == short {
			return true
		}
	}
	return false
}

const imageProbeScript = `import hashlib,importlib.metadata as m,json,os,re,sys,torch
p="/opt/cozy/base-runtime/WheelhouseManifest.json"
raw=open(p,"rb").read()
house=json.loads(raw)
norm=lambda s:re.sub(r"[-_.]+","-",s).lower()
got=sorted(({"distribution":norm(d.metadata["Name"]),"version":d.version} for d in m.distributions(path=["/usr/local/lib/python3.12/dist-packages"])),key=lambda x:x["distribution"])
if got!=house["base_distributions"]: raise SystemExit("installed distribution inventory mismatch")
print(json.dumps({"manifest_sha256":hashlib.sha256(raw).hexdigest(),"python":".".join(map(str,sys.version_info[:3])),"runtime":m.version("cozy-runtime"),"tensorfs":m.version("tensorfs"),"torch":torch.__version__,"torch_cuda":torch.version.cuda or "none","worker":os.access("/usr/local/bin/cozy-runtime-worker",os.X_OK)},sort_keys=True,separators=(",",":"))) # //cozy:allow exact managed-base installed proof`

func probeImage(docker, image string) (imageProbe, error) {
	cmd := exec.Command(docker, "run", "--rm", "--pull=never", "--network=none",
		"--entrypoint", "/usr/bin/python", image, "-c", imageProbeScript)
	cmd.Env = config.Frozen().Tool()
	var stderr strings.Builder
	cmd.Stderr = &stderr
	raw, err := cmd.Output()
	if err != nil {
		return imageProbe{}, fmt.Errorf("%s", condense(stderr.String()))
	}
	var probe imageProbe
	if json.Unmarshal(bytes.TrimSpace(raw), &probe) != nil {
		return imageProbe{}, fmt.Errorf("image proof returned invalid JSON")
	}
	return probe, nil
}

func verifyProbe(probe imageProbe, input Input, labels map[string]string) error {
	if probe.ManifestSHA256 != strings.TrimPrefix(input.Manifest.Digest, "sha256:") {
		return fmt.Errorf("baked WheelhouseManifest digest changed")
	}
	for name, gotWant := range map[string][2]string{
		"Python":   {probe.Python, labels["cozy.python.version"]},
		"Runtime":  {probe.Runtime, "0.0.11"},
		"TensorFS": {probe.TensorFS, "0.0.3"},
		"Torch":    {probe.Torch, "2.13.0+cu130"},
		"CUDA":     {probe.TorchCUDA, "13.0"},
	} {
		if gotWant[0] != gotWant[1] {
			return fmt.Errorf("%s is %q, want %q", name, gotWant[0], gotWant[1])
		}
	}
	if !probe.Worker {
		return fmt.Errorf("cozy-runtime-worker entrypoint is absent")
	}
	return nil
}

func generationDir(root, profile, digest string) string {
	return filepath.Join(root, profile, strings.TrimPrefix(digest, "sha256:"))
}

func validateRecord(record *Record, dir string, input Input, runtimeWheel, tensorFSWheel Wheel) error {
	if record.Format != RecordFormat || record.InstalledRoot != dir ||
		record.Identity.Profile != input.Profile || record.Identity.WheelhouseManifest.Digest != input.Manifest.Digest ||
		record.Identity.WheelhouseManifest.Length != input.Manifest.Length ||
		!reflect.DeepEqual(record.Identity.Runtime, runtimeWheel) ||
		!reflect.DeepEqual(record.Identity.TensorFS, tensorFSWheel) ||
		record.Identity.PythonABI != "cp312" || record.Identity.PythonVersion != "3.12.3" ||
		record.Identity.TorchVersion != "2.13.0+cu130" || record.Identity.CUDA != "13.0" ||
		!strings.HasPrefix(record.Identity.Image, imageRepository+"@sha256:") {
		return fmt.Errorf("record fields")
	}
	identityBytes, _ := json.Marshal(record.Identity)
	sum := sha256.Sum256(identityBytes)
	if record.BaseID != "sha256:"+hex.EncodeToString(sum[:]) {
		return fmt.Errorf("base identity")
	}
	return nil
}

func persist(dir string, record *Record, manifest []byte) *exit.Error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return exit.Internalf("cannot create managed base generation %s: %s", dir, err)
	}
	if err := atomicWrite(filepath.Join(dir, "WheelhouseManifest.json"), manifest, 0o600); err != nil {
		return exit.Internalf("cannot persist the managed base inventory: %s", err)
	}
	raw, err := json.Marshal(record)
	if err != nil {
		return exit.Internalf("cannot encode the managed base receipt: %s", err)
	}
	if err := atomicWrite(filepath.Join(dir, "receipt.json"), raw, 0o600); err != nil {
		return exit.Internalf("cannot persist the managed base receipt: %s", err)
	}
	return nil
}

func activate(root string, record *Record) *exit.Error {
	dir := filepath.Join(root, "active")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return exit.Internalf("cannot create the managed base activation directory: %s", err)
	}
	raw, _ := json.Marshal(struct {
		BaseID string `json:"base_id"`
		Root   string `json:"root"`
	}{record.BaseID, record.InstalledRoot})
	if err := atomicWrite(filepath.Join(dir, record.Identity.Profile+".json"), raw, 0o600); err != nil {
		return exit.Internalf("cannot activate managed base %s: %s", record.BaseID, err)
	}
	return nil
}

func atomicWrite(path string, data []byte, mode os.FileMode) error {
	file, err := os.CreateTemp(filepath.Dir(path), ".refresh-*")
	if err != nil {
		return err
	}
	tmp := file.Name()
	defer os.Remove(tmp)
	if err := file.Chmod(mode); err != nil {
		file.Close()
		return err
	}
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	parent, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer parent.Close()
	return parent.Sync()
}

func run(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Env = config.Frozen().Tool()
	var output strings.Builder
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s: %s", err, output.String())
	}
	return nil
}

func condense(value string) string {
	return strings.Join(strings.Fields(value), " ")
}
