package producttest

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// maintenancePod is a development pod's SSH maintenance endpoint: fake_pod_ssh.py as
// `ssh` and `sftp` on the daemon's PATH, over a local pod root (runs 1403, citron).
type maintenancePod struct{ dir string }

func newMaintenancePod(t *testing.T) *maintenancePod {
	t.Helper()
	p := &maintenancePod{dir: t.TempDir()}
	script, err := filepath.Abs(filepath.Join("testdata", "fake_pod_ssh.py"))
	must(t, err)
	bin := filepath.Join(p.dir, "bin")
	must(t, os.MkdirAll(bin, 0o755))
	must(t, os.MkdirAll(filepath.Join(p.dir, "pod", "var", "lib", "cozy", "dev"), 0o755))
	for _, name := range []string{"ssh", "sftp"} {
		must(t, os.Symlink(script, filepath.Join(bin, name)))
	}
	observed, err := json.Marshal(map[string]any{
		"runtime": map[string]any{"distribution": "0.18.54", "supports_guarded_restart": true}, "tensorfs": "0.3.63",
		"python": "3.12.3", "tags": []string{"cp312-abi3-manylinux_2_28_x86_64"}, "updater": true, "durable_updates": true,
		"markers":   map[string]string{"python_version": "3.12", "sys_platform": "linux", "platform_machine": "x86_64"},
		"selection": "", "torch": ""})
	must(t, err)
	must(t, os.WriteFile(filepath.Join(p.dir, "observed.json"), observed, 0o644))
	return p
}

// path is the daemon's PATH with this pod's endpoint first. The product's configuration
// is loaded once per process, so it is imposed on the daemon rather than set here.
func (p *maintenancePod) path(t *testing.T, root string) string {
	path := filepath.Join(p.dir, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	return "PATH=" + path
}

func (p *maintenancePod) sftpMode(t *testing.T, mode string) {
	t.Helper()
	must(t, os.WriteFile(filepath.Join(p.dir, "sftp-mode"), []byte(mode), 0o644))
}

func (p *maintenancePod) exists(name string) bool {
	_, err := os.Stat(filepath.Join(p.dir, name))
	return err == nil
}

func (p *maintenancePod) sftpLog(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(p.dir, "sftp.log"))
	must(t, err)
	return string(data)
}

// localRuntimePair writes a local Runtime and TensorFS build large enough to take
// several interrupted transfers, as the owner's --runtime-wheel/--tensorfs-wheel.
func localRuntimePair(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	write := func(file, name, version string, requires ...string) string {
		var buffer bytes.Buffer
		archive := zip.NewWriter(&buffer)
		distInfo := strings.ReplaceAll(name, "-", "_") + "-" + version + ".dist-info/"
		metadata, err := archive.Create(distInfo + "METADATA")
		must(t, err)
		text := "Metadata-Version: 2.3\nName: " + name + "\nVersion: " + version + "\nRequires-Python: >=3.12\n"
		for _, requirement := range requires {
			text += "Requires-Dist: " + requirement + "\n"
		}
		_, err = metadata.Write([]byte(text))
		must(t, err)
		padding, err := archive.CreateHeader(&zip.FileHeader{Name: distInfo + "padding", Method: zip.Store})
		must(t, err)
		_, err = padding.Write(bytes.Repeat([]byte(name), 12_000))
		must(t, err)
		must(t, archive.Close())
		path := filepath.Join(dir, file)
		must(t, os.WriteFile(path, buffer.Bytes(), 0o644))
		return path
	}
	return write("cozy_runtime-0.18.60+dev.proof-cp312-abi3-manylinux_2_28_x86_64.whl", "cozy-runtime", "0.18.60+dev.proof", "tensorfs>=0.3.60"), //cozy:allow a wheel's distribution name, not the binary
		write("tensorfs-0.3.66+dev.proof-cp312-abi3-manylinux_2_28_x86_64.whl", "tensorfs", "0.3.66+dev.proof")
}

// serveMaintenance makes the Hub serve the attached rental as a ready development pod.
func serveMaintenance(t *testing.T, layout home.Layout, creator []byte, set func(key string, value any)) {
	t.Helper()
	cert, err := os.ReadFile(layout.RentalCert(podRental))
	must(t, err)
	token, problem := rental.MediaToken(layout, podRental)
	fatal(t, problem)
	for key, value := range map[string]any{"development": true, "ssh_address": "127.0.0.1:2222", "cert_pem": string(cert),
		"worker_id": podWorkerID, "worker_boot_id": podBootID, "creator_public_key": base64.RawURLEncoding.EncodeToString(creator),
		"media_token_sha256": []string{secret.HashHex(token)}} {
		set(key, value)
	}
	// The maintenance identity the owner's SSH key would be; the fake endpoint ignores it.
	must(t, os.MkdirAll(filepath.Join(layout.Root, "auth"), 0o700))
	must(t, os.WriteFile(filepath.Join(layout.Root, "auth", "rental-ssh"), []byte("fixture key"), 0o600))
}

// transportLogs is every update's retained transport diagnostics under root.
func transportLogs(root string) string {
	paths, _ := filepath.Glob(filepath.Join(root, "tmp", "runtime-updates", "*", "transport.log"))
	text := ""
	for _, path := range paths {
		data, _ := os.ReadFile(path)
		text += path + ":\n" + string(data) + "\n"
	}
	return text
}

// eventually waits for ok; the bound only catches a hang on a loaded shared box.
func eventually(t *testing.T, root, what string, ok func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Minute); !ok(); time.Sleep(50 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("%s did not happen: %s\n%s", what, tail(filepath.Join(root, "daemon.log")), transportLogs(root))
		}
	}
}

// cozyWithin runs one CLI command, failing the test if it outlives `within`.
func cozyWithin(t *testing.T, root string, within time.Duration, args ...string) (int, string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), within)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	data, _ := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("cozy %s hung past %s: %s\n%s", strings.Join(args, " "), within, data, tail(filepath.Join(root, "daemon.log")))
	}
	return cmd.ProcessState.ExitCode(), string(data)
}

// A Runtime update whose wheel transfer never reaches the worker ends there: the worker was
// not changed and keeps serving, and the update says so instead of staying "updating"
// (citron, run 1403). An interrupted upload resumes from what arrived, while it progresses.
func TestAnUpdateThatNeverReachedTheWorkerEndsAndTheUploadResumes(t *testing.T) {
	pod := newMaintenancePod(t)
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	identity, problem := rental.PendingCreatorIdentity(layout, "runtime-update-failure")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	worker := &fakePod{controlKey: public}
	connection, certPath := startFakePod(t, root, worker)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "citron")
	for key, value := range map[string]any{"requested_accelerator_model": "fake-4090", "worker_address": connection.Addr,
		"media_address": connection.Media.Addr} {
		hub.set(podRental, key, value)
	}
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "citron", State: "ready", SKU: "cpu",
		AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID},
		string(cert), connection.Media.Token, identity))
	serveMaintenance(t, layout, public, func(key string, value any) { hub.set(podRental, key, value) })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	startDaemonProcess(t, root, pod.path(t, root))
	runtimeWheel, tensorfsWheel := localRuntimePair(t)

	// A pod rented without SSH maintenance is refused before anything is recorded.
	hub.set(podRental, "development", false)
	if code, out := cozyWithin(t, root, time.Minute, "rental", "update", "citron"); code == 0 || !strings.Contains(out, "rented without SSH maintenance") {
		t.Fatalf("an update of a pod without maintenance was not refused up front [exit %d]: %s", code, out)
	}
	if update, problem := store.RuntimeUpdate(podRental); problem != nil || update != nil {
		t.Fatalf("the refused update was recorded: %+v %v", update, problem)
	}
	hub.set(podRental, "development", true)

	// The connection drops before a byte arrives, every time.
	pod.sftpMode(t, "drop")
	code, out := cozyWithin(t, root, 5*time.Minute, "rental", "update", "citron", "--runtime-wheel", runtimeWheel, "--tensorfs-wheel", tensorfsWheel)
	if code == 0 || !strings.Contains(out, "never reached the worker") ||
		!strings.Contains(out, "keeps serving Runtime 0.18.54") {
		t.Fatalf("an update that never reached the worker did not end as failed [exit %d]: %s\n%s", code, out, transportLogs(root))
	}
	update, problem := store.RuntimeUpdate(podRental)
	fatal(t, problem)
	if update == nil || update.State != "failed" {
		t.Fatalf("the update did not end: %+v", update)
	}
	if _, list := cozyWithin(t, root, time.Minute, "rental", "list"); !strings.Contains(list, "citron") || strings.Contains(list, "Runtime") {
		t.Fatalf("rental list does not show citron serving again:\n%s", list)
	}

	// Each attempt now lands a slice of the pair: the upload resumes where it stopped.
	pod.sftpMode(t, "partial:20000")
	code, out = cozyWithin(t, root, 5*time.Minute, "rental", "update", "citron", "--runtime-wheel", runtimeWheel, "--tensorfs-wheel", tensorfsWheel)
	if code != 0 || !strings.Contains(out, "0.18.60+dev.proof") {
		t.Fatalf("an interrupted upload did not resume to a finished update [exit %d]: %s\n%s", code, out, pod.sftpLog(t))
	}
	if log := pod.sftpLog(t); !strings.Contains(log, "reput ") {
		t.Fatalf("the upload restarted from zero instead of resuming:\n%s", log)
	}
}

// A run parked behind its rental's Runtime update is on the rental's books while it waits,
// and the failed update releases it to the machine instead of leaving it parked until the
// idle release destroys the pod under it (run 1403).
func TestAParkedRunReachesItsMachineWhenTheUpdateFails(t *testing.T) {
	pod := newMaintenancePod(t)
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &runtimeMachine{}
	worker := &fakePod{machine: machine}
	root, layout := rentedLadderHome(t, h, worker, nil)
	serveMaintenance(t, layout, worker.controlKey, func(key string, value any) {
		h.mu.Lock()
		h.rentals[podRental][key] = value
		h.mu.Unlock()
	})
	startDaemonProcess(t, root, pod.path(t, root))
	runtimeWheel, tensorfsWheel := localRuntimePair(t)

	pod.sftpMode(t, "hold")
	updated := make(chan string, 1)
	go func() {
		cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "rental", "update", "tessa", "--runtime-wheel", runtimeWheel, "--tensorfs-wheel", tensorfsWheel)
		cmd.Env = childEnv(t, root)
		data, _ := cmd.CombinedOutput()
		updated <- string(data)
	}()
	eventually(t, root, "the update's wheel transfer", func() bool { return pod.exists("sftp-holding") })

	code, out := cozyWithin(t, root, time.Minute, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json")
	if code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	var list struct {
		Rentals []struct {
			Machine       string `json:"machine"`
			State         string `json:"state"`
			RuntimeUpdate string `json:"runtime_update"`
			Queued        int    `json:"queued"`
			IdleSince     string `json:"idle_since_at"`
		} `json:"rentals"`
	}
	raw := ""
	eventually(t, root, "rental list showing the run waiting on the updating rental", func() bool {
		_, raw = cozyWithin(t, root, time.Minute, "rental", "list", "--json")
		return json.Unmarshal([]byte(raw), &list) == nil && len(list.Rentals) == 1 &&
			list.Rentals[0].Queued == 1 && list.Rentals[0].RuntimeUpdate == "updating"
	})
	if list.Rentals[0].IdleSince != "" {
		t.Fatalf("the rental counts idle time while a run waits on it: %+v", list.Rentals[0])
	}
	if machine.submitted() != nil {
		t.Fatal("the run reached the machine during its Runtime update")
	}

	must(t, os.WriteFile(filepath.Join(pod.dir, "sftp-release"), nil, 0o644))
	select {
	case result := <-updated:
		if !strings.Contains(result, "never reached the worker") {
			t.Fatalf("the update did not end as never applied: %s", result)
		}
	case <-time.After(5 * time.Minute):
		t.Fatalf("the update did not end: %s", tail(filepath.Join(root, "daemon.log")))
	}
	eventually(t, root, "the parked run's submission to its machine", func() bool { return machine.submitted() != nil })
}

// An update the worker took but did not finish leaves the rental plainly unusable, never
// "updating": work sent to it fails at once naming the remedy, rental list says so, and the
// owner's `cozy rental update` resumes the same operation to a serving worker.
func TestAnUnfinishedUpdateMakesTheRentalUnusableUntilItsOwnerResumesIt(t *testing.T) {
	pod := newMaintenancePod(t)
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &runtimeMachine{}
	worker := &fakePod{machine: machine}
	root, layout := rentedLadderHome(t, h, worker, nil)
	serveMaintenance(t, layout, worker.controlKey, func(key string, value any) {
		h.mu.Lock()
		h.rentals[podRental][key] = value
		h.mu.Unlock()
	})
	startDaemonProcess(t, root, pod.path(t, root))
	runtimeWheel, tensorfsWheel := localRuntimePair(t)
	must(t, os.WriteFile(filepath.Join(pod.dir, "update-outcome"), []byte("recovery_required"), 0o644))

	code, out := cozyWithin(t, root, 5*time.Minute, "rental", "update", "tessa", "--runtime-wheel", runtimeWheel, "--tensorfs-wheel", tensorfsWheel)
	if code == 0 || !strings.Contains(out, "tessa is unusable") || !strings.Contains(out, "cozy rental update tessa retries it") {
		t.Fatalf("an unfinished update did not leave the rental plainly unusable [exit %d]: %s", code, out)
	}
	if _, list := cozyWithin(t, root, time.Minute, "rental", "list"); !strings.Contains(list, "unusable (Runtime update unfinished)") {
		t.Fatalf("rental list does not say the rental is unusable:\n%s", list)
	}
	if code, out := cozyWithin(t, root, time.Minute, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json"); code == 0 ||
		!strings.Contains(out, "rental.unusable") || !strings.Contains(out, "tessa is unusable") {
		t.Fatalf("a run sent to the unusable rental did not fail at once naming it [exit %d]: %s", code, out)
	}
	if machine.submitted() != nil {
		t.Fatal("a run reached a worker whose update is unfinished")
	}

	must(t, os.Remove(filepath.Join(pod.dir, "update-outcome")))
	if code, out := cozyWithin(t, root, 5*time.Minute, "rental", "update", "tessa"); code != 0 || !strings.Contains(out, "0.18.60+dev.proof") {
		t.Fatalf("the owner's resume did not finish the recorded update [exit %d]: %s", code, out)
	}
	if _, list := cozyWithin(t, root, time.Minute, "rental", "list"); strings.Contains(list, "Runtime") {
		t.Fatalf("rental list still holds the rental after its update finished:\n%s", list)
	}
}

// A Runtime update keeps the worker boot but not the Runtime: the capabilities read before
// it are not kept after it. The run after `cozy rental update` takes the one-message path
// the new Runtime offers, with no daemon restart.
func TestARuntimeUpdateRefreshesTheMachinesCapabilities(t *testing.T) {
	pod := newMaintenancePod(t)
	h := newLadderHub(t)
	h.bind(goodLadder())
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	machines.older.Store(true)
	worker := &fakePod{machine: machines}
	root, layout := rentedLadderHome(t, h, worker, nil)
	serveMaintenance(t, layout, worker.controlKey, func(key string, value any) {
		h.mu.Lock()
		h.rentals[podRental][key] = value
		h.mu.Unlock()
	})
	startDaemonProcess(t, root, pod.path())
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()

	rentedRun(t, root, store, "before", "generate", "steps=1")
	if submitted := machines.submitted(); len(submitted) != 1 || submitted[0].ReleaseRoot != nil {
		t.Fatalf("the older Runtime was not sent the compatibility submission: %d", len(submitted))
	}
	runtimeWheel, tensorfsWheel := localRuntimePair(t)
	machines.older.Store(false) // what the updated Runtime reports
	if code, out := cozyWithin(t, root, 5*time.Minute, "rental", "update", "tessa", "--runtime-wheel", runtimeWheel, "--tensorfs-wheel", tensorfsWheel); code != 0 {
		t.Fatalf("the Runtime update failed [exit %d]: %s\n%s", code, out, transportLogs(root))
	}
	row, _ := rentedRun(t, root, store, "after", "generate", "steps=1")
	submitted := machines.submitted()
	if row.State != "succeeded" || len(submitted) != 2 || submitted[1].ReleaseRoot == nil {
		t.Fatalf("the run after the update did not take the release-roots path (%s, %d submissions)", row.State, len(submitted))
	}
}
