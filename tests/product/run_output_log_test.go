package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// A run reports its work as it happens, over the real CLI, daemon, Host and Runtime. The
// package publishes three revisions of one output and one list item per revision. Each lands
// in its item's one stable file in the outputs folder while the run goes on: the CLI names
// the file once and each revision after, an image is replaced whole, a video is appended in
// place, and the result is the fold of the log.
// Canceled after its second revision, a run keeps that revision as its result. The Hub is
// asked nothing while the runs go on.
func TestRunReportsProductsAsTheyArrive(t *testing.T) {
	integration(t)
	if *privateScriptRuntimeWheel == "" || *machineHostBinary == "" {
		t.Skip("requires -script-runtime-wheel and -machine-host: a real Runtime on this computer's machine")
	}
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	hub := newMachineHub(t)
	// The caller's account is read once per credential and kept; a run asks no Hub.
	hub.mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"name":"proof"}`))
	})
	var asked sync.Mutex
	var calls []string
	served := hub.server.Config.Handler
	hub.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked.Lock()
		calls = append(calls, time.Now().Format("15:04:05.000")+" "+r.Method+" "+r.URL.Path)
		asked.Unlock()
		served.ServeHTTP(w, r)
	})
	hubCalls := func() []string {
		asked.Lock()
		defer asked.Unlock()
		return append([]string(nil), calls...)
	}
	root, err := os.MkdirTemp("", "cozy-output-log-")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\n"), 0o600))
	control := filepath.Join(t.TempDir(), "control")
	uv := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("/usr/bin/nice", append([]string{"-n", "19", "uv"}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v %s", err, out)
		}
	}
	uv("venv", control, "--python", "3.12")
	uv("pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if !t.Failed() {
			must(t, removeAllForce(root))
		}
	})
	project := outputLogProof(t, wheel)
	cozy := func(args ...string) *exec.Cmd {
		command := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
		command.Env = childEnv(t, root, "PATH="+path)
		return command
	}
	if out, err := cozy("package", "install", project, "--editable", "--json").CombinedOutput(); err != nil {
		t.Fatalf("package install: %v %s", err, out)
	}
	directory := filepath.Join(root, "outputs", "local-output-log-proof")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	// start runs grow with --await, as a person would, and returns its request and stderr.
	start := func(gate string) (*exec.Cmd, *bytes.Buffer, *records.Request) {
		t.Helper()
		must(t, os.MkdirAll(gate, 0o700))
		command := cozy("run", "local/output-log-proof/grow", "gate="+gate, "--await")
		var stderr bytes.Buffer
		command.Stdout, command.Stderr = &stderr, &stderr
		must(t, command.Start())
		var request *records.Request
		landed(t, "the run's record", func() bool {
			requests, _ := store.Requests("", "local/output-log-proof", 10)
			for _, row := range requests {
				if strings.Contains(string(row.Payload), gate) {
					request, _ = store.RequestByReference(row.ID)
				}
			}
			return request != nil
		})
		return command, &stderr, request
	}
	// revision opens gate k and waits until revision k is the image's one stable file, whole.
	revision := func(request *records.Request, gate string, k int) records.Product {
		t.Helper()
		must(t, os.WriteFile(filepath.Join(gate, fmt.Sprintf("go-%d", k)), nil, 0o600))
		stable := filepath.Join(directory, fmt.Sprintf("%d-image.png", request.Number))
		var shown records.Product
		landed(t, fmt.Sprintf("revision %d", k), func() bool {
			products, _ := store.Products(request.ID)
			var sets []records.Product
			for _, product := range products {
				if product.Output == "image" {
					sets = append(sets, product)
				}
			}
			if len(sets) < k {
				return false
			}
			shown = sets[k-1]
			data, err := os.ReadFile(stable)
			// The file is always some revision whole, and soon this one.
			return err == nil && digestOf(data) == shown.Digest
		})
		if shown.Op != records.ProductSet || shown.Label != fmt.Sprintf("Revision %d of 3", k) || shown.Path != stable ||
			shown.Rev != uint32(k) || shown.AppendedFrom != nil || shown.Item != fmt.Sprintf("%d/image", request.Number) {
			t.Fatalf("revision %d landed as %+v", k, shown)
		}
		return shown
	}

	gate := filepath.Join(t.TempDir(), "complete")
	command, stderr, request := start(gate)
	var last records.Product
	// From the first product on, the run is on its machine: nothing after asks the Hub.
	var before int
	for k := 1; k <= 3; k++ {
		last = revision(request, gate, k)
		if k == 1 {
			before = len(hubCalls())
		}
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("the completed run exited %v:\n%s", err, stderr)
	}
	// The CLI names each item's stable file once, then one line per revision.
	image := filepath.Join(directory, fmt.Sprintf("%d-image.png", request.Number))
	if strings.Count(stderr.String(), "  image  "+image+"\n") != 1 {
		t.Fatalf("the CLI did not name the image's file exactly once:\n%s", stderr)
	}
	for k := 1; k <= 3; k++ {
		frame := filepath.Join(directory, fmt.Sprintf("%d-frames-%d.png", request.Number, k))
		for _, line := range []string{fmt.Sprintf("  image r%d  ", k), fmt.Sprintf("Revision %d of 3", k), fmt.Sprintf("Frame %d", k), "  frames " + fmt.Sprint(k) + "  " + frame} {
			if !strings.Contains(stderr.String(), line) {
				t.Fatalf("the CLI never named %q as it landed:\n%s", line, stderr)
			}
		}
	}
	products, problem := store.Products(request.ID)
	fatal(t, problem)
	fold := records.Fold(products)
	if len(fold) != 4 || fold[0].Output != "image" || fold[0].Digest != last.Digest || fold[0].Rev != 3 {
		t.Fatalf("the completed run's fold is %+v", fold)
	}
	for index, product := range fold[1:] {
		want := filepath.Join(directory, fmt.Sprintf("%d-frames-%d.png", request.Number, index+1))
		if product.Output != "frames" || product.Op != records.ProductAppend || product.Index != uint32(index) || product.Path != want || product.Rev != 1 {
			t.Fatalf("frame %d is %+v", index, product)
		}
	}
	export, problem := store.OutputExportOf(request.ID)
	fatal(t, problem)
	if export == nil || export.State != "published" || len(export.PublishedPaths) != 4 {
		t.Fatalf("the completed run's result was not written: %+v", export)
	}
	final, err := os.ReadFile(image)
	if err != nil || digestOf(final) != last.Digest {
		t.Fatalf("the final image is not the last revision: %v", err)
	}
	// Its events are OpenAI Responses' shape: each item added once, a delta per revision, done
	// when it can no longer change (a list element at once, the image at the terminal), and
	// the terminal carrying the run's output; every sequence_number above the one before.
	events, problem := store.EventsAfter(request.ID, 0, 1000)
	fatal(t, problem)
	var sequence []string
	var previous int64
	var output []records.OutputItem
	for _, event := range events {
		if event.Seq <= previous {
			t.Fatalf("event %d did not follow %d", event.Seq, previous)
		}
		previous = event.Seq
		switch event.Type {
		case records.OutputItemAdded, records.OutputItemDone:
			var envelope struct {
				Item records.OutputItem `json:"item"`
			}
			must(t, json.Unmarshal(event.Raw, &envelope))
			sequence = append(sequence, event.Type+" "+envelope.Item.ID+" "+envelope.Item.Status+" "+envelope.Item.Sha256)
		case records.OutputItemDelta:
			var delta records.OutputDelta
			must(t, json.Unmarshal(event.Raw, &delta))
			sequence = append(sequence, fmt.Sprintf("%s %s r%d", event.Type, delta.ItemID, delta.Rev))
		case "run.completed":
			var terminal struct {
				Output []records.OutputItem `json:"output"`
			}
			must(t, json.Unmarshal(event.Raw, &terminal))
			output = terminal.Output
			sequence = append(sequence, event.Type)
		}
	}
	n := request.Number
	frameDigest := func(k int) string {
		data, err := os.ReadFile(filepath.Join(directory, fmt.Sprintf("%d-frames-%d.png", n, k)))
		must(t, err)
		return digestOf(data)
	}
	want := []string{fmt.Sprintf("output_item.added %d/image in_progress ", n)}
	for k := 1; k <= 3; k++ {
		want = append(want, fmt.Sprintf("output_item.delta %d/image r%d", n, k),
			fmt.Sprintf("output_item.added %d/frames/%d in_progress ", n, k),
			fmt.Sprintf("output_item.delta %d/frames/%d r1", n, k),
			fmt.Sprintf("output_item.done %d/frames/%d completed %s", n, k, frameDigest(k)))
	}
	want = append(want, fmt.Sprintf("output_item.done %d/image completed %s", n, last.Digest), "run.completed")
	if !slices.Equal(sequence, want) {
		t.Fatalf("the run's output events are\n%s\nwant\n%s", strings.Join(sequence, "\n"), strings.Join(want, "\n"))
	}
	if len(output) != 4 || output[0].ID != fmt.Sprintf("%d/image", n) || output[0].Sha256 != last.Digest || output[0].Rev != 3 || output[0].Path != image {
		t.Fatalf("run.completed does not carry the run's output: %+v", output)
	}
	// The machine itself serves the image, the current bytes of its third revision.
	machineServesOutput(t, root, "image", last.Digest, 3)

	// Run 1591's shape: --input, --await, --json and --out, for a run that publishes and ends
	// at once, its last revisions and its outcome landing together. --await returns once each
	// output is in the folder at its final revision, and names each saved file.
	gate = filepath.Join(t.TempDir(), "at-once")
	must(t, os.MkdirAll(gate, 0o700))
	for k := 1; k <= 3; k++ {
		must(t, os.WriteFile(filepath.Join(gate, fmt.Sprintf("go-%d", k)), nil, 0o600))
	}
	input, out := filepath.Join(t.TempDir(), "input.json"), filepath.Join(t.TempDir(), "out")
	must(t, os.WriteFile(input, []byte(fmt.Sprintf(`{"gate":%q}`, gate)), 0o600))
	awaited, err := cozy("run", "local/output-log-proof/grow", "--input", input, "--await", "--json", "--out", out).Output()
	if err != nil {
		t.Fatalf("the awaited run exited %v:\n%s", err, awaited)
	}
	var result struct {
		Saved []struct{ Path, Digest string } `json:"saved"`
	}
	must(t, json.Unmarshal(awaited, &result))
	files, err := os.ReadDir(out)
	must(t, err)
	if len(result.Saved) != 4 || len(files) != 4 {
		t.Fatalf("the awaited run saved %d files and its folder holds %d, want its 4 outputs:\n%s", len(result.Saved), len(files), awaited)
	}
	for _, saved := range result.Saved {
		data, err := os.ReadFile(saved.Path)
		if err != nil || filepath.Dir(saved.Path) != out || digestOf(data) != saved.Digest {
			t.Fatalf("%s is not its output's final revision: %v", saved.Path, err)
		}
	}

	// A folder that refuses the outputs once the run is under way: --await exits non-zero,
	// naming each output it could not deliver and the machine that keeps its bytes. Once the
	// folder takes them again, watching the run delivers them.
	gate = filepath.Join(t.TempDir(), "refused")
	must(t, os.MkdirAll(gate, 0o700))
	refused := filepath.Join(t.TempDir(), "refused-out")
	must(t, os.WriteFile(input, []byte(fmt.Sprintf(`{"gate":%q}`, gate)), 0o600))
	command = cozy("run", "local/output-log-proof/grow", "--input", input, "--await", "--json", "--out", refused)
	var refusal bytes.Buffer
	command.Stdout = &refusal
	must(t, command.Start())
	landed(t, "the refused run's folder", func() bool { _, err := os.Stat(refused); return err == nil })
	must(t, os.Chmod(refused, 0o500))
	t.Cleanup(func() { _ = os.Chmod(refused, 0o700) })
	for k := 1; k <= 3; k++ {
		must(t, os.WriteFile(filepath.Join(gate, fmt.Sprintf("go-%d", k)), nil, 0o600))
	}
	if err := command.Wait(); err == nil || !strings.Contains(refusal.String(), `"code":"run.outputs_undelivered"`) ||
		!strings.Contains(refusal.String(), `"bytes_on":"machine local"`) || !strings.Contains(refusal.String(), "/image") {
		t.Fatalf("an awaited run whose folder refused its outputs exited %v:\n%s", err, refusal.String())
	}
	var undelivered struct {
		Error struct {
			Details struct {
				Number int64 `json:"number"`
			} `json:"details"`
		} `json:"error"`
	}
	must(t, json.Unmarshal(refusal.Bytes(), &undelivered))
	must(t, os.Chmod(refused, 0o700))
	if watched, err := cozy("run", "watch", fmt.Sprint(undelivered.Error.Details.Number), "--json").Output(); err != nil {
		t.Fatalf("watching the run once its folder took the outputs exited %v:\n%s", err, watched)
	}
	if files, err := os.ReadDir(refused); err != nil || len(files) != 4 {
		t.Fatalf("the restored folder holds %d files, want the run's 4 outputs: %v", len(files), err)
	}

	// Canceled after its second revision, the run keeps that revision as its result.
	gate = filepath.Join(t.TempDir(), "canceled")
	command, stderr, request = start(gate)
	revision(request, gate, 1)
	second := revision(request, gate, 2)
	if out, err := cozy("run", "cancel", fmt.Sprint(request.Number)).CombinedOutput(); err != nil {
		t.Fatalf("run cancel: %v %s", err, out)
	}
	if err := command.Wait(); err == nil || command.ProcessState.ExitCode() == 0 {
		t.Fatalf("a canceled run exited 0:\n%s", stderr)
	}
	landed(t, "the canceled run's result", func() bool {
		export, _ := store.OutputExportOf(request.ID)
		return export != nil && export.State == "published"
	})
	export, problem = store.OutputExportOf(request.ID)
	fatal(t, problem)
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	kept, err := os.ReadFile(filepath.Join(directory, fmt.Sprintf("%d-image.png", request.Number)))
	if row.State != "canceled" || len(export.PublishedPaths) != 3 || err != nil || digestOf(kept) != second.Digest {
		t.Fatalf("the canceled run did not keep revision 2 as its result: state %s export %+v: %v", row.State, export, err)
	}
	if !strings.Contains(stderr.String(), "kept") {
		t.Fatalf("the canceled run did not say what it kept:\n%s", stderr)
	}
	// Its image is done incomplete, at its last revision: revision 2's bytes and sha256.
	events, problem = store.EventsAfter(request.ID, 0, 1000)
	fatal(t, problem)
	done := false
	for _, event := range events {
		if event.Type == records.OutputItemDone {
			var envelope struct {
				Item records.OutputItem `json:"item"`
			}
			must(t, json.Unmarshal(event.Raw, &envelope))
			if envelope.Item.Name == "image" {
				done = envelope.Item.Status == "incomplete" && envelope.Item.Sha256 == second.Digest && envelope.Item.Rev == 2
			}
		}
	}
	if !done || events[len(events)-1].Type != "run.canceled" && !slices.ContainsFunc(events, func(e records.Event) bool { return e.Type == "run.canceled" }) {
		t.Fatal("the canceled run's image was not done incomplete at revision 2 before run.canceled")
	}
	// A growing video is one file a player reads while the run goes on, appended in place;
	// its last revision is the final MP4.
	gate = filepath.Join(t.TempDir(), "film")
	must(t, os.MkdirAll(gate, 0o700))
	command = cozy("run", "local/output-log-proof/film", "gate="+gate, "--await")
	var filmed bytes.Buffer
	command.Stdout, command.Stderr = &filmed, &filmed
	must(t, command.Start())
	var film *records.Request
	landed(t, "the film's record", func() bool {
		requests, _ := store.Requests("", "local/output-log-proof", 10)
		for _, row := range requests {
			if strings.Contains(string(row.Payload), gate) {
				film, _ = store.RequestByReference(row.ID)
			}
		}
		return film != nil
	})
	revisions := func() []records.Product {
		products, _ := store.Products(film.ID)
		return products
	}
	// The video is one file rewritten in place: each revision's bytes begin with the last
	// one's, and the file is only ever appended to.
	var takes [][]byte
	take := func(k int) records.Product {
		t.Helper()
		must(t, os.WriteFile(filepath.Join(gate, fmt.Sprintf("go-%d", k)), nil, 0o600))
		var shown records.Product
		landed(t, fmt.Sprintf("film revision %d", k), func() bool {
			if all := revisions(); len(all) == k {
				shown = all[k-1]
				info, err := os.Stat(shown.Path)
				return err == nil && info.Size() == shown.Length
			}
			return false
		})
		data, err := os.ReadFile(shown.Path)
		must(t, err)
		if shown.Path != filepath.Join(directory, fmt.Sprintf("%d-video.mp4", film.Number)) || shown.Rev != uint32(k) || digestOf(data) != shown.Digest {
			t.Fatalf("film revision %d is not its stable file whole: %+v", k, shown)
		}
		if k > 1 && (shown.AppendedFrom == nil || *shown.AppendedFrom != int64(len(takes[k-2])) || !bytes.HasPrefix(data, takes[k-2])) {
			t.Fatalf("film revision %d did not append in place to revision %d: %+v", k, k-1, shown)
		}
		takes = append(takes, data)
		return shown
	}
	take(1)
	take(2)
	film1 := filepath.Join(directory, fmt.Sprintf("%d-video.mp4", film.Number))
	frames := func() int {
		out, err := exec.Command("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
			"-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", film1).CombinedOutput()
		if err != nil {
			t.Fatalf("ffprobe %s: %v %s", film1, err, out)
		}
		count := 0
		fmt.Sscan(strings.TrimSpace(string(out)), &count)
		return count
	}
	if count := frames(); count != 24 {
		t.Fatalf("mid-run the video file should play 2 segments: %d frames", count)
	}
	take(3)
	if err := command.Wait(); err != nil {
		t.Fatalf("the film run exited %v:\n%s", err, filmed.String())
	}
	if count := frames(); count != 36 {
		t.Fatalf("the finished video file should play 3 segments: %d frames", count)
	}
	last = revisions()[2]
	if len(last.Parts) != 4 || last.Parts[0].DurationUs != 0 || last.Parts[1].DurationUs != 500000 {
		t.Fatalf("the last revision is not init plus three half-second fragments: %+v", last.Parts)
	}
	movie, err := os.ReadFile(last.Path)
	if err != nil || digestOf(movie) != last.Digest || !bytes.Equal(movie, takes[2]) {
		t.Fatalf("the final MP4 is not the last revision: %v", err)
	}
	// The outputs folder holds only each run's items, each one stable file at its last
	// revision: no partial files, no superseded revisions, no content-hash names.
	var finals []string
	requests, problem := store.Requests("", "local/output-log-proof", 10)
	fatal(t, problem)
	for _, row := range requests {
		export, problem := store.OutputExportOf(row.ID)
		fatal(t, problem)
		if export != nil && export.Directory == directory {
			finals = append(finals, export.PublishedPaths...)
		}
	}
	entries, err := os.ReadDir(directory)
	must(t, err)
	var present []string
	for _, entry := range entries {
		present = append(present, filepath.Join(directory, entry.Name()))
	}
	slices.Sort(finals)
	slices.Sort(present)
	if !slices.Equal(present, finals) {
		t.Fatalf("the outputs folder is not exactly the runs' final files:\nhas  %v\nwant %v", present, finals)
	}
	t.Logf("the film as the CLI showed it:\n%s", filmed.String())
	if asked := hubCalls()[before:]; len(asked) != 0 {
		t.Fatalf("the Hub was asked %d things while the runs went on:\n%s", len(asked), strings.Join(asked, "\n"))
	}
	// The machine serves the finished film itself, its parts joined, and keeps serving it with its
	// Runtime stopped: a finished run's media never wakes the Runtime or renews idle.
	filmRun := machineServesOutput(t, root, "video", last.Digest, 3)
	machineServesOutputAsleep(t, root, filmRun, "video", last.Digest, 3)
}

func digestOf(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func landed(t *testing.T, what string, done func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Minute); !done(); time.Sleep(50 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

// outputLogProof is a package whose jobs publish as they go: grow (three revisions of an image
// and a list of frames) and film (a 3-segment video joined as it lands), each step waiting
// for its gate file. wheel is the Runtime it runs on.
func outputLogProof(t *testing.T, wheel string) string {
	t.Helper()
	project := t.TempDir()
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(fmt.Sprintf(`[project]
name="output-log-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime[media]>=%s"]
[tool.uv.sources]
cozy-runtime={path=%q}
[project.entry-points."cozy.application"]
default="proof:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["proof.py"]
`, runtimeFixtureVersion(t, wheel), wheel)), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "proof.py"), []byte(`import time
from fractions import Fraction
from pathlib import Path
from typing import Annotated
import msgspec
from cozy_runtime.author import (App, AssetBound, Context, DecodedAudioChunk, DecodedAudioFormat,
    DecodedMediaHeader, DecodedVideoFormat, DecodedVideoFrame, ImageAsset, ImageFrame, Outputs, VideoAsset)
app = App()
class Request(msgspec.Struct):
    gate: str
RATE, PER, TICK = 48000, 2000, Fraction(1, 24)
AUDIO = DecodedAudioFormat(2, RATE, "stereo", ("FL", "FR"), Fraction(1, RATE))
def silence(samples, start):
    return DecodedAudioChunk(2, samples, RATE, "stereo", ("FL", "FR"), (bytes(samples * 4),) * 2, start, Fraction(1, RATE))
def segment(out, k, frames):
    """One H3-shaped segment: fragmented MP4, H.264 and AAC, 12 frames."""
    video = DecodedVideoFormat(64, 48, TICK, Fraction(1), Fraction(24), 1, 1, 1, 1)
    def events():
        yield DecodedMediaHeader(video=video, audio=AUDIO)
        for f in range(frames):
            yield DecodedVideoFrame(64, 48, bytes([(9 * f + 70 * k) % 256, 90, 160]) * (64 * 48), f, 1, TICK, Fraction(1), 1, 1, 1, 1)
            yield silence(PER, f * PER)
    return out.save_video_stream(events()).video
class Film(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
@app.job(emits_media=True)
def film(ctx: Context, payload: Request, out: Outputs) -> Film:
    """Three segments joined as they land: each revision is the video so far."""
    join = out.join_video(AUDIO)
    written = 0
    for k in (1, 2, 3):
        while not (Path(payload.gate) / f"go-{k}").exists():
            ctx.raise_if_cancelled()
            time.sleep(0.02)
        revision = join.append(segment(out, k, 12), lambda frames, at=written: [silence(frames * PER, at)], last=k == 3)
        written += 12 * PER
        out.publish("video", revision, label=f"Video (segments 1-{k} of 3)")
    return Film(join.finish().video)
class Grown(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    frames: list[ImageAsset]
@app.job(emits_media=True)
def grow(ctx: Context, payload: Request, out: Outputs) -> Grown:
    """Three revisions of image and one frame each, each after its gate file appears."""
    frames = []
    for k in (1, 2, 3):
        while not (Path(payload.gate) / f"go-{k}").exists():
            ctx.raise_if_cancelled()
            time.sleep(0.02)
        pixel = bytes([60 * k, 255 - 60 * k, 30])
        image = out.save_image(ImageFrame(8, 8, pixel * 64), format="png")
        out.publish("image", image, label=f"Revision {k} of 3")
        frame = out.save_image(ImageFrame(2, 2, pixel * 4), format="png")
        out.publish("frames", frame, label=f"Frame {k}")
        frames.append(frame)
    return Grown(image=image, frames=frames)
`), 0o600))
	if out, err := exec.Command("/usr/bin/nice", "-n", "19", "uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v %s", err, out)
	}
	return project
}
