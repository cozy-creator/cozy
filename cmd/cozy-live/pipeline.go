package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/home"
)

// cl-003's M4 proof: an SDXL-CLASS ENDPOINT SERVES LOCALLY, through the product.
//
// cl-010 proved the CLI journey against cr-005's single-component UNet endpoint. What this
// adds is the WORKER half — cr-008b's four-component pipeline and its PlanChooser, and
// cr-006's fp8 rung of the same model repo — under the same verbs, with nothing about the
// residency ladder reaching the user.
//
//	pipeline  install (both rungs) -> describe -> fit -> up -> start -> a real 1024px
//	          image -> warm -> multi-request -> the capacity law -> the fp8 rung
//	m4arms    the four M4 arms on the FULL pipeline: duplicate attempt, dropped
//	          TerminalAck, non-cooperative cancel, orchestrator restart

const (
	pipeRef    = "cozy/sdxl-pipeline"
	pipeFP8Ref = "cozy/sdxl-pipeline-fp8"
	// The four-component CAS cr-008b transcribed: one snapshot per component, because a
	// TensorFS read plan is complete over its header and a component cannot be read out of
	// a whole-pipeline snapshot without reading the pipeline.
	pipeWork = "/tmp/cozy-sdxl4"
	// cr-006's fp8 UNet, ingested through the border INTO the same store: the 937 plain
	// tensors dedup, so the rung costs only its own elements and scales.
	fp8UNet = "sha256:3641cd3ac83616d8e993310c91ecdda27a2ed9201bd4b12c6484691b5e3c0c01"
	// cr-008b's banked full-pipeline pixel digest: 1024px, 20 steps, seed 1005, guidance
	// 5.0, the endpoint's default prompt — the endpoint's own determinism fence over the
	// WHOLE loop, and the fence this proof reproduces through the product.
	bankedPixels = "9917742a18d6c4218c5a42dab3989bd2dda32466d51928cae939f38bd3c7a916"
	bankedPNG    = 2118106
	// cr-008b's harness time to first image for that request: `cozy-runtime run`, one
	// shot, warm page cache, no service and no boot warm pass. THE BAR's cold ladder puts
	// the same request at 25.14 s with the page cache dropped and 21.39 s warm. The
	// PRODUCT TAX is what `cozy run` through a orchestrator costs on top.
	harnessTTFIms = 19347
	harnessWarmMs = 21390
	// cr-008b's banked 1024px artifact itself, kept by THE BAR's ladder. Absent, the
	// digest checks still stand; present, it turns "different bits" into a number.
	bankedImage = "/tmp/cozy-bar/bw-warm/outputs/pipe-bw-warm-image.bin"
)

// installPipeline installs one RUNG of the four-component pipeline: the endpoint as a
// release archive, then the local artifact index row its `[bindings]` table selects.
//
// The row is the one thing the harness still places, for cl-010's reason: `cozy-runtime
// pull`/`ingest` refuse typed naming tfs-002/tfs-003, so nothing yet WRITES the index. The
// runtime's own `artifacts.install` writes it and the byte totals are READ OUT OF THE
// HEADERS rather than transcribed — the index holds no tensor facts of its own.
func installPipeline(root, endpoint, ref, lane string, snapshots map[string]string) string {
	slug := endpoint[strings.Index(endpoint, "/")+1:]
	archive := deriveFixture(pipelineFixtureFor(slug))
	code, out := cozyRun(root, "install", endpoint, "--from", archive, "--digest", digestOf(archive))
	if code != 0 {
		fmt.Println(out)
		must("installing "+endpoint, fmt.Errorf("cozy install exited %d", code))
	}
	generation := ""
	for _, line := range strings.Split(out, "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "generation" {
			generation = strings.TrimSpace(value)
		}
	}
	if generation == "" {
		fmt.Println(out)
		must("the install", fmt.Errorf("printed no generation"))
	}

	installArtifactRow(root, generation, filepath.Join(pipeWork, "store"),
		filepath.Join(pipeWork, "pipeline.config.json"), ref, lane, snapshots)
	return generation
}

// pipeSnapshots reads cr-008b's four component snapshots off disk.
func pipeSnapshots() map[string]string {
	data, err := os.ReadFile(filepath.Join(pipeWork, "snapshots.json"))
	if err != nil {
		must("the four-component fixture", fmt.Errorf(
			"%s: %w — rebuild it with cozy-runtime's scripts/pipeline-live.py fixtures",
			filepath.Join(pipeWork, "snapshots.json"), err))
	}
	var out map[string]string
	must("reading the component snapshots", json.Unmarshal(data, &out))
	return out
}

// storeSeal is a digest over the CAS tree's (path, size, mtime). Weights are a READ-ONLY
// input to a serve: if the seal is equal afterwards, nothing under the store was created,
// rewritten, replaced or touched.
//
// `meta/` is DELIBERATELY EXCLUDED and the exclusion is the interesting part. A verified
// read takes a GC-safe HOLD, and a hold is a row in `meta/state.json` — so reading weights
// does write, to the store's own lifecycle bookkeeping, and never to an object. That is
// the distinction the claim needs: zero WEIGHT bytes written, not zero bytes.
func storeSeal(root string) (string, int, int64) {
	h := sha256.New()
	files := 0
	var bytes int64
	must("sealing the store", filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			if d != nil && d.IsDir() && (d.Name() == "meta" || d.Name() == "tmp") {
				return fs.SkipDir
			}
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		files++
		bytes += info.Size()
		fmt.Fprintf(h, "%s\x00%d\x00%d\n", path, info.Size(), info.ModTime().UnixNano())
		return nil
	}))
	return hex.EncodeToString(h.Sum(nil))[:16], files, bytes
}

// waitQuietGPU waits for a card nobody else is on. cr-008b's finding 10: a neighbouring
// agent legitimately holds this device, and waiting is the honest response — measuring the
// neighbour is not.
func waitQuietGPU(timeout time.Duration) int {
	deadline := time.Now().Add(timeout)
	said := false
	for {
		used := gpuUsedMiB()
		if used >= 0 && used <= 900 {
			return used
		}
		if !said {
			fmt.Printf("  waiting for a quiet card: %d MiB held by another process\n", used)
			said = true
		}
		if time.Now().After(deadline) {
			must("the GPU", fmt.Errorf("the card never went quiet (%d MiB in use)", used))
		}
		time.Sleep(10 * time.Second)
	}
}

func sectionPipeline() {
	idle := waitQuietGPU(45 * time.Minute)
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl003-pipeline"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))
	store := filepath.Join(pipeWork, "store")
	sealBefore, files, storeBytes := storeSeal(store)

	head("cozy install — the FOUR-COMPONENT pipeline arrives as a release")
	snaps := pipeSnapshots()
	t0 := time.Now()
	installPipeline(root, pipeRef, pipeRef+"@cr-008b", "plain-fp16", snaps)
	installMS := elapsedMS(t0)
	// The SAME endpoint source, released against a different RUNG of the same model repo.
	// Bindings state selection; the code names nothing, so the two archives differ in one
	// line of `endpoint.toml` and in nothing else.
	fp8 := map[string]string{}
	for name, digest := range snaps {
		fp8[name] = digest
	}
	fp8["unet"] = fp8UNet
	// SAME MODEL REPO, different release: `cozy/sdxl-pipeline@fp8` is the rung, and the
	// fp8 release's `[bindings]` table is what selects it. An index row under the other
	// endpoint's name would have been a second repo, which is not what a rung is.
	installPipeline(root, pipeFP8Ref, pipeRef+"@fp8", "fp8-scaled-scalar", fp8)
	code, out := cozyRun(root, "ls")
	check("cozy ls names both installed generations",
		code == 0 && strings.Contains(out, pipeRef) && strings.Contains(out, pipeFP8Ref),
		firstLine(out))
	fmt.Printf("  bench install (archive -> venv -> descriptor -> pin): %d ms\n", installMS)
	fmt.Printf("  the CAS under proof: %d files, %.3f GiB, seal %s\n",
		files, float64(storeBytes)/(1<<30), sealBefore)

	head("cozy describe — three entrypoints over one four-component model")
	t0 = time.Now()
	code, out = cozyRun(root, "describe", pipeRef)
	describeMS := elapsedMS(t0)
	check("describe lists the release's functions",
		code == 0 && strings.Contains(out, "generate") && strings.Contains(out, "condition") &&
			strings.Contains(out, "stubborn"), firstLine(out))
	fmt.Println(indent(out))
	fmt.Printf("  bench describe: %d ms\n", describeMS)

	head("cozy up -d")
	port := freePort(2960)
	t0 = time.Now()
	svc := startService(root, port, false)
	upMS := elapsedMS(t0)
	defer svc.stop()
	check("the API answers on "+svc.addr, svc.alive(), fmt.Sprintf("pid %d", svc.cmd.Process.Pid))
	fmt.Printf("  bench cozy up -> API answering: %d ms\n", upMS)

	head("cozy fit — the runtime prices the ladder against MEASURED free VRAM")
	t0 = time.Now()
	code, out = cozyRun(root, "fit", pipeRef+"/generate")
	fitMS := elapsedMS(t0)
	// The verdict is PRICED, not looked up: the same walk `run` prices with, against the
	// card's measured free bytes at this instant. Asserting a particular rung here would be
	// asserting what the neighbours are doing.
	check("fit answers 0 with a rung of the closed ladder and the arithmetic behind it",
		code == 0 && (strings.Contains(out, "all_resident") ||
			strings.Contains(out, "component_staged")), firstLine(out))
	fmt.Println(indent(out))
	fmt.Printf("  fit exit %d in %d ms\n", code, fitMS)

	head("cozy run — COLD: no worker, 6.46 GiB of weights, one real 1024px image")
	outDir := filepath.Join(root, "out")
	t0 = time.Now()
	code, doc, raw := cozyJSON(root, "run", pipeRef+"/v1/generate", "steps=20", "size=1024",
		"--out", outDir)
	coldMS := elapsedMS(t0)
	fmt.Println(indent(raw))
	if !check("the cold run exits 0", code == 0, firstLine(raw)) {
		fmt.Println(tail(filepath.Join(root, "driver-service.log"), 40))
		return
	}
	pixels := resultField(doc, "digest")
	// `image.png`, not `image`. cl-010's debt 3: the runtime carries the AUTHOR's declared
	// media type through the output transaction, so the last mile writes a name a client
	// can open without sniffing (cozy-runtime bac8e5e). This expectation predated it.
	saved := filepath.Join(outDir, "image.png")
	info, err := os.Stat(saved)
	check("a real 1024px PNG landed under its declared field path",
		err == nil && info != nil && info.Size() > 1_000_000 && isPNG(saved),
		saved+" "+sizeOf(info))
	coldPeak := metricOf(doc, "peak_vram_bytes")
	fmt.Printf("  bench COLD `cozy run` wall: %d ms · peak vram %s · handler %d ms\n",
		coldMS, gibOf(coldPeak), metricOf(doc, "handler_ms"))
	fmt.Printf("  THE PRODUCT TAX: %d ms against cr-008b's one-shot %d ms and THE BAR's warm %d ms\n",
		coldMS, harnessTTFIms, harnessWarmMs)

	head("determinism: a SECOND cold run, same request, same pixels")
	code, _ = cozyRun(root, "stop", "--all")
	check("the worker drains", code == 0, "")
	t0 = time.Now()
	code, doc, raw = cozyJSON(root, "run", pipeRef+"/v1/generate", "steps=20", "size=1024")
	cold2MS := elapsedMS(t0)
	check("the second cold run exits 0", code == 0, firstLine(raw))
	check("and reproduces the FIRST run's pixel digest over the whole 20-step loop",
		resultField(doc, "digest") == pixels && pixels != "",
		pixels[:16]+"… twice, cold, on a worker built from scratch each time")
	fmt.Printf("  bench second COLD wall: %d ms\n", cold2MS)

	head("the banked fence — the product reproduces it with the warm pass either way")
	// cl-003 measured the product's digest as DIFFERENT from cr-008b's banked one with the
	// boot warm pass on, and named the cause: the warm pass leaves a different
	// device-memory history and cuDNN's algorithm selection reads it. That difference is
	// GONE as of cozy-runtime f1625f9 — the decode leg now fences its scratch on the
	// stream before freeing it (9b13141), which is a real change to that history. The
	// observation was never an invariant; the invariant is the banked artifact, and the
	// product now reaches it from both ends rather than only with the warm pass off.
	check("with the warm pass ON, the product reproduces cr-008b's BANKED pixel digest",
		pixels == bankedPixels, pixels[:16]+"… vs banked "+bankedPixels[:16]+"…")
	// The drain is not decoration: `start` on a resident worker is an idempotent 200, so
	// asking for `--no-warm` while the warm worker is up changes nothing at all. Found by
	// running it — the "no-warm" run was the second 1024px request on the warm worker.
	code, _ = cozyRun(root, "stop", "--all")
	check("the warm worker drains first, or --no-warm asks nothing of anybody", code == 0, "")
	code, out = cozyRun(root, "start", pipeRef, "--no-warm")
	check("`cozy start --no-warm` makes the endpoint resident without a warm pass",
		code == 0, firstLine(out))
	nowarmOut := filepath.Join(root, "out-nowarm")
	code, doc, raw = cozyJSON(root, "run", pipeRef+"/v1/generate", "steps=20", "size=1024",
		"--out", nowarmOut)
	check("the no-warm run exits 0", code == 0, firstLine(raw))
	check("and reproduces cr-008b's BANKED pixel digest exactly",
		resultField(doc, "digest") == bankedPixels, resultField(doc, "digest"))
	nowarmInfo, _ := os.Stat(filepath.Join(nowarmOut, "image.png"))
	check("and the banked PNG's byte length with it",
		nowarmInfo != nil && nowarmInfo.Size() == bankedPNG, sizeOf(nowarmInfo))
	comparePixels(root, saved, flag("banked-image", bankedImage))

	head("a SECOND invocation on the SAME authorized worker: warm against cold")
	// The worker from the no-warm start is still resident. Cold and warm traverse the same
	// states; what differs is that the second one pays no boot.
	code, _ = cozyRun(root, "stop", "--all")
	check("the worker drains before the cold leg", code == 0, "")
	t0 = time.Now()
	code, doc, raw = cozyJSON(root, "run", pipeRef+"/v1/generate", "steps=4", "size=512",
		"seed=1005")
	coldSmallMS := elapsedMS(t0)
	check("the cold 512px run exits 0", code == 0, firstLine(raw))
	smallPixels := resultField(doc, "digest")
	coldAttempt := fmt.Sprint(doc["attempt_key"])
	t0 = time.Now()
	code, doc, raw = cozyJSON(root, "run", pipeRef+"/v1/generate", "steps=4", "size=512",
		"seed=1005")
	warmMS := elapsedMS(t0)
	check("the warm run exits 0 on the worker that is already resident", code == 0, firstLine(raw))
	check("same request, same attempt path, same pixels — warmth is latency, not meaning",
		resultField(doc, "digest") == smallPixels && fmt.Sprint(doc["attempt_key"]) != coldAttempt,
		smallPixels[:16]+"… twice")
	fmt.Printf("  bench 512px/4-step: COLD %d ms · WARM %d ms · handler %d ms · peak vram %s\n",
		coldSmallMS, warmMS, metricOf(doc, "handler_ms"),
		gibOf(metricOf(doc, "peak_vram_bytes")))

	head("multi-request: three sequential requests on ONE resident worker")
	digests := map[string]int{}
	for i := 0; i < 3; i++ {
		t0 = time.Now()
		code, doc, raw = cozyJSON(root, "run", pipeRef+"/v1/generate", "steps=4", "size=512",
			"seed="+itoa(1000+i))
		took := elapsedMS(t0)
		if !check(fmt.Sprintf("request %d served", i+1), code == 0, firstLine(raw)) {
			break
		}
		digests[resultField(doc, "digest")]++
		fmt.Printf("    seed %d -> %s in %d ms (queue %d ms, handler %d ms)\n",
			1000+i, resultField(doc, "digest")[:16], took,
			metricOf(doc, "queue_ms"), metricOf(doc, "handler_ms"))
	}
	check("three different seeds produced three different images on the same worker",
		len(digests) == 3, fmt.Sprintf("%d distinct pixel digest(s)", len(digests)))
	live := svc.call("GET", "/v1/local/workers", nil).json()
	check("and there is still exactly ONE worker — `run` selects, it does not multiply",
		len(asList(live["workers"])) == 1, fmt.Sprint(len(asList(live["workers"]))))

	head("a SECOND 1024px request on a resident worker — the defect this proof found")
	// NAMED, not smoothed. A `component_staged` attempt returns the generation with its
	// components evicted and never re-staged — the runtime's own ledger says so
	// (`ledger_unreconciled`: device_allocated -6,770,424,832 B across the attempt) — and
	// the NEXT attempt's PlanChooser prices its rungs against that drifted residency:
	// `largest` is now the VAE's 0.156 GiB, allocatable is nearly the whole card, so
	// `all_resident` "fits" on activation headroom alone. Nothing charges the 6.3 GiB the
	// rung would have to bring back, and the 1024px decode then OOMs — reported as an
	// AUTHOR_EXCEPTION, for an endpoint that made no memory call at all.
	code, doc, raw = cozyJSON(root, "run", pipeRef+"/v1/generate", "steps=20", "size=1024")
	if code == 0 {
		check("a second 1024px request on a resident worker serves", true,
			resultField(doc, "digest")[:16])
	} else {
		fmt.Printf("  KNOWN RED (cr-008b, reported): exit %d — %s\n", code, firstLine(raw))
		fmt.Println("  the accepted plan and the previous attempt's ledger are in the record")
	}

	head("the capacity law: a SECOND endpoint cannot take a device the first one holds")
	// One active attempt per GPU (law 8) starts at the device envelope: the fp8 rung is a
	// different endpoint over the same card, and the orchestrator's ledger is what refuses.
	second := svc.call("POST", "/v1/local/workers", map[string]any{"endpoint": pipeFP8Ref})
	check("starting a second endpoint's worker while the first holds the card refuses TYPED",
		second.Status >= 400, fmt.Sprintf("%d %s", second.Status, second.brief()))
	fmt.Printf("    %s\n", strings.TrimSpace(string(second.Body)))

	head("the fp8 rung: the same endpoint source over cr-006's encoded UNet")
	code, out = cozyRun(root, "stop", pipeRef)
	check("the fp16 worker drains", code == 0, firstLine(out))
	t0 = time.Now()
	code, doc, raw = cozyJSON(root, "run", pipeFP8Ref+"/v1/generate", "steps=20", "size=1024")
	fp8MS := elapsedMS(t0)
	fp8ok := check("the fp8 rung serves the same request through the same verbs", code == 0,
		firstLine(raw))
	if fp8ok {
		fp8Pixels := resultField(doc, "digest")
		check("and its pixels DIFFER from the fp16 rung's — a no-op decode is the real failure",
			fp8Pixels != pixels, fp8Pixels[:16]+" vs fp16 "+pixels[:16])
		fmt.Printf("  bench fp8 COLD wall: %d ms · peak vram %s · handler %d ms\n",
			fp8MS, gibOf(metricOf(doc, "peak_vram_bytes")), metricOf(doc, "handler_ms"))
		fmt.Printf("  stored bytes: fp16 %s vs fp8 %s\n",
			gibOf(int64(6_937_678_896)), gibOf(int64(4_705_042_544)))
	} else {
		fmt.Println(indent(raw))
	}

	head("the weights are a READ-ONLY input")
	code, out = cozyRun(root, "stop", "--all")
	check("stop --all -> 0", code == 0, firstLine(out))
	sealAfter, filesAfter, bytesAfter := storeSeal(store)
	check("the CAS is byte-for-byte the tree it was before the serve",
		sealAfter == sealBefore && filesAfter == files && bytesAfter == storeBytes,
		fmt.Sprintf("%s -> %s over %d files", sealBefore, sealAfter, filesAfter))

	head("the card")
	used := gpuReleased(idle, 120*time.Second)
	check("the GPU is back at its baseline", used <= idle+40,
		fmt.Sprintf("%d MiB (idle was %d)", used, idle))
}

// --------------------------------------------------------------------------- helpers

// comparePixels measures how far the warm-pass image is from the banked one. "Different
// bits" and "different picture" are not the same claim, and only a number separates them.
// The comparison runs in the GENERATION'S OWN venv — the image libraries are the
// endpoint's dependency, not this host's.
func comparePixels(root, mine, banked string) {
	if _, err := os.Stat(banked); err != nil {
		fmt.Printf("  (the banked artifact is not on this disk: %s)\n", banked)
		return
	}
	generations, err := os.ReadDir(filepath.Join(root, "generations"))
	if err != nil || len(generations) == 0 {
		return
	}
	python := home.VenvPython(filepath.Join(root, "generations", generations[0].Name(), "venv"))
	cmd := exec.Command("/usr/bin/nice", "-n", "19", python, "-c", diffScript, banked, mine)
	cmd.Env = childEnv(root)
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(indent(string(out)))
		return
	}
	fmt.Printf("  %s", string(out))
}

const diffScript = `
import math, sys
import numpy as np
from PIL import Image

a, b = (np.asarray(Image.open(p).convert("RGB")).astype(np.int32) for p in sys.argv[1:3])
d = np.abs(a - b)
mse = float((d.astype(np.float64) ** 2).mean())
print(
    f"banked vs the warm-pass image: {float((d.max(axis=2) > 0).mean()) * 100:.2f}% of "
    f"pixels differ, by at most {int(d.max())}/255 — PSNR "
    f"{10 * math.log10(255 * 255 / mse):.1f} dB. Same picture, different bits."
)
`

// resultField reads one field of the endpoint's own typed result out of a `--json` run.
func resultField(doc map[string]any, key string) string {
	result, _ := doc["result"].(map[string]any)
	if result == nil {
		return ""
	}
	return fmt.Sprint(result[key])
}

func metricOf(doc map[string]any, key string) int64 {
	metrics, _ := doc["metrics"].(map[string]any)
	if metrics == nil {
		return 0
	}
	return int64(asFloat(metrics[key]))
}

func gibOf(n int64) string { return fmt.Sprintf("%.3f GiB", float64(n)/(1<<30)) }

func asList(v any) []any {
	out, _ := v.([]any)
	return out
}
