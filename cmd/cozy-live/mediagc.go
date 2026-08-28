package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// --------------------------------------------------------------------------- mediagc
//
// cl-033: LOCAL OUTPUT RETENTION AND RECLAMATION, end to end, with no GPU anywhere in it.
//
// The subject is what a rental leaves behind. Outputs are mirrored onto this host before
// the terminal is acked and are deliberately kept when the pod dies, so this root is the
// one plane that grows forever. The arms are about the two halves of bounding it without
// ever being the reason someone lost work:
//
//	the horizon protects  a bare `cozy gc` plans ZERO media, because nothing is old enough
//	the plan is a read    `--keep-media 0` lists every output and removes NOT ONE byte
//	`--yes` performs      the bytes go, the rows stay, and the freed total is reported
//	the row is a tombstone `GET /v1/media/{id}` answers 410 media_reclaimed, never 404
//	it converges          a second `--yes` finds nothing and frees nothing
//
// The producer is `fakeworker --arm output`: a real second process speaking the committed
// protocol, writing a real PNG where the grant said. Nothing here is a mock and nothing
// here is a model.
func sectionMediaGC() {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "cl032-mediagc"))
	must("clearing the root", os.RemoveAll(root))
	must("creating the root", os.MkdirAll(root, 0o755))

	head("the empty state: a root that has published nothing")
	code, out := cozyRun(root, "media", "ls")
	check("`cozy media ls` answers the empty state, exit 0",
		code == 0 && strings.Contains(out, "0 retained outputs"), firstLine(out))
	code, out = cozyRun(root, "gc")
	check("`cozy gc` says so too, and NAMES the horizon it decided under",
		code == 0 && strings.Contains(out, "30d horizon"), lastLines(out, 2))

	head("two real outputs, from a real peer over the committed protocol")
	lv := hostCoordinator("cl032-mediagc", false)
	spec := fakeSpec("mediagc", "7", "--arm", "output", "--cozy-home", lv.root)
	instance, _, e := lv.c.EnsureWorker(spec)
	if !check("the peer registers", e == nil, briefly(e)) {
		return
	}
	planID := planIDOf(spec, "fake")
	if e := lv.c.EnsurePlacementReady(instance, planID); e != nil {
		check("the peer is ready", false, e.Message)
		return
	}
	var mediaIDs, paths []string
	for i := 0; i < 2; i++ {
		sub := submissionKey(planID, payload(map[string]any{"n": i}), fmt.Sprintf("cl032-%d", i))
		sub.Endpoint, sub.Entrypoint = "fake/mediagc", "fake"
		requestID, attempt, e := lv.c.Submit(sub)
		if !check(fmt.Sprintf("attempt %d dispatched", i), e == nil, briefly(e)) {
			return
		}
		if _, e := lv.c.Await(requestID, attempt, 60*time.Second); e != nil {
			check("the attempt closes", false, briefly(e))
			return
		}
		outs, _ := lv.store.VisibleOutputs(requestID)
		if len(outs) != 1 {
			check("exactly one output is visible", false, fmt.Sprintf("%d", len(outs)))
			return
		}
		mediaIDs = append(mediaIDs, outs[0].MediaID)
		paths = append(paths, outs[0].Path)
	}
	onDisk := 0
	for _, p := range paths {
		if info, err := os.Stat(p); err == nil && info.Size() > 0 {
			onDisk++
		}
	}
	check("both outputs are mirrored under the owner's own output namespace", onDisk == 2,
		fmt.Sprintf("%d of 2 on disk, e.g. %s", onDisk, trimPath(paths[0])))
	for _, w := range mustLiveWorkers(lv) {
		lv.c.ShutdownWorker(w, 10*time.Second)
	}
	lv.close()

	head("the LocalService serves them by opaque id, with the pod long gone")
	port := freePort(2833)
	svc := startService(root, port, false)
	defer svc.stop()
	first := svc.call("GET", "/v1/media/"+mediaIDs[0], nil)
	check("GET /v1/media/{id} serves the bytes", first.Status == http.StatusOK &&
		len(first.Body) > 8 && string(first.Body[1:4]) == "PNG",
		fmt.Sprintf("%s, %d B", mediaIDs[0], len(first.Body)))

	head("THE HORIZON: a bare `cozy gc` protects everything inside it")
	code, out = cozyRun(root, "gc")
	check("the default 30d horizon plans ZERO media and says what it kept",
		code == 0 && strings.Contains(out, "media: 0 output(s)") &&
			strings.Contains(out, "retained: 2 output(s)"), lastLines(out, 3))
	check("and nothing was removed", allExist(paths), "2 files")

	head("`cozy media ls`: what this host is storing")
	code, out = cozyRun(root, "media", "ls")
	check("every retained output is listed, on disk, with its age",
		code == 0 && strings.Count(out, "on disk") >= 2, lastLines(out, 6))
	code, out = cozyRun(root, "media", "ls", "--keep-media", "0")
	check("under a zero horizon the SAME rows are marked reclaimable, still untouched",
		code == 0 && strings.Count(out, "reclaimable") >= 2 && allExist(paths),
		lastLines(out, 4))

	head("THE PLAN/PERFORM SPLIT: a bare invocation removes nothing")
	code, out = cozyRun(root, "gc", "--keep-media", "0")
	check("`cozy gc --keep-media 0` PLANS both outputs, exit 0",
		code == 0 && strings.Contains(out, "media: 2 output(s)") &&
			strings.Contains(out, "this is the plan; nothing was removed"), lastLines(out, 3))
	check("and every byte is still on disk", allExist(paths), "2 files")

	head("`--yes` performs: the bytes go, the rows stay")
	code, out = cozyRun(root, "gc", "--keep-media", "0", "--yes")
	check("it reports what it freed", code == 0 && strings.Contains(out, "media: 2 output(s)"),
		lastLines(out, 3))
	check("the mirrored bytes are gone from disk", noneExist(paths), "0 of 2 files")
	code, out = cozyRun(root, "media", "ls")
	check("the ROWS survive their bytes, marked reclaimed",
		code == 0 && strings.Count(out, "reclaimed") >= 2, lastLines(out, 3))

	head("the tombstone: a reclaimed id is 410, never 404")
	gone := svc.call("GET", "/v1/media/"+mediaIDs[0], nil)
	check("GET /v1/media/{id} answers 410 media_reclaimed",
		gone.Status == http.StatusGone && gone.code() == "media_reclaimed", gone.brief())
	never := svc.call("GET", "/v1/media/med-000000000000000000000000", nil)
	check("an id that NEVER existed is still a 404 — the two are different facts",
		never.Status == http.StatusNotFound && never.code() == "not_found", never.brief())

	head("convergence: a second pass finds nothing")
	code, out = cozyRun(root, "gc", "--keep-media", "0", "--yes")
	check("re-running the reclamation frees nothing and refuses nothing",
		code == 0 && strings.Contains(out, "media: 0 output(s)"), lastLines(out, 2))
	code, out = cozyRun(root, "gc")
	check("and the bare plan is back to the empty answer", code == 0 &&
		strings.Contains(out, "nothing is unreferenced"), lastLines(out, 2))
}

func mustLiveWorkers(lv *live) []string {
	rows, _ := lv.store.LiveWorkers()
	out := make([]string, 0, len(rows))
	for _, w := range rows {
		out = append(out, w.InstanceID)
	}
	return out
}

func allExist(paths []string) bool {
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			return false
		}
	}
	return true
}

func noneExist(paths []string) bool {
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return false
		}
	}
	return true
}

// lastLines is the evidence line: the aggregates and notes a `cozy` verb ends with.
func lastLines(s string, n int) string {
	lines := []string{}
	for _, line := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		if strings.TrimSpace(line) != "" {
			lines = append(lines, strings.TrimSpace(line))
		}
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, " | ")
}
