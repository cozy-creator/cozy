package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/api"
	"github.com/cozy-creator/cozy-creator/internal/manifest"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

// cl-006's live sections. Everything here drives a SEPARATE, REAL `cozy up` process over
// real HTTP — the driver is a client and nothing more, which is what makes the
// orchestrator-kill arm possible at all.

// --------------------------------------------------------------------------- apiarms
//
// The security refusal matrix. No GPU: every arm here is about the door, not the work.

func sectionAPIArms() {
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "api-arms"))
	port := freePort(2790)
	svc := startService(root, port, true)
	defer svc.stop()

	head("the door: loopback, Host, Origin, bearer")

	// A credential is required, and the refusal is the TYPED envelope.
	no := svc.call("GET", "/v1/capabilities", nil, "Authorization", "")
	check("a request with NO credential is refused", no.Status == http.StatusUnauthorized &&
		no.code() == "unauthenticated", no.brief())
	check("and it names the scheme it wants", no.Header.Get("WWW-Authenticate") != "",
		no.Header.Get("WWW-Authenticate"))

	wrong := svc.call("GET", "/v1/capabilities", nil, "Authorization",
		"Bearer 0000000000000000000000000000000000000000000000000000000000000000")
	check("a WRONG credential is refused the same way", wrong.Status == http.StatusUnauthorized &&
		wrong.code() == "unauthenticated", wrong.brief())

	// The browser credential is a DIFFERENT value from the CLI's. Proven negatively:
	// the CLI credential works and a value derived from it does not.
	tweaked := "Bearer " + svc.token[:len(svc.token)-1] + "0"
	if strings.HasSuffix(svc.token, "0") {
		tweaked = "Bearer " + svc.token[:len(svc.token)-1] + "1"
	}
	near := svc.call("GET", "/v1/capabilities", nil, "Authorization", tweaked)
	check("a credential differing in ONE character is refused",
		near.Status == http.StatusUnauthorized, near.brief())

	// THE DNS-REBINDING KILL SWITCH. The request reaches 127.0.0.1 — because that is
	// what the attacker's DNS answered — and carries the attacker's hostname in Host.
	rebind := svc.call("GET", "/v1/capabilities", nil, "Host", "cozy.attacker.example")
	check("a REBINDING-style foreign Host is refused (the Ollama CVE class)",
		rebind.Status == http.StatusForbidden && rebind.code() == "host_not_allowed",
		rebind.brief())
	rebindPost := svc.call("POST", "/v1/requests", map[string]any{"endpoint": "x/y", "function": "f"},
		"Host", "cozy.attacker.example", "Idempotency-Key", "arm-rebind")
	check("and a rebinding MUTATION is refused before anything is recorded",
		rebindPost.Status == http.StatusForbidden && rebindPost.code() == "host_not_allowed",
		rebindPost.brief())

	// Origin, on a mutation and on a stream open. A cross-site fetch, a cross-site form
	// POST and a cross-site EventSource all send Origin; a same-origin page does not
	// send one on a GET, and a CLI never sends one at all.
	xsMutate := svc.call("POST", "/v1/requests", map[string]any{"endpoint": "x/y", "function": "f"},
		"Origin", "https://evil.example", "Idempotency-Key", "arm-origin")
	check("a CROSS-ORIGIN mutation is refused", xsMutate.Status == http.StatusForbidden &&
		xsMutate.code() == "origin_not_allowed", xsMutate.brief())

	xsStream := svc.openSSE("/v1/events?from=now", 5*time.Second, "Origin", "https://evil.example")
	check("a CROSS-ORIGIN stream open is refused", xsStream.closeErr != nil &&
		strings.Contains(xsStream.closeErr.Error(), "origin_not_allowed"),
		fmt.Sprint(xsStream.closeErr))

	nullOrigin := svc.call("POST", "/v1/requests", map[string]any{"endpoint": "x/y", "function": "f"},
		"Origin", "null", "Idempotency-Key", "arm-null")
	check("a sandboxed-iframe `Origin: null` mutation is refused",
		nullOrigin.Status == http.StatusForbidden, nullOrigin.brief())

	sameOrigin := svc.call("GET", "/v1/capabilities", nil,
		"Origin", fmt.Sprintf("http://127.0.0.1:%d", port))
	check("a SAME-ORIGIN request passes", sameOrigin.Status == http.StatusOK, sameOrigin.brief())

	head("no CORS, no cookies, no sniffing")
	caps := svc.call("GET", "/v1/capabilities", nil)
	corsHeader := "Access-Control-Allow-Origin" //cozy:allow the VERIFIER reads this header to prove its ABSENCE; the product sets none
	check("NO "+corsHeader+" is ever sent", caps.Header.Get(corsHeader) == "", "absent")
	check("NO Set-Cookie is ever sent", caps.Header.Get("Set-Cookie") == "", "absent")
	check("nosniff on every response", caps.Header.Get("X-Content-Type-Options") == "nosniff",
		caps.Header.Get("X-Content-Type-Options"))
	check("a strict CSP on every response",
		strings.Contains(caps.Header.Get("Content-Security-Policy"), "default-src 'none'"),
		caps.Header.Get("Content-Security-Policy"))
	check("no referrer leaves this origin", caps.Header.Get("Referrer-Policy") == "no-referrer",
		caps.Header.Get("Referrer-Policy"))

	// A cookie the server never reads: presenting one instead of a bearer authenticates
	// nothing, which is the property that makes cross-site ambient authority impossible.
	cookied := svc.call("GET", "/v1/capabilities", nil, "Authorization", "",
		"Cookie", "cozy="+svc.token)
	check("a CREDENTIAL PRESENTED AS A COOKIE authenticates nothing",
		cookied.Status == http.StatusUnauthorized, cookied.brief())

	head("the media plane: opaque ids only")
	for _, attempt := range []string{
		"/v1/media/../../../../etc/passwd",
		"/v1/media/..%2f..%2f..%2fetc%2fpasswd",
		"/v1/media/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
		"/v1/media/" + strings.ReplaceAll(root, "/", "%2F") + "%2Frecords.db",
		"/v1/media/med-000000000000000000000000",
	} {
		res := svc.call("GET", attempt, nil)
		check("a client-supplied PATH is not media: "+trimPath(attempt),
			res.Status == http.StatusNotFound || res.Status == http.StatusMovedPermanently,
			res.brief())
	}
	// And the structural claim behind those five: the route table has ONE media route
	// and its only parameter is an id.
	pathRoutes := 0
	for _, r := range apiRouteTable() {
		if strings.Contains(r, "{path") || strings.Contains(r, "filename") ||
			strings.Contains(r, "{file") {
			pathRoutes++
		}
	}
	check("NO route in the whole surface takes a path parameter", pathRoutes == 0,
		fmt.Sprintf("%d of %d routes", pathRoutes, len(apiRouteTable())))

	head("the envelope, everywhere")
	unknown := svc.call("GET", "/v1/does-not-exist", nil)
	check("an unknown route answers with the TYPED envelope, not a bare 404 string",
		unknown.Status == http.StatusNotFound && unknown.code() == "unknown_route", unknown.brief())
	badMethod := svc.call("DELETE", "/v1/requests", nil)
	check("a method mismatch answers with the envelope too",
		badMethod.code() != "", badMethod.brief())
	noKey := svc.call("POST", "/v1/requests", map[string]any{"endpoint": "x/y", "function": "f"},
		"Idempotency-Key", "")
	check("a submission with NO Idempotency-Key is refused before anything is recorded",
		noKey.Status == http.StatusBadRequest && noKey.code() == "idempotency_key_required",
		noKey.brief())

	head("the bind: loopback only, both families, and no widening flag")
	v6 := svc.callAddr(fmt.Sprintf("[::1]:%d", port), "GET", "/v1/capabilities",
		fmt.Sprintf("[::1]:%d", port))
	check("the IPv6 loopback listener answers", v6.Status == http.StatusOK, v6.brief())
	v6bad := svc.callAddr(fmt.Sprintf("[::1]:%d", port), "GET", "/v1/capabilities", "evil.example")
	check("and it enforces the SAME Host allowlist", v6bad.Status == http.StatusForbidden, v6bad.brief())

	lan := lanAddress()
	if lan != "" {
		res := svc.callAddr(fmt.Sprintf("%s:%d", lan, port), "GET", "/healthz", "")
		check("the LAN address is NOT bound — nothing answers there", res.Status == 0,
			fmt.Sprintf("%s:%d — %s", lan, port, strings.TrimSpace(string(res.Body))))
	} else {
		check("the LAN address is NOT bound", true, "no non-loopback interface on this host")
	}
	// The claim behind it: there is no flag that widens the bind. The manifest is the
	// surface, so the absence is checkable rather than asserted.
	check("NO flag on `cozy up` widens the bind", !manifestHasBindFlag(),
		"--port only; the LAN door is deferred behind TLS and its own threat review")

	head("the stub page")
	stub := svc.call("GET", "/", nil, "Authorization", "")
	check("the stub page is served without a credential (it holds none)",
		stub.Status == http.StatusOK, fmt.Sprintf("%d, %d B", stub.Status, len(stub.Body)))
	check("and carries NO token in its bytes", !strings.Contains(string(stub.Body), svc.token),
		"the credential rides the URL FRAGMENT, which never reaches this server")
	check("its CSP forbids inline script", strings.Contains(
		stub.Header.Get("Content-Security-Policy"), "script-src 'self'"),
		stub.Header.Get("Content-Security-Policy"))

	head("the credential file")
	info, err := os.Stat(filepath.Join(root, "client.cred"))
	// Unix mode bits are a fiction on Windows (Go reports 0666 for every file); the
	// boundary there is the user profile's ACL, so the bit assertion is Unix-only.
	modeOK := err == nil && (runtime.GOOS == "windows" || info.Mode().Perm() == 0o600)
	check("the CLI credential is handed over through a 0600 file, never argv",
		modeOK, fmt.Sprintf("%v", info.Mode().Perm()))
	logBody, _ := os.ReadFile(filepath.Join(root, "driver-service.log"))
	check("and the service's own log never contains it",
		!strings.Contains(string(logBody), svc.token), fmt.Sprintf("%d B of log", len(logBody)))
	errBody := svc.call("GET", "/v1/requests/req-nope", nil)
	check("nor does any rendered error", !strings.Contains(string(errBody.Body), svc.token),
		errBody.brief())

	head("the browser credential must never name host paths (cl-026: a job's trees)")
	// The browser half of the per-launch pair is deliberately unreachable from outside a
	// real service (it exists only in the process and the --open URL's fragment), so this
	// arm hosts the REAL api.Server in-process with a credential pair the driver minted
	// itself — the same Server, the same guard chain, over real loopback HTTP.
	lv := hostCoordinator("api-arms-paths", true)
	defer lv.close()
	rawBrowser, rawCLI := "arm-browser-"+randomHex(24), "arm-cli-"+randomHex(24)
	ln, lerr := net.Listen("tcp", "127.0.0.1:0") //cozy:allow the DRIVER hosts the real api.Server to arm the browser-credential path gate; the product binds through internal/api
	must("binding the arm server", lerr)
	defer ln.Close()
	armServer := api.New(api.Options{
		Orchestrator: lv.c, Cfg: lv.cfg, Addr: ln.Addr().String(), Bound: []string{"ipv4"},
		Creds: api.Credentials{Browser: secret.New(rawBrowser), CLI: secret.New(rawCLI)},
	})
	armHandler, he := armServer.Handler()
	must("the arm server's handler", errOf(he))
	go func() { _ = http.Serve(ln, armHandler) }()

	postJob := func(token, idem string, body map[string]any) (int, string) {
		data, err := json.Marshal(body)
		must("rendering the job body", err)
		req, err := http.NewRequest("POST", "http://"+ln.Addr().String()+"/v1/local/jobs",
			strings.NewReader(string(data)))
		must("building the job request", err)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Idempotency-Key", idem)
		res, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
		if err != nil {
			return 0, err.Error()
		}
		defer res.Body.Close()
		raw, _ := io.ReadAll(res.Body)
		return res.StatusCode, string(raw)
	}
	trees := map[string]any{"endpoint": "fake/paths", "function": "census",
		"trees": []string{"data=" + root}}
	status, body := postJob(rawBrowser, "arm-trees-browser", trees)
	check("a BROWSER-credential job naming trees is refused (403 cli_credential_required)",
		status == http.StatusForbidden && strings.Contains(body, "cli_credential_required"),
		fmt.Sprintf("%d %s", status, brieflyBody(body)))
	status, body = postJob(rawCLI, "arm-trees-cli", trees)
	check("the SAME submission under the CLI credential passes the path gate",
		status != http.StatusForbidden && !strings.Contains(body, "cli_credential_required"),
		fmt.Sprintf("%d %s", status, brieflyBody(body)))
	status, body = postJob(rawBrowser, "arm-notrees-browser",
		map[string]any{"endpoint": "fake/paths", "function": "census"})
	check("a browser submission WITHOUT trees is not refused by the path gate",
		!strings.Contains(body, "cli_credential_required"),
		fmt.Sprintf("%d %s", status, brieflyBody(body)))
}

func brieflyBody(body string) string {
	body = strings.TrimSpace(body)
	if len(body) > 120 {
		return body[:120] + "…"
	}
	return body
}

func trimPath(p string) string {
	if len(p) > 46 {
		return p[:46] + "…"
	}
	return p
}

// --------------------------------------------------------------------------- api
//
// The whole contract core against the REAL supervisor, the REAL executor and cr-005's
// REAL SDXL UNet: submit -> SSE progress -> terminal -> media by opaque id -> triage.

func sectionAPI() {
	idle := requireFreeGPU()
	root := flag("home", filepath.Join(os.TempDir(), "cozy-live", "api"))
	must("clearing the service root", os.RemoveAll(root))
	must("creating the service root", os.MkdirAll(root, 0o755))

	installEndpoint(root)

	port := freePort(2801)
	svc := startService(root, port, false)
	defer svc.stop()

	head("the LOCAL module: an endpoint this host can serve, and a worker to serve it")
	eps := svc.call("GET", "/v1/local/endpoints", nil)
	check("the endpoint listing names the dev tree and its functions",
		strings.Contains(string(eps.Body), "cozy/sdxl-unet") &&
			strings.Contains(string(eps.Body), "denoise"), eps.brief())

	bootStart := time.Now()
	start := svc.call("POST", "/v1/local/workers", map[string]any{"endpoint": "cozy/sdxl-unet"})
	check("POST /v1/local/workers spawns the REAL cozy-runtime supervisor",
		start.Status == http.StatusAccepted, start.brief())
	instance, _ := start.json()["instance_id"].(string)

	if !waitReady(svc, 300*time.Second) {
		fmt.Println(tail(filepath.Join(root, "workers", instance, "worker.log"), 30))
		check("the worker reported READY", false, "see the log above")
		return
	}
	boot := time.Since(bootStart)
	check("the worker's placement reports DISPATCHABLE through the API", true, ms(boot))

	// IDEMPOTENT, and it SAYS SO rather than leaving a client to infer it (#484): the
	// route answers `change: none` for a worker already hosting this placement, which is a
	// different answer from `placement_added` on a live worker that gained one — and a
	// `resident: true` boolean was true of both.
	again := svc.call("POST", "/v1/local/workers", map[string]any{"endpoint": "cozy/sdxl-unet"})
	check("starting it again is IDEMPOTENT and names the change", again.Status == http.StatusOK &&
		again.json()["change"] == "none", again.brief())

	head("submit -> SSE -> terminal -> media, as an HTTP client")
	body := map[string]any{"steps": 4, "latent": 64, "seed": 1005}
	submitAt := time.Now()
	// The stream is opened BEFORE the terminal can land, from the head of the log, so
	// what it observes is live rather than replayed.
	streamed := make(chan *sseStream, 1)
	res := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": "cozy/sdxl-unet", "function": "denoise", "input": body,
	}, "Idempotency-Key", "api-1")
	acceptedHTTP := time.Since(submitAt)
	check("POST /v1/requests answers 202 with the handle",
		res.Status == http.StatusAccepted, res.brief())
	handle := res.json()
	requestID, _ := handle["request_id"].(string)
	check("the handle carries the contract's own url set",
		handle["status_url"] != nil && handle["cancel_url"] != nil && handle["events_url"] != nil,
		fmt.Sprintf("%v", handle["status_url"]))
	check("and says this call STARTED the work", handle["idempotent_replay"] == false,
		fmt.Sprintf("attempt %v, status %v", handle["attempt"], handle["status"]))

	go func() {
		streamed <- svc.openSSE("/v1/requests/"+requestID+"/events", 180*time.Second)
	}()

	head("idempotency, from the client's seat")
	replay := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": "cozy/sdxl-unet", "function": "denoise", "input": body,
	}, "Idempotency-Key", "api-1")
	check("the SAME key with the SAME body answers 200 with the SAME request",
		replay.Status == http.StatusOK && replay.json()["request_id"] == requestID,
		replay.brief())
	check("and says so explicitly", replay.json()["idempotent_replay"] == true, "idempotent_replay")

	changed := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": "cozy/sdxl-unet", "function": "denoise", "input": map[string]any{"steps": 2},
	}, "Idempotency-Key", "api-1")
	check("the same key with a DIFFERENT body is a typed conflict",
		changed.Status == http.StatusConflict, changed.brief())
	crossFn := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": "cozy/sdxl-unet", "function": "pair", "input": body,
	}, "Idempotency-Key", "api-1")
	check("the same key naming a DIFFERENT FUNCTION conflicts too (the digest covers the whole submission)",
		crossFn.Status == http.StatusConflict, crossFn.brief())

	stream := <-streamed
	total := time.Since(submitAt)
	check("the stream TERMINAL-STOPPED", stream.stopped, strings.Join(stream.types(), " · "))
	check("it announced a reconnect hint", stream.retryMS > 0, fmt.Sprintf("retry %d ms", stream.retryMS))
	check("durable lifecycle arrived in order",
		orderedPrefix(stream.types(), "request.dispatched", "request.accepted", "request.completed"),
		strings.Join(stream.types(), " · "))
	progress := stream.count("request.progress")
	check("LIVE progress frames arrived on the same stream", progress > 0,
		fmt.Sprintf("%d progress · %d log · %d stage · %d metric", progress,
			stream.count("request.log"), stream.count("request.stage"),
			stream.count("request.metric")))
	for _, e := range stream.events {
		if e.EventID == 0 && e.Payload["live"] != true {
			check("every event is either durable (event_id > 0) or marked live", false, e.Type)
			break
		}
	}
	check("every event is either durable (event_id > 0) or marked live", true,
		fmt.Sprintf("%d events, cursor %d", len(stream.events), stream.cursor))

	// THE ANNOUNCEMENT LAW: bytes are never pushed onto the stream.
	raw, _ := json.Marshal(stream.events)
	check("NO frame carried image bytes — media is ANNOUNCED by opaque id",
		!strings.Contains(string(raw), "iVBORw0KGgo"), fmt.Sprintf("%d B of stream", len(raw)))
	terminal := stream.find("request.completed")
	check("the terminal event carries media ids", terminal != nil &&
		strings.Contains(fmt.Sprint(terminal.Payload["outputs"]), "med-"),
		fmt.Sprint(terminal.Payload["status"]))

	head("the lifecycle document and the media plane")
	life := svc.call("GET", "/v1/requests/"+requestID, nil)
	check("GET /v1/requests/{id} is `completed`", life.json()["status"] == "completed", life.brief())
	outputs, _ := life.json()["outputs"].([]any)
	check("exactly one output is visible", len(outputs) == 1, fmt.Sprintf("%d", len(outputs)))
	if len(outputs) != 1 {
		return
	}
	out := outputs[0].(map[string]any)
	mediaID, _ := out["media_id"].(string)
	check("its id is OPAQUE and its output_id is the RESULT FIELD PATH",
		strings.HasPrefix(mediaID, "med-") && out["output_id"] == "image",
		fmt.Sprintf("%s -> %v", mediaID, out["output_id"]))
	check("the typed result rode the lifecycle document, decoded",
		life.json()["result"] != nil, fmt.Sprint(life.json()["result"]))

	fetchAt := time.Now()
	media := svc.call("GET", "/v1/media/"+mediaID, nil)
	fetch := time.Since(fetchAt)
	check("GET /v1/media/{id} serves the bytes", media.Status == http.StatusOK &&
		len(media.Body) > 8 && string(media.Body[1:4]) == "PNG",
		fmt.Sprintf("%d B PNG in %s", len(media.Body), ms(fetch)))
	check("with the digest the manifest declared, so a client verifies rather than trusts",
		media.Header.Get("X-Cozy-Digest") == out["digest"], media.Header.Get("X-Cozy-Digest"))
	check("nosniff and a filename derived from the OPAQUE ID",
		media.Header.Get("X-Content-Type-Options") == "nosniff" &&
			strings.Contains(media.Header.Get("Content-Disposition"), mediaID),
		media.Header.Get("Content-Disposition"))

	headRes := svc.call("HEAD", "/v1/media/"+mediaID, nil)
	check("HEAD answers the size without the bytes", headRes.Status == http.StatusOK &&
		len(headRes.Body) == 0, headRes.Header.Get("Content-Length"))
	ranged := svc.call("GET", "/v1/media/"+mediaID, nil, "Range", "bytes=0-99")
	check("ONE Range is honoured", ranged.Status == http.StatusPartialContent &&
		len(ranged.Body) == 100, ranged.Header.Get("Content-Range"))
	multi := svc.call("GET", "/v1/media/"+mediaID, nil, "Range", "bytes=0-9,20-29")
	check("a MULTI-range request is refused, never partly honoured",
		multi.Status == http.StatusRequestedRangeNotSatisfiable, multi.brief())

	head("triage: the retained bundle, by opaque attempt key")
	triageRef, _ := life.json()["triage"].(map[string]any)
	if triageRef == nil {
		check("the terminal named a triage bundle", false, "no triage ref on the lifecycle document")
	} else {
		key, _ := triageRef["attempt_key"].(string)
		check("the lifecycle document carries the OPAQUE attempt key", strings.HasPrefix(key, "att-"),
			fmt.Sprintf("%s -> %v (%v B, kept %v)", key, triageRef["subject_id"],
				triageRef["length"], triageRef["kept"]))
		bundle := svc.call("GET", "/v1/local/attempts/"+key+"/triage", nil)
		check("the bundle reads back VERIFIED against what its terminal declared",
			bundle.Status == http.StatusOK && bundle.json()["verified"] == true, bundle.brief())
		if lines, ok := bundle.json()["explain"].([]any); ok && len(lines) > 0 {
			check("and explains itself", true, fmt.Sprint(lines[0]))
			for _, l := range lines {
				fmt.Printf("    | %s\n", l)
			}
		}
		// PERSISTENCE IS THIS HOST'S: the bytes live under the local root, not in the
		// worker's, so they survive the worker root going away.
		subject, _ := triageRef["subject_id"].(string)
		_, err := os.Stat(filepath.Join(root, "triage", subject+".json"))
		check("the bundle was COPIED out of the worker root into the local store", err == nil,
			filepath.Join(root, "triage", subject+".json"))

		// The corruption arm: edit the retained bytes and the read REFUSES rather than
		// rendering a plausible story.
		kept := filepath.Join(root, "triage", subject+".json")
		original, _ := os.ReadFile(kept)
		must("planting a corruption", os.WriteFile(kept, append(original, ' '), 0o600))
		corrupt := svc.call("GET", "/v1/local/attempts/"+key+"/triage", nil)
		check("a bundle edited on disk is a DETECTED corruption, not a story",
			corrupt.Status == http.StatusConflict && corrupt.code() == "bundle_corrupt",
			corrupt.brief())
		must("restoring the bundle", os.WriteFile(kept, original, 0o600))
		restored := svc.call("GET", "/v1/local/attempts/"+key+"/triage", nil)
		check("and it reads green again once restored", restored.Status == http.StatusOK,
			restored.brief())
	}

	head("benchmarks — RTX 4070 Laptop, nice -n 19, shared box")
	benchAPI(svc, boot, acceptedHTTP, total, fetch, len(media.Body), mediaID)

	head("two outputs, one nested: field paths, not positions")
	pairRes := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": "cozy/sdxl-unet", "function": "pair", "input": body,
	}, "Idempotency-Key", "api-pair")
	pairID, _ := pairRes.json()["request_id"].(string)
	pairStream := svc.openSSE("/v1/requests/"+pairID+"/events", 180*time.Second)
	check("the pair attempt settled", pairStream.stopped &&
		pairStream.find("request.completed") != nil, strings.Join(pairStream.types(), " · "))
	pairLife := svc.call("GET", "/v1/requests/"+pairID, nil)
	pairOuts, _ := pairLife.json()["outputs"].([]any)
	names := []string{}
	for _, o := range pairOuts {
		names = append(names, fmt.Sprint(o.(map[string]any)["output_id"]))
	}
	check("both outputs are visible under their FIELD PATHS", len(pairOuts) == 2 &&
		strings.Join(names, ",") == "detail.thumb,preview", strings.Join(names, ", "))
	distinct := map[string]bool{}
	for _, o := range pairOuts {
		distinct[fmt.Sprint(o.(map[string]any)["media_id"])] = true
	}
	check("each has its OWN opaque id", len(distinct) == 2, fmt.Sprintf("%d distinct ids", len(distinct)))

	head("cancel, mid-attempt")
	// `denoise` is the COOPERATIVE handler: it calls ctx.raise_if_cancelled() every step,
	// so a cancel lands as a real CANCELED terminal rather than as a dead executor. 64
	// steps at latent 80 is the longest run its own schema admits (steps <= 64), which is
	// several seconds of real GPU work — a wide enough window to cancel INTO.
	cancelRes := svc.call("POST", "/v1/requests", map[string]any{
		"endpoint": "cozy/sdxl-unet", "function": "denoise",
		"input": map[string]any{"steps": 64, "latent": 80, "seed": 7},
	}, "Idempotency-Key", "api-cancel")
	cancelID, _ := cancelRes.json()["request_id"].(string)
	check("a long attempt is running", cancelRes.Status == http.StatusAccepted, cancelRes.brief())
	cancelStream := make(chan *sseStream, 1)
	go func() { cancelStream <- svc.openSSE("/v1/requests/"+cancelID+"/events", 180*time.Second) }()
	waitProgress(svc, cancelID, 60*time.Second)
	cancelAt := time.Now()
	cancelled := svc.call("POST", "/v1/requests/"+cancelID+"/cancel?grace_ms=2000", nil)
	check("POST /cancel is ACCEPTED, and says a terminal still decides",
		cancelled.Status == http.StatusAccepted &&
			cancelled.json()["status"] == "cancel_requested", cancelled.brief())
	cs := <-cancelStream
	cancelSettled := time.Since(cancelAt)
	cancelTerminal := ""
	for _, e := range cs.events {
		if strings.HasPrefix(e.Type, "request.") && (e.Type == "request.canceled" ||
			e.Type == "request.failed" || e.Type == "request.completed") {
			cancelTerminal = fmt.Sprint(e.Type, " ", e.Payload["cause"])
		}
	}
	check("the attempt settled on its OWN journaled terminal", cs.stopped,
		cancelTerminal+" in "+ms(cancelSettled))
	cancelLife := svc.call("GET", "/v1/requests/"+cancelID, nil)
	check("and the request is settled in the authority too",
		cancelLife.json()["status"] != "in_progress", fmt.Sprint(cancelLife.json()["status"]))
	check("the CANCELED terminal is what settled it, not the cancel call",
		strings.Contains(cancelTerminal, "canceled"), cancelTerminal)
	cancelLifeOuts, _ := cancelLife.json()["outputs"].([]any)
	check("a canceled attempt published NOTHING", len(cancelLifeOuts) == 0,
		fmt.Sprintf("%d visible output(s)", len(cancelLifeOuts)))
	lateCancel := svc.call("POST", "/v1/requests/"+cancelID+"/cancel", nil)
	check("a cancel arriving AFTER the terminal is idempotent, never an error",
		lateCancel.Status == http.StatusOK, lateCancel.brief())

	head("the multiplexed stream and cursor resume")
	multiplexed := svc.call("GET", "/v1/events?cursor=0", nil)
	_ = multiplexed
	// Resume from a cursor in the MIDDLE of the first request's history and prove the
	// tail replays exactly, in order.
	first := svc.openSSE("/v1/requests/"+requestID+"/events?cursor=0", 20*time.Second)
	check("a settled request replays its whole durable history from cursor 0",
		first.stopped && len(first.events) >= 3, strings.Join(first.types(), " · "))
	if len(first.events) >= 2 {
		mid := first.events[0].EventID
		resumed := svc.openSSE(fmt.Sprintf("/v1/requests/%s/events?cursor=%d", requestID, mid),
			20*time.Second)
		check("resuming from an event id replays exactly the tail after it",
			len(resumed.events) == len(first.events)-1 && resumed.stopped,
			fmt.Sprintf("%d of %d after id %d", len(resumed.events), len(first.events), mid))
		check("and NO live frame is replayed (it was never durable)",
			resumed.count("request.progress") == 0, "0 progress frames on a resume")
	}
	// The multiplexed stream carries every request's rows under one cursor. It does NOT
	// terminal-stop: a terminal there ends one request, not the connection, which is the
	// whole reason a UI can watch ten requests with one EventSource.
	all := svc.openMultiplexed("/v1/events?cursor=0", 4*time.Second)
	ids := map[string]bool{}
	for _, e := range all.events {
		ids[e.RequestID] = true
	}
	check("the MULTIPLEXED stream carries every request on ONE connection", len(ids) >= 3,
		fmt.Sprintf("%d requests, %d events, cursor %d", len(ids), len(all.events), all.cursor))
	monotonic := true
	last := int64(0)
	for _, e := range all.events {
		if e.EventID <= last {
			monotonic = false
		}
		last = e.EventID
	}
	check("its ids are monotonic and totally ordered across requests", monotonic,
		fmt.Sprintf("head %d", last))

	head("teardown")
	stop := svc.call("DELETE", "/v1/local/workers/"+instance, nil)
	check("DELETE /v1/local/workers drains and stops the process group",
		stop.Status == http.StatusOK && stop.json()["stopped"] == true, stop.brief())
	now := gpuReleased(idle, 60*time.Second)
	check("the GPU is back at its idle baseline", now <= idle+40,
		fmt.Sprintf("%d MiB now, %d MiB before", now, idle))
}

// stat is min/max/mean over a run of samples, so a single slow one is visible rather than
// averaged away.
type stat struct {
	n        int
	sum      time.Duration
	min, max time.Duration
}

func (s *stat) add(d time.Duration) {
	s.n++
	s.sum += d
	if s.min == 0 || d < s.min {
		s.min = d
	}
	if d > s.max {
		s.max = d
	}
}

func (s stat) String() string {
	if s.n == 0 {
		return "no samples"
	}
	return fmt.Sprintf("%s min / %s max / %s mean over %d", ms(s.min), ms(s.max),
		ms(s.sum/time.Duration(s.n)), s.n)
}

// benchAPI measures what cl-006 adds ON TOP of cl-001's direct numbers. The comparison is
// the point: 35–43 ms submit->accepted through the orchestrator's Go surface, versus the
// same thing through a real HTTP client on the same machine.
func benchAPI(svc *liveService, boot, acceptedHTTP, total, fetch time.Duration, mediaBytes int, mediaID string) {
	fmt.Printf("  worker spawn -> READY through the API                : %s\n", ms(boot))
	fmt.Printf("  POST /v1/requests -> 202 (cold)                      : %s\n", ms(acceptedHTTP))
	fmt.Printf("  submit -> terminal on the SSE stream                 : %s\n", ms(total))

	// Ten warm runs, each measured on FOUR boundaries at once. `submit -> accepted` is
	// the one that compares directly with cl-001's 35–43 ms cold / 19.4–33.6 ms warm:
	// there the orchestrator's Go surface was called in-process, here the same boundary is
	// crossed by a real HTTP client watching a real SSE stream.
	var httpS, acceptS, terminalS, progressS, ackS stat
	for i := 0; i < 10; i++ {
		began := time.Now()
		res := svc.call("POST", "/v1/requests", map[string]any{
			"endpoint": "cozy/sdxl-unet", "function": "denoise",
			"input": map[string]any{"steps": 2, "latent": 64, "seed": 900 + i},
		}, "Idempotency-Key", fmt.Sprintf("bench-%d", i))
		if res.Status != http.StatusAccepted {
			continue
		}
		httpS.add(time.Since(began))
		rid := fmt.Sprint(res.json()["request_id"])
		st := svc.openSSE("/v1/requests/"+rid+"/events?cursor=0", 120*time.Second)
		if e := st.find("request.accepted"); e != nil {
			acceptS.add(e.received.Sub(began))
		}
		if e := st.find("request.progress"); e != nil {
			progressS.add(e.received.Sub(began))
		}
		if e := st.find("request.completed"); e != nil {
			terminalS.add(e.received.Sub(began))
			// The stream's OWN latency: the interval between the terminal row's
			// committed timestamp and the frame arriving at this client.
			if at, err := time.Parse(time.RFC3339Nano, e.At); err == nil {
				ackS.add(e.received.Sub(at))
			}
		}
	}
	fmt.Printf("  POST /v1/requests -> 202 (warm, 10 runs)             : %s\n", httpS)
	fmt.Printf("  submit -> `request.accepted` seen by the SSE client  : %s\n", acceptS)
	fmt.Printf("  submit -> FIRST live progress frame at the client    : %s\n", progressS)
	fmt.Printf("  submit -> terminal frame at the client               : %s\n", terminalS)
	fmt.Printf("  terminal committed -> SSE frame at the client        : %s\n", ackS)

	// Media throughput, 100 fetches of the real 512 px PNG. At this object size the
	// number is LATENCY-bound rather than bandwidth-bound, and saying so is the honest
	// reading: what it measures is the per-fetch floor a UI pays per thumbnail.
	if mediaID != "" {
		began := time.Now()
		total := 0
		for i := 0; i < 100; i++ {
			r := svc.call("GET", "/v1/media/"+mediaID, nil)
			total += len(r.Body)
		}
		elapsed := time.Since(began)
		fmt.Printf("  media fetch, 100 x %d B (latency-bound)         : %s total, %.2f ms/fetch, %.1f MiB/s\n",
			total/100, ms(elapsed), float64(elapsed.Microseconds())/1000/100,
			float64(total)/elapsed.Seconds()/(1<<20))
		// The same bytes through one ranged read, which is what a video element does:
		// it isolates the transfer from the per-request floor.
		began = time.Now()
		streamedBytes := 0
		for i := 0; i < 100; i++ {
			r := svc.call("GET", "/v1/media/"+mediaID, nil, "Range", "bytes=0-4095")
			streamedBytes += len(r.Body)
		}
		fmt.Printf("  media fetch, 100 x 4 KiB Range                       : %s total, %.2f ms/fetch\n",
			ms(time.Since(began)), float64(time.Since(began).Microseconds())/1000/100)
	}
	fmt.Printf("  one media fetch, cold (%d B)                      : %s\n", mediaBytes, ms(fetch))
	fmt.Printf("  service RSS while serving                           : %s\n", serviceRSS(svc))
}

func serviceRSS(svc *liveService) string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", svc.cmd.Process.Pid))
	if err != nil {
		return "unreadable"
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			var kb float64
			fmt.Sscanf(strings.TrimPrefix(line, "VmRSS:"), "%f", &kb)
			return fmt.Sprintf("%.1f MiB", kb/1024)
		}
	}
	return "unreadable"
}

// waitReady polls the LOCAL module for an intake-READY worker — the same fact a client
// has, through the same route, with no privileged view of the orchestrator.
// waitReady polls until the placement's SERVING AXIS is DISPATCHABLE for a real plan
// (rev-2 retired IntakeState with both of its uses), and RETURNS EARLY on a fault. A
// worker this owner has already refused — a foreign instance, an unpinned release, a wire
// schema this build does not speak — is not slow, and waiting out five minutes to call it
// slow reports a timeout for a decision that was reached in milliseconds.
func waitReady(svc *liveService, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		res := svc.call("GET", "/v1/local/workers", nil)
		body := string(res.Body)
		if strings.Contains(body, `"serving":"DISPATCHABLE"`) && strings.Contains(body, "sha256:") {
			return true
		}
		// A REFUSAL ends the wait; a FAULT does not. This host's own claim-time verdict is
		// settled, and a placement holding a degraded warm case still activates.
		if strings.Contains(body, `"refusal":`) && !strings.Contains(body, `"refusal":""`) {
			fmt.Println("    this host refused the worker: " + between(body, `"refusal":"`, `"`))
			return false
		}
		time.Sleep(200 * time.Millisecond)
	}
	return false
}

// between lifts one JSON string value out of a rendered body for a diagnostic line. It is
// a DIAGNOSTIC, deliberately not a parse: the arm's verdict never depends on it.
func between(s, open, close string) string {
	i := strings.Index(s, open)
	if i < 0 {
		return ""
	}
	rest := s[i+len(open):]
	j := strings.Index(rest, close)
	if j < 0 {
		return rest
	}
	return rest[:j]
}

// waitProgress blocks until the attempt is provably RUNNING — a live progress frame, not
// a clock. Cancelling before the executor has started is a different arm.
func waitProgress(svc *liveService, requestID string, timeout time.Duration) bool {
	done := make(chan bool, 1)
	go func() {
		st := svc.openSSE("/v1/requests/"+requestID+"/events?from=now", timeout)
		done <- st.count("request.progress") > 0
	}()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		res := svc.call("GET", "/v1/requests/"+requestID, nil)
		if res.json()["status"] == "in_progress" {
			time.Sleep(1500 * time.Millisecond) // let real GPU work start
			return true
		}
		select {
		case ok := <-done:
			return ok
		default:
		}
		time.Sleep(100 * time.Millisecond)
	}
	return false
}

func orderedPrefix(types []string, want ...string) bool {
	i := 0
	for _, t := range types {
		if i < len(want) && t == want[i] {
			i++
		}
	}
	return i == len(want)
}

func lanAddress() string {
	out, err := runOut("hostname", "-I")
	if err != nil {
		return ""
	}
	for _, f := range strings.Fields(out) {
		if !strings.HasPrefix(f, "127.") && strings.Contains(f, ".") {
			return f
		}
	}
	return ""
}

// apiRouteTable and manifestHasBindFlag read the PRODUCT's own declarations, so the two
// structural claims — "no route takes a path" and "no flag widens the bind" — are
// checked against the surface itself rather than asserted in prose.
func apiRouteTable() []string {
	out := make([]string, 0, len(api.Routes))
	for _, r := range api.Routes {
		out = append(out, r.Method+" "+r.Path)
	}
	return out
}

func manifestHasBindFlag() bool {
	for _, c := range manifest.Commands {
		if c.Name() != "up" {
			continue
		}
		for _, f := range c.Flags {
			switch f.Name {
			case "--host", "--bind", "--address", "--listen", "--interface", "--lan", "--public":
				return true
			}
		}
	}
	return false
}
