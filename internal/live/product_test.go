package live

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

const weightlessRef = "cozy/weightless"

// TestProductPath is the one "does the product work" test: a release is installed the way
// a user installs it, a service comes up, a real invoke crosses the whole stack — the
// generation's own venv, the supervisor, the executor, the worker protocol, the terminal
// transaction, the output publication — and comes back as a typed result with a file on
// disk. Both terminal verdicts are exercised, the payload grammar refuses client-side
// before a request exists, and a `kill -9` of the record owner mid-attempt never yields a
// false success.
//
// The fixture is `fixtures/weightless/`: a real endpoint with no model, no weights and no
// GPU, so this is the whole product path minus the card.
func TestProductPath(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-live", "product")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))

	// REFUSAL PRECEDENCE, before anything is built: -h outranks every gate even on a line
	// that carries no arguments, a planned row names its issue rather than the arguments it
	// would never read, and an implemented row still refuses a short line.
	for _, arm := range []struct {
		args []string
		code int
		want string
	}{
		{[]string{"job", "submit", "-h"}, 0, "usage: cozy job submit"},
		{[]string{"datasets", "push"}, 2, "it lands with issue th-035"},
		{[]string{"describe"}, 2, "`cozy describe` needs"},
	} {
		if code, out := runCozy(t, root, arm.args...); code != arm.code || !strings.Contains(out, arm.want) {
			t.Errorf("cozy %s: exit %d wanted %d, and %q\n%s",
				strings.Join(arm.args, " "), code, arm.code, arm.want, out)
		}
	}

	// THE SERVICE IS DOWN: every server-backed verb is typed exit 9 with the start remedy.
	// This needs no endpoint and no runtime peer, so it is checked before anything is built.
	for _, args := range [][]string{
		{"run", weightlessRef + "/v1/tile", "size=8"},
		{"start", weightlessRef},
		{"stop", "--all"},
		{"doctor"},
	} {
		code, out := runCozy(t, root, args...)
		if code != 9 || !strings.Contains(out, "error(unavailable)") ||
			!strings.Contains(out, "next: cozy up") {
			t.Errorf("cozy %s with the service down [exit %d]\n%s", strings.Join(args, " "), code, out)
		}
	}

	archive := weightlessRelease(t)
	sum, err := os.ReadFile(archive)
	must(t, err)
	digest := sha256.Sum256(sum)
	code, out := runCozy(t, root, "install", weightlessRef,
		"--from", archive, "--digest", "sha256:"+hex.EncodeToString(digest[:]))
	if code != 0 {
		t.Fatalf("cozy install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "describe", weightlessRef); code != 0 ||
		!strings.Contains(out, "tile") || !strings.Contains(out, "refuse") {
		t.Errorf("cozy describe did not list the release's functions [exit %d]\n%s", code, out)
	}

	svc := startService(t, root)

	// THE INVOKE: one real request, all the way through, with the file on disk.
	outDir := filepath.Join(root, "out")
	code, out = runCozy(t, root, "run", weightlessRef+"/v1/tile", "size=32", "seed=7", "--out", outDir)
	if code != 0 {
		t.Fatalf("cozy run [exit %d]\n%s", code, out)
	}
	if !strings.Contains(out, "result") || !strings.Contains(out, "attempt_key") {
		t.Errorf("the run printed no typed result or attempt key\n%s", out)
	}
	saved := filepath.Join(outDir, "image.png")
	info, err := os.Stat(saved)
	if err != nil || info.Size() == 0 {
		t.Fatalf("no image landed under its declared field path at %s: %v", saved, err)
	}
	if head, _ := os.ReadFile(saved); len(head) < 8 || string(head[1:4]) != "PNG" {
		t.Errorf("%s is not the PNG the endpoint encoded", saved)
	}

	// THE OTHER VERDICT on the same path: a failure terminal publishes nothing.
	code, out = runCozy(t, root, "run", weightlessRef+"/v1/refuse")
	if code == 0 {
		t.Errorf("`refuse` returned success\n%s", out)
	}

	// THE PAYLOAD GRAMMAR is typed against the RECORDED schema, client-side, before a
	// request exists — so a typo costs nothing and records nothing.
	for _, arm := range []struct {
		args []string
		code int
		want string
	}{
		{[]string{"run", weightlessRef + "/tile"}, 2, "org/endpoint/vN/function"},
		{[]string{"run", weightlessRef}, 2, "org/endpoint/vN/function"},
		{[]string{"run", weightlessRef + "/v1/tile", "mystery=1"}, 3, "declares no request field"},
		{[]string{"run", weightlessRef + "/v1/tile", "size=lots"}, 3, "declared int"},
		{[]string{"run", weightlessRef + "/v1/nosuch"}, 4, "registers no function"},
		{[]string{"describe", "nobody/nothing"}, 4, "not installed"},
	} {
		code, out := runCozy(t, root, arm.args...)
		if code != arm.code || !strings.Contains(out, arm.want) {
			t.Errorf("cozy %s: exit %d wanted %d, and %q\n%s",
				strings.Join(arm.args, " "), code, arm.code, arm.want, out)
		}
	}

	// THE CRASH. `kill -9` the record owner with an attempt running, bring the same root
	// back, and ask the only question that matters to a user: was anything the dead owner
	// had not committed ever visible, and did the key start a second execution?
	frame := filepath.Join(root, "frame.png")
	must(t, os.WriteFile(frame, decodeHex(t, onePixelPNG), 0o644))
	stream := &strings.Builder{}
	slow := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "run",
		weightlessRef+"/v1/relay", "delay_ms=4000", "--asset", "image="+frame,
		"--idempotency-key", "crash-key", "--stream")
	slow.Env = childEnv(t, root)
	slow.Stdout, slow.Stderr = stream, stream
	setProcessGroup(slow)
	must(t, slow.Start())
	defer func() { _ = slow.Wait() }()

	requestID := ""
	deadline := time.Now().Add(120 * time.Second)
	for time.Now().Before(deadline) && requestID == "" {
		if _, rest, ok := strings.Cut(stream.String(), `"request_id":"`); ok {
			if id, _, ok := strings.Cut(rest, `"`); ok {
				requestID = id
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	if requestID == "" {
		t.Fatalf("the slow run never announced a request id\n%s", stream.String())
	}
	for time.Now().Before(deadline) {
		if svc.call(t, "GET", "/v1/requests/"+requestID, nil).json(t)["status"] == "in_progress" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	must(t, killGroup(slow.Process.Pid))
	must(t, killGroup(svc.cmd.Process.Pid))
	restarted := startService(t, root)

	life := restarted.call(t, "GET", "/v1/requests/"+requestID, nil).json(t)
	if life["request_id"] != requestID {
		t.Fatalf("the request did not survive the crash: %v", life)
	}
	if outs, _ := life["outputs"].([]any); len(outs) != 0 || life["status"] == "completed" {
		t.Errorf("something the dead owner had not committed is visible: %v", life)
	}
	// ONE KEY, ONE REQUEST, ACROSS THE CRASH. The replay is the SAME command line, because
	// the host digests the WHOLE submission — asset identities included — so a hand-rolled
	// body that dropped the image would (correctly) conflict instead of replaying.
	code, out = runCozy(t, root, "run", weightlessRef+"/v1/relay", "delay_ms=4000",
		"--asset", "image="+frame, "--idempotency-key", "crash-key", "--stream")
	if !strings.Contains(out, requestID) {
		t.Errorf("the idempotency key stopped naming the request across the crash [exit %d]\n%s",
			code, out)
	}
}

// weightlessRelease builds the fixture archive from THIS tree and the runtime commit that
// speaks this tree's wire schema. There is no cache and no derivation record: an archive
// built from a peer at another revision refuses `wire_schema_mismatch` at Claim, which is
// a fact about the two repositories rather than something a test can work around.
func weightlessRelease(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	must(t, err)
	repo := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow the peer REPOSITORY the fixture is built from, not the runtime binary; nothing here executes it
	declared, err := exec.Command("git", "-C", repo, "show",
		"HEAD:src/cozy/worker/v1/wire_identity.py").Output()
	if err != nil {
		t.Skipf("no cozy-runtime peer at %s: %v", repo, err)
	}
	if !strings.Contains(string(declared), pb.SchemaDigest) {
		t.Skipf("cr-043: the cozy-runtime peer at %s declares another wire schema; this tree "+
			"speaks rev %d %s, so every release built from it would refuse "+
			"wire_schema_mismatch at Claim", repo, pb.WireSchemaRev, pb.SchemaDigest)
	}
	dir := t.TempDir()
	build := exec.Command("/usr/bin/nice", "-n", "19", "bash",
		"scripts/weightless-release.sh", "--out", dir)
	build.Dir = "../.."
	build.Env = childEnv(t, repo, "RUNTIME_REPO="+repo)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the weightless release: %v\n%s", err, out)
	}
	return filepath.Join(dir, "weightless-1.0.0.tar.gz")
}

func decodeHex(t *testing.T, s string) []byte {
	t.Helper()
	data, err := hex.DecodeString(s)
	must(t, err)
	return data
}

// onePixelPNG is a 1x1 PNG, hex-encoded: the smallest thing that is really an image.
const onePixelPNG = "89504e470d0a1a0a0000000d4948445200000001000000010806000000" +
	"1f15c4890000000d49444154789c6360000002000100ffff03000006000557bfabd40000000049454e44ae426082"
