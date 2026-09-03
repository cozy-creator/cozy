package orchestrator

// A WALK MUST OUTLIVE ONE GRANT'S LIFETIME.
//
// A publication grant is signed for a bounded life -- publish.MaxGrantLifetime, ten minutes.
// That is a bound on how long a stolen capability is worth having, and it is only compatible
// with a large artifact if the client mints where the bytes move. Asking for a batch up front
// stamps ONE expiry on every grant in it: the last object's URL is signed beside the first and
// dies while the stream is still in the opening objects, which silently turns a URL lifetime
// into a limit on how large an artifact may be published at all.
//
// This is not speculative. A 900 s constant already lost job-001's 99 GB artifact, and the H3
// four-lane emits four checkpoints totalling 290-420 GB. Raising the number was rejected twice
// (cozy-runtime's derived form was "the same ceiling one step removed"; tensorfs #87 rejected
// it for the download half) because a bigger constant only moves the cliff to a bigger
// artifact.
//
// These proofs drive the REAL grantWindow and the REAL hub.Client over the REAL wire shape
// against an origin that ENFORCES expiry the way an object store does: a signature carries the
// instant it dies, and a PUT presented after it is refused 403.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

// grantLife is this proof's stand-in for publish.MaxGrantLifetime, shortened so a walk can
// outlive it in a test instead of in ten minutes. Nothing reads it but the fake hub.
const grantLife = 2 * time.Second

// objectsInWalk is more objects than one window, so the walk refills mid-flight.
const objectsInWalk = 40

// expiringOrigin is an object store that honours a signature only until the instant the
// signature itself names, exactly as S3/R2 validate X-Amz-Expires on arrival.
type expiringOrigin struct {
	server   *httptest.Server
	accepted atomic.Int64
	refused  atomic.Int64
}

func newExpiringOrigin() *expiringOrigin {
	origin := &expiringOrigin{}
	origin.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deadline, err := strconv.ParseInt(r.URL.Query().Get("expires_at"), 10, 64)
		if err != nil || time.Now().Unix() > deadline {
			origin.refused.Add(1)
			w.WriteHeader(http.StatusForbidden)
			return
		}
		origin.accepted.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	return origin
}

// fakeHub serves the one publication route this walk uses, in the exact wire shape Tensorhub
// answers with: every grant carries expires_at_unix, and the response carries the hub's own
// server_time_unix beside them.
func newFakeHub(t *testing.T, origin *expiringOrigin, mints *atomic.Int64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ObjectIDs []string `json:"object_ids"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		signature := mints.Add(1) // a real signer answers a different signature every time
		now := time.Now()
		expires := now.Add(grantLife).Unix()
		response := map[string]any{"server_time_unix": now.Unix(), "held": []any{}}
		grants := make([]map[string]any, 0, len(body.ObjectIDs))
		for _, id := range body.ObjectIDs {
			grants = append(grants, map[string]any{
				"object_id": id, "length": int64(1),
				"url": fmt.Sprintf("%s/o/%s?expires_at=%d&signature=%d",
					origin.server.URL, url.PathEscape(id), expires, signature),
				"required_headers": map[string]string{},
				"expires_at_unix":  expires,
			})
		}
		response["grants"] = grants
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
}

func walkObjects() []string {
	ids := make([]string, 0, objectsInWalk)
	for i := range objectsInWalk {
		ids = append(ids, fmt.Sprintf("sha256:%064x", i))
	}
	return ids
}

// spend presents one grant to the origin, as the pod does with the URL the owner handed it.
func spend(t *testing.T, decision WeightsTransferDecision) int {
	t.Helper()
	request, err := http.NewRequest(http.MethodPut, decision.URL, http.NoBody)
	if err != nil {
		t.Fatal(err)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	return response.StatusCode
}

func hubMinter(t *testing.T, hubURL string) WeightsGrantMinter {
	t.Helper()
	client := hub.New(config.Config{HubURL: hubURL, HubToken: secret.New("proof")}, "granttest")
	ref := hub.Ref{Org: "proof", Name: "four-lane"}
	return func(ctx context.Context, ids []string) (WeightsGrantWindow, *exit.Error) {
		granted, problem := client.GrantKnownTransfers(ctx, ref, "op", ids, "proof")
		if problem != nil {
			return WeightsGrantWindow{}, problem
		}
		window := WeightsGrantWindow{ServerTimeUnix: granted.ServerTimeUnix}
		for _, grant := range granted.Grants {
			window.Decisions = append(window.Decisions, WeightsTransferDecision{
				ObjectID: grant.ObjectID, Length: grant.Length, URL: grant.URL,
				Headers: grant.Headers, ExpiresAtUnix: uint64(grant.ExpiresAtUnix)})
		}
		return window, nil
	}
}

// TestAWalkThatOutlivesOneGrantLifetimeCompletes is the whole arm. The walk takes several
// times one grant's declared life; every object still starts its transfer under a signature
// the origin honours, because the window re-mints once the hub's OWN declared life is half
// spent.
func TestAWalkThatOutlivesOneGrantLifetimeCompletes(t *testing.T) {
	origin := newExpiringOrigin()
	defer origin.server.Close()
	var mints atomic.Int64
	hubServer := newFakeHub(t, origin, &mints)
	defer hubServer.Close()

	ids := walkObjects()
	window := &grantWindow{mint: hubMinter(t, hubServer.URL)}
	// Each object takes a slice of wall clock, so the whole walk runs several times longer
	// than any one signature lives.
	perObject := 3 * grantLife / objectsInWalk
	for index, id := range ids {
		decision, problem := window.spendable(context.Background(), id, ids[index:], time.Now())
		if problem != nil {
			t.Fatalf("object %d could not be authorized: %s", index, problem.Message)
		}
		if code := spend(t, decision); code != http.StatusOK {
			t.Fatalf("object %d presented a signature the store refused with %d "+
				"after %d mint(s): a grant went cold before its bytes moved",
				index, code, mints.Load())
		}
		time.Sleep(perObject)
	}
	if origin.accepted.Load() != objectsInWalk || origin.refused.Load() != 0 {
		t.Fatalf("the store accepted %d and refused %d of %d objects",
			origin.accepted.Load(), origin.refused.Load(), objectsInWalk)
	}
	if mints.Load() < 2 {
		t.Fatalf("the walk minted %d time(s); a walk this long must ask again", mints.Load())
	}
}

// TestAnUpFrontMintCannotOutliveItsOwnLifetime is the control. It proves the arm above can
// actually detect the defect: the SAME origin, the SAME walk, the SAME short lifetime, with
// the grants asked for once up front the way finalizeOutput used to. The store refuses.
func TestAnUpFrontMintCannotOutliveItsOwnLifetime(t *testing.T) {
	origin := newExpiringOrigin()
	defer origin.server.Close()
	var mints atomic.Int64
	hubServer := newFakeHub(t, origin, &mints)
	defer hubServer.Close()

	ids := walkObjects()
	minted, problem := hubMinter(t, hubServer.URL)(context.Background(), ids)
	if problem != nil {
		t.Fatal(problem.Message)
	}
	perObject := 3 * grantLife / objectsInWalk
	for index, decision := range minted.Decisions {
		if code := spend(t, decision); code != http.StatusOK {
			if index == 0 {
				t.Fatal("the control refused the very first object; it proves nothing")
			}
			return // the cliff, exactly where the up-front mint puts it
		}
		time.Sleep(perObject)
	}
	t.Fatal("every up-front grant outlived the walk; this control can no longer detect " +
		"the defect the point-of-use mint exists to remove")
}

// TestAnExpiredGrantIsReMintedRatherThanReplayed is the credential-replay arm. A refusal that
// says the signature aged out is answered by a NEW signature; re-presenting the same URL under
// a bumped revision would be a lie about time.
func TestAnExpiredGrantIsReMintedRatherThanReplayed(t *testing.T) {
	origin := newExpiringOrigin()
	defer origin.server.Close()
	var mints atomic.Int64
	hubServer := newFakeHub(t, origin, &mints)
	defer hubServer.Close()

	ids := walkObjects()
	window := &grantWindow{mint: hubMinter(t, hubServer.URL)}
	first, problem := window.spendable(context.Background(), ids[0], ids, time.Now())
	if problem != nil {
		t.Fatal(problem.Message)
	}
	// The pod reports weights_grant_expired: nothing about the bytes was decided.
	window.expire()
	second, problem := window.spendable(context.Background(), ids[0], ids, time.Now())
	if problem != nil {
		t.Fatal(problem.Message)
	}
	if second.URL == first.URL {
		t.Fatal("an expired grant was replayed instead of re-minted")
	}
	if mints.Load() != 2 {
		t.Fatalf("the expiry drove %d mint(s), want exactly 2", mints.Load())
	}
}

// TestStalenessIsMeasuredAgainstTheHubsDeclaredLife pins the rule that makes one bounded
// lifetime correct: an object never starts its transfer holding less than half a grant.
func TestStalenessIsMeasuredAgainstTheHubsDeclaredLife(t *testing.T) {
	now := time.Now()
	window := &grantWindow{mintedAt: now, life: 10 * time.Minute}
	if window.stale(now.Add(4 * time.Minute)) {
		t.Fatal("a grant with more than half its life left was re-minted")
	}
	if !window.stale(now.Add(6 * time.Minute)) {
		t.Fatal("a grant past half its declared life was presented instead of re-minted")
	}
	// A hub that declares no usable life leaves nothing to spend.
	if !(&grantWindow{mintedAt: now}).stale(now) {
		t.Fatal("a grant with no declared life was treated as fresh")
	}
	// The window's life is the SHORTEST in it, so the first grant to go cold drives refill.
	minted := WeightsGrantWindow{ServerTimeUnix: 1000, Decisions: []WeightsTransferDecision{
		{ObjectID: "a", ExpiresAtUnix: 1600},
		{ObjectID: "b", ExpiresAtUnix: 1300},
		{ObjectID: "c", Held: true},
	}}
	if life := minted.declaredLife(); life != 300*time.Second {
		t.Fatalf("declared life is %s, want the shortest grant's 300s", life)
	}
	if life := (WeightsGrantWindow{ServerTimeUnix: 1000, Decisions: []WeightsTransferDecision{
		{ObjectID: "a", ExpiresAtUnix: 900},
	}}).declaredLife(); life != 0 {
		t.Fatalf("a grant that expired before it was signed declared %s of life", life)
	}
}
