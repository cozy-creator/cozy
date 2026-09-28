package producttest

// A WALK MUST OUTLIVE ONE GRANT'S LIFETIME.
//
// A publication grant is signed for a bounded life -- publish.MaxGrantLifetime, ten minutes.
// That is a bound on how long a stolen capability is worth having, and it is only compatible
// with a large artifact if the client MINTS WHERE THE BYTES MOVE. Asking for a batch up front
// stamps ONE expiry on every grant in it: the last object's URL is signed beside the first and
// dies while the stream is still in the opening objects, which silently turns a URL lifetime
// into a limit on how large an artifact may be published at all.
//
// This is not speculative. A 900 s constant already lost job-001's 99 GB artifact, and the H3
// four-lane emits four checkpoints totalling 290-420 GB. Raising the number was rejected twice
// -- cozy-runtime's derived form was ruled "the same ceiling one step removed", and tensorfs
// #87 rejected it for the download half -- because a bigger constant only moves the cliff to a
// bigger artifact.
//
// These arms drive the REAL WeightsGrantWindow and the REAL hub.Client over the REAL wire
// shape against an origin that ENFORCES expiry the way an object store does: a signature
// carries the instant it dies, and a PUT presented after it is refused 403.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/secret"
)

// grantLife stands in for publish.MaxGrantLifetime, shortened so a walk can outlive it in a
// test instead of in ten minutes. Only the stand-in hub reads it.
const grantLife = 2 * time.Second

// objectsInWalk is more objects than one window, so the walk refills mid-flight.
const objectsInWalk = 40

// grantClock is the one clock the stand-in hub, the origin and the walk share, so a walk
// several grant lifetimes long is replayed exactly instead of waited out.
type grantClock struct {
	mu  sync.Mutex
	now time.Time
}

func newGrantClock() *grantClock { return &grantClock{now: time.Now()} }

func (c *grantClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *grantClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.mu.Unlock()
}

// expiringOrigin honours a signature only until the instant the signature itself names,
// exactly as S3/R2 validate X-Amz-Expires on arrival.
type expiringOrigin struct {
	server   *httptest.Server
	accepted atomic.Int64
	refused  atomic.Int64
}

func newExpiringOrigin(clock *grantClock) *expiringOrigin {
	origin := &expiringOrigin{}
	origin.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deadline, err := strconv.ParseInt(r.URL.Query().Get("expires_at"), 10, 64)
		if err != nil || clock.Now().Unix() > deadline {
			origin.refused.Add(1)
			w.WriteHeader(http.StatusForbidden)
			return
		}
		origin.accepted.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	return origin
}

// standInGrantHub serves the one publication route this walk uses, in the exact wire shape
// Tensorhub answers with: every grant carries expires_at_unix, and the response carries the
// hub's own server_time_unix beside them.
func standInGrantHub(origin *expiringOrigin, mints *atomic.Int64, clock *grantClock, heldIDs ...string) *httptest.Server {
	heldSet := make(map[string]bool, len(heldIDs))
	for _, id := range heldIDs {
		heldSet[id] = true
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			ObjectIDs []string `json:"object_ids"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		signature := mints.Add(1) // a real signer answers a different signature every time
		now := clock.Now()
		expires := now.Add(grantLife).Unix()
		grants := make([]map[string]any, 0, len(body.ObjectIDs))
		held := make([]map[string]any, 0, len(body.ObjectIDs))
		for _, id := range body.ObjectIDs {
			if heldSet[id] {
				held = append(held, map[string]any{"object_id": id, "length": int64(1)})
				continue
			}
			grants = append(grants, map[string]any{
				"object_id": id, "length": int64(1),
				"url": fmt.Sprintf("%s/o/%s?expires_at=%d&signature=%d",
					origin.server.URL, url.PathEscape(id), expires, signature),
				"required_headers": map[string]string{},
				"expires_at_unix":  expires,
			})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"server_time_unix": now.Unix(), "held": held, "grants": grants})
	}))
}

func walkObjects() []string {
	ids := make([]string, 0, objectsInWalk)
	for i := range objectsInWalk {
		ids = append(ids, fmt.Sprintf("sha256:%064x", i))
	}
	return ids
}

// spendGrant presents one grant to the origin, as the pod does with the URL it was handed.
func spendGrant(t *testing.T, decision orchestrator.WeightsTransferDecision) int {
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

// hubGrantMinter is the production minter's shape, over the real client and the real route.
func hubGrantMinter(hubURL string) orchestrator.WeightsGrantMinter {
	client := hub.New(config.Config{HubURL: hubURL, HubToken: secret.New("proof")}, "granttest")
	ref := hub.Ref{Org: "proof", Name: "four-lane"}
	return func(ctx context.Context, ids []string) (orchestrator.WeightsGrantMint, *exit.Error) {
		granted, problem := client.GrantKnownTransfers(ctx, ref, "op", ids, "proof")
		if problem != nil {
			return orchestrator.WeightsGrantMint{}, problem
		}
		mint := orchestrator.WeightsGrantMint{ServerTimeUnix: granted.ServerTimeUnix}
		for _, grant := range granted.Grants {
			mint.Decisions = append(mint.Decisions, orchestrator.WeightsTransferDecision{
				ObjectID: grant.ObjectID, Length: grant.Length, URL: grant.URL,
				Headers: grant.Headers, ExpiresAtUnix: uint64(grant.ExpiresAtUnix)})
		}
		for _, held := range granted.Held {
			mint.Decisions = append(mint.Decisions, orchestrator.WeightsTransferDecision{
				ObjectID: held.ObjectID, Length: held.Length, Held: true})
		}
		return mint, nil
	}
}

func TestHeldObjectsReuseOneGrantWindowWithoutAnExpiry(t *testing.T) {
	clock := newGrantClock()
	origin := newExpiringOrigin(clock)
	defer origin.server.Close()
	ids := walkObjects()
	var mints atomic.Int64
	hubServer := standInGrantHub(origin, &mints, clock, ids...)
	defer hubServer.Close()
	window := orchestrator.NewWeightsGrantWindow(hubGrantMinter(hubServer.URL))
	start := clock.Now()
	for index, id := range ids {
		decision, problem := window.Spendable(context.Background(), id, ids[index:],
			start.Add(time.Duration(index)*time.Hour))
		if problem != nil || !decision.Held || decision.URL != "" || decision.ObjectID != id {
			t.Fatalf("object %d: decision=%+v problem=%v", index, decision, problem)
		}
	}
	if got := mints.Load(); got != 1 {
		t.Fatalf("%d already-held objects needed %d grant requests, want one cached window", len(ids), got)
	}
	window.Expire()
	if _, problem := window.Spendable(context.Background(), ids[0], ids, start); problem != nil {
		t.Fatal(problem)
	}
	if got := mints.Load(); got != 2 {
		t.Fatalf("explicit invalidation made %d grant requests, want 2", got)
	}
}

func TestHeldDecisionSurvivesExpiredURLInSameWindow(t *testing.T) {
	clock := newGrantClock()
	origin := newExpiringOrigin(clock)
	defer origin.server.Close()
	ids := walkObjects()[:3]
	var mints atomic.Int64
	hubServer := standInGrantHub(origin, &mints, clock, ids[0], ids[1])
	defer hubServer.Close()
	window := orchestrator.NewWeightsGrantWindow(hubGrantMinter(hubServer.URL))
	start := clock.Now()
	if _, problem := window.Spendable(context.Background(), ids[0], ids, start); problem != nil {
		t.Fatal(problem)
	}
	later := start.Add(grantLife)
	if decision, problem := window.Spendable(context.Background(), ids[1], ids[1:], later); problem != nil || !decision.Held || mints.Load() != 1 {
		t.Fatalf("URL expiry invalidated held custody: decision=%+v problem=%v mints=%d",
			decision, problem, mints.Load())
	}
	decision, problem := window.Spendable(context.Background(), ids[2], ids[2:], later)
	if problem != nil || decision.Held || mints.Load() != 2 {
		t.Fatalf("expired URL was not refreshed: decision=%+v problem=%v mints=%d",
			decision, problem, mints.Load())
	}
	if code := spendGrant(t, decision); code != http.StatusOK {
		t.Fatalf("fresh upload grant was refused: %d", code)
	}
}

// TestAWalkThatOutlivesOneGrantLifetimeCompletes is the whole arm. The walk takes several
// times one signature's declared life; every object still starts its transfer under a
// signature the origin honours, because the window re-mints once the hub's OWN declared life
// is half spent.
func TestAWalkThatOutlivesOneGrantLifetimeCompletes(t *testing.T) {
	clock := newGrantClock()
	origin := newExpiringOrigin(clock)
	defer origin.server.Close()
	var mints atomic.Int64
	hubServer := standInGrantHub(origin, &mints, clock)
	defer hubServer.Close()

	ids := walkObjects()
	window := orchestrator.NewWeightsGrantWindow(hubGrantMinter(hubServer.URL))
	perObject := 3 * grantLife / objectsInWalk
	for index, id := range ids {
		decision, problem := window.Spendable(context.Background(), id, ids[index:], clock.Now())
		if problem != nil {
			t.Fatalf("object %d could not be authorized: %s", index, problem.Message)
		}
		if code := spendGrant(t, decision); code != http.StatusOK {
			t.Fatalf("object %d presented a signature the store refused with %d after %d "+
				"mint(s): a grant went cold before its bytes moved", index, code, mints.Load())
		}
		clock.Advance(perObject)
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
// detect the defect: the SAME origin, the SAME walk, the SAME lifetime, with the grants asked
// for once up front the way finalizeOutput used to. The store refuses.
func TestAnUpFrontMintCannotOutliveItsOwnLifetime(t *testing.T) {
	clock := newGrantClock()
	origin := newExpiringOrigin(clock)
	defer origin.server.Close()
	var mints atomic.Int64
	hubServer := standInGrantHub(origin, &mints, clock)
	defer hubServer.Close()

	ids := walkObjects()
	minted, problem := hubGrantMinter(hubServer.URL)(context.Background(), ids)
	if problem != nil {
		t.Fatal(problem.Message)
	}
	perObject := 3 * grantLife / objectsInWalk
	for index, decision := range minted.Decisions {
		if code := spendGrant(t, decision); code != http.StatusOK {
			if index == 0 {
				t.Fatal("the control refused the very first object; it proves nothing")
			}
			return // the cliff, exactly where an up-front mint puts it
		}
		clock.Advance(perObject)
	}
	t.Fatal("every up-front grant outlived the walk; this control can no longer detect the " +
		"defect the point-of-use mint exists to remove")
}

// TestAnExpiredGrantIsReMintedRatherThanReplayed is the credential-replay arm. A refusal that
// says the signature aged out is answered by a NEW signature; re-presenting the same URL under
// a bumped GrantRevision would be a lie about time.
func TestAnExpiredGrantIsReMintedRatherThanReplayed(t *testing.T) {
	clock := newGrantClock()
	origin := newExpiringOrigin(clock)
	defer origin.server.Close()
	var mints atomic.Int64
	hubServer := standInGrantHub(origin, &mints, clock)
	defer hubServer.Close()

	ids := walkObjects()
	window := orchestrator.NewWeightsGrantWindow(hubGrantMinter(hubServer.URL))
	first, problem := window.Spendable(context.Background(), ids[0], ids, clock.Now())
	if problem != nil {
		t.Fatal(problem.Message)
	}
	window.Expire() // the pod reported weights_grant_expired: nothing about the bytes decided
	second, problem := window.Spendable(context.Background(), ids[0], ids, clock.Now())
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
	var mints atomic.Int64
	clock := newGrantClock()
	origin := newExpiringOrigin(clock)
	defer origin.server.Close()
	hubServer := standInGrantHub(origin, &mints, clock)
	defer hubServer.Close()

	// A window minted now is fresh, and past half its declared life it is not.
	window := orchestrator.NewWeightsGrantWindow(hubGrantMinter(hubServer.URL))
	start := clock.Now()
	if _, problem := window.Spendable(context.Background(), "sha256:"+fmt.Sprintf("%064x", 0),
		walkObjects(), start); problem != nil {
		t.Fatal(problem.Message)
	}
	if window.Stale(start.Add(grantLife/2 - 100*time.Millisecond)) {
		t.Fatal("a grant with more than half its life left was re-minted")
	}
	if !window.Stale(start.Add(grantLife/2 + 100*time.Millisecond)) {
		t.Fatal("a grant past half its declared life was presented instead of re-minted")
	}
	// A dropped window has nothing left to spend.
	window.Expire()
	if !window.Stale(start) {
		t.Fatal("a window with no declared life was treated as fresh")
	}

	// The mint's life is the SHORTEST grant in it, so the first to go cold drives the refill.
	mint := orchestrator.WeightsGrantMint{ServerTimeUnix: 1000,
		Decisions: []orchestrator.WeightsTransferDecision{
			{ObjectID: "a", ExpiresAtUnix: 1600},
			{ObjectID: "b", ExpiresAtUnix: 1300},
			{ObjectID: "c", Held: true},
		}}
	if life := mint.DeclaredLife(); life != 300*time.Second {
		t.Fatalf("declared life is %s, want the shortest grant's 300s", life)
	}
	expired := orchestrator.WeightsGrantMint{ServerTimeUnix: 1000,
		Decisions: []orchestrator.WeightsTransferDecision{{ObjectID: "a", ExpiresAtUnix: 900}}}
	if life := expired.DeclaredLife(); life != 0 {
		t.Fatalf("a grant that expired before it was signed declared %s of life", life)
	}
}
