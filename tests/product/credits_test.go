package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// creditHub is Tensorhub's embedded billing (OpenRails at /billing, th-103) as
// the CLI meets it: the balance, the credit history, the catalog's deposit
// bounds, and a checkout session paid through Stripe's hosted page, which this
// stand-in settles once the buyer has been sent there.
type creditHub struct {
	mu       sync.Mutex
	balance  int64
	minted   []map[string]string
	paidWith string
	reads    int
	paid     bool
}

func (h *creditHub) routes(t *testing.T, base func() string) func(*http.ServeMux) {
	reply := func(w http.ResponseWriter, status int, body any) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(body)
	}
	signedIn := func(w http.ResponseWriter, r *http.Request) bool {
		if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer user-") {
			reply(w, 401, map[string]any{"error": map[string]string{"code": "authentication_required", "message": "authentication required"}})
			return false
		}
		return true
	}
	return func(mux *http.ServeMux) {
		mux.HandleFunc("GET /billing/v1/me/balance", func(w http.ResponseWriter, r *http.Request) {
			if !signedIn(w, r) {
				return
			}
			if r.URL.Query().Get("currency") != "USD" {
				reply(w, 400, map[string]any{"error": map[string]string{"code": "invalid_param", "message": "currency required"}})
				return
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			amount := strconv.FormatInt(h.balance, 10)
			reply(w, 200, map[string]string{"currency": "USD", "balance_amount": amount, "held_amount": "0",
				"available_amount": amount, "owed_amount": "0", "billing_mode": "prepaid"})
		})
		mux.HandleFunc("GET /billing/v1/me/transactions", func(w http.ResponseWriter, r *http.Request) {
			if !signedIn(w, r) {
				return
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			data := []any{}
			if h.paid {
				data = append(data, map[string]any{"id": "ctx_1", "type": "deposit", "amount": strconv.FormatInt(h.balance, 10),
					"currency": "USD", "source": "checkout", "source_id": "ocs_1", "created_at": "2026-10-09T05:00:00Z"})
			}
			reply(w, 200, map[string]any{"data": data, "next_cursor": nil})
		})
		mux.HandleFunc("GET /billing/v1/products", func(w http.ResponseWriter, r *http.Request) {
			reply(w, 200, map[string]any{"next_cursor": nil, "data": []any{map[string]any{
				"key": "api-credit", "prices": []any{map[string]any{"key": "deposit", "archived": false, "currency": "USD",
					"unit_amount": "0", "customer_amount": map[string]string{"min_amount": "10000000", "max_amount": "500000000"}}},
			}}})
		})
		mux.HandleFunc("POST /billing/v1/me/checkout-sessions", func(w http.ResponseWriter, r *http.Request) {
			if !signedIn(w, r) {
				return
			}
			var body map[string]string
			_ = json.NewDecoder(r.Body).Decode(&body)
			h.mu.Lock()
			h.minted = append(h.minted, body)
			h.mu.Unlock()
			if body["success_url"] != base()+"/credits/return" {
				reply(w, 400, map[string]any{"error": map[string]string{"code": "invalid_param", "message": "success_url not allowed"}})
				return
			}
			reply(w, 201, map[string]any{"id": "ocs_1", "url": nil, "expires_at": time.Now().Add(30 * time.Minute).UTC().Format(time.RFC3339)})
		})
		mux.HandleFunc("GET /billing/v1/checkout-sessions/ocs_1", func(w http.ResponseWriter, r *http.Request) {
			h.mu.Lock()
			defer h.mu.Unlock()
			status := "created"
			if h.paidWith != "" {
				// The buyer pays at Stripe; the webhook lands the lot a moment later.
				h.reads++
				status = "requires_action"
				if h.reads > 1 {
					status = "succeeded"
					if !h.paid {
						amount, _ := strconv.ParseInt(h.minted[len(h.minted)-1]["amount"], 10, 64)
						h.balance += amount
						h.paid = true
					}
				}
			}
			reply(w, 200, map[string]any{"id": "ocs_1", "status": status, "failure_message": nil, "options": []any{
				map[string]string{"id": "opt_elements", "driver": "stripe_elements", "rail": "stripe"},
				map[string]string{"id": "opt_hosted", "driver": "redirect", "rail": "stripe"},
			}})
		})
		mux.HandleFunc("POST /billing/v1/checkout-sessions/ocs_1/pay", func(w http.ResponseWriter, r *http.Request) {
			var body map[string]string
			_ = json.NewDecoder(r.Body).Decode(&body)
			h.mu.Lock()
			h.paidWith = body["option_id"]
			h.mu.Unlock()
			reply(w, 200, map[string]any{"status": "requires_action", "next_action": map[string]any{
				"type": "redirect_to_url", "url": "https://checkout.stripe.test/c/pay/ocs_1"}})
		})
	}
}

// TestCreditsReadAndBuy is cl-077's surface: `cozy credits` and its history read
// the signed-in user's prepaid credit; `cozy credits buy` refuses an amount
// outside the hub's deposit bounds before any checkout exists, and otherwise
// opens a checkout session, pays it through hosted Checkout and waits until the
// lot lands.
func TestCreditsReadAndBuy(t *testing.T) {
	credits := &creditHub{}
	var hubURL string
	hub := newAccountHubWith(t, credits.routes(t, func() string { return hubURL }))
	hubURL = hub.URL
	root := t.TempDir()
	run := func(stdin string, args ...string) (int, string) {
		t.Helper()
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, append(args, "--tensorhub="+hub.URL)...)...)
		cmd.Env = childEnv(t, root)
		cmd.Stdin = strings.NewReader(stdin) //cozy:stdin-value test login code and account name
		var out bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &out
		_ = cmd.Run()
		return cmd.ProcessState.ExitCode(), out.String()
	}

	if code, out := run("", "credits"); code == 0 || !strings.Contains(out, "cozy auth login") {
		t.Fatalf("credits before login: exit %d\n%s", code, out)
	}
	if code, out := run("123456\nbuyer\n", "auth", "login", "buyer@example.test"); code != 0 {
		t.Fatalf("login: exit %d\n%s", code, out)
	}
	if code, out := run("", "credits"); code != 0 || !strings.Contains(out, "available: $0.00") ||
		!strings.Contains(out, "cozy credits buy <usd>") {
		t.Fatalf("an empty balance: exit %d\n%s", code, out)
	}

	for _, amount := range []string{"5", "500.01"} {
		if code, out := run("", "credits", "buy", amount, "--no-browser"); code == 0 ||
			!strings.Contains(out, "$10.00 to $500.00") {
			t.Fatalf("buy %s, outside the bounds: exit %d\n%s", amount, code, out)
		}
	}
	if code, out := run("", "credits", "buy", "1.0000001"); code != 2 || !strings.Contains(out, "not a dollar amount") {
		t.Fatalf("buy a sub-micro amount: exit %d\n%s", code, out)
	}
	credits.mu.Lock()
	minted := len(credits.minted)
	credits.mu.Unlock()
	if minted != 0 {
		t.Fatalf("a refused amount opened a checkout session")
	}

	code, out := run("", "credits", "buy", "25.50", "--no-browser")
	if code != 0 || !strings.Contains(out, "https://checkout.stripe.test/c/pay/ocs_1") ||
		!strings.Contains(out, "paid") || !strings.Contains(out, "available: $25.50") {
		t.Fatalf("buy $25.50: exit %d\n%s", code, out)
	}
	credits.mu.Lock()
	body, paidWith := credits.minted[0], credits.paidWith
	credits.mu.Unlock()
	if body["product_key"] != "api-credit" || body["price_key"] != "deposit" || body["amount"] != "25500000" {
		t.Fatalf("the checkout session asked for %v", body)
	}
	if paidWith != "opt_hosted" {
		t.Fatalf("paid with option %q, not hosted Checkout", paidWith)
	}

	code, out = run("", "credits", "--json", "--full")
	var balance map[string]any
	if code != 0 || json.Unmarshal([]byte(out), &balance) != nil || balance["available_micros"] != "25500000" {
		t.Fatalf("the balance as JSON: exit %d\n%s", code, out)
	}
	if code, out := run("", "credits", "history"); code != 0 || !strings.Contains(out, "deposit") ||
		!strings.Contains(out, "$25.50") {
		t.Fatalf("history after the purchase: exit %d\n%s", code, out)
	}
}
