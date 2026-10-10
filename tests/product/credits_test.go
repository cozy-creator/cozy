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
	// GET /v1/credit (th-242) states the balance against these; null for an unmetered account.
	warning, floor int64
	unmetered      bool
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
		mux.HandleFunc("GET /billing/v1/me", func(w http.ResponseWriter, r *http.Request) {
			if !signedIn(w, r) {
				return
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			// A customer never billed holds no balance yet.
			balances := []any{}
			if h.balance != 0 {
				amount := strconv.FormatInt(h.balance, 10)
				balances = append(balances, map[string]string{"customer_id": "0192f1d4-0000-7000-8000-000000000001", "currency": "USD",
					"balance_amount": amount, "held_amount": "0", "available_amount": amount, "owed_amount": "0", "billing_mode": "prepaid"})
			}
			reply(w, 200, map[string]any{"id": "0192f1d4-0000-7000-8000-000000000001", "balances": balances,
				"collection_payment_methods": []any{}, "unread_notifications": 0})
		})
		mux.HandleFunc("GET /v1/credit", func(w http.ResponseWriter, r *http.Request) {
			if !signedIn(w, r) {
				return
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			var credit map[string]int64
			if !h.unmetered {
				credit = map[string]int64{"available_usd_micros": h.balance, "warning_usd_micros": h.warning, "floor_usd_micros": h.floor}
			}
			reply(w, 200, map[string]any{"credit": credit})
		})
		mux.HandleFunc("GET /billing/v1/me/balance/transactions", func(w http.ResponseWriter, r *http.Request) {
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
		mux.HandleFunc("GET /billing/v1/catalog/products", func(w http.ResponseWriter, r *http.Request) {
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

// TestLowCreditWarns is th-242's CLI half: `cozy credits` says when the account's credit
// is under the Hub's warning and how far it may run before rentals are shut off; a later
// rental command repeats the warning from that statement without asking the Hub; an
// account the Hub meters nothing for is never warned.
func TestLowCreditWarns(t *testing.T) {
	credits := &creditHub{balance: 500_000, warning: 1_000_000, floor: -2_000_000}
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
	if code, out := run("123456\nlow\n", "auth", "login", "low@example.test"); code != 0 {
		t.Fatalf("login: exit %d\n%s", code, out)
	}
	const warning = "Tensorhub credit is low: $0.50 available. Running rentals continue until your balance passes -$2.00"
	if code, out := run("", "credits"); code != 0 || !strings.Contains(out, warning) || !strings.Contains(out, "cozy credits buy <usd>") {
		t.Fatalf("credits under the warning: exit %d\n%s", code, out)
	}
	if _, out := run("", "rental", "list", "--no-watch"); !strings.Contains(out, "warning: "+warning) {
		t.Fatalf("a rental command after a low statement:\n%s", out)
	}
	if _, out := run("", "credits", "history"); strings.Contains(out, "credit is low") {
		t.Fatalf("a command that spends nothing warned:\n%s", out)
	}

	credits.mu.Lock()
	credits.unmetered = true
	credits.mu.Unlock()
	if code, out := run("", "credits"); code != 0 || strings.Contains(out, "credit is low") ||
		!strings.Contains(out, "billing: unmetered") || !strings.Contains(out, "never stopped for balance") ||
		strings.Contains(out, "available") || strings.Contains(out, "cozy credits buy") {
		t.Fatalf("an unmetered account was shown a balance or warned: exit %d\n%s", code, out)
	}
	if code, out := run("", "credits", "--json"); code != 0 || !strings.Contains(out, `"billing":"unmetered"`) ||
		strings.Contains(out, `"available"`) {
		t.Fatalf("an unmetered account as JSON: exit %d\n%s", code, out)
	}
	if _, out := run("", "rental", "list", "--no-watch"); strings.Contains(out, "credit is low") {
		t.Fatalf("a rental command after the Hub stopped metering warned:\n%s", out)
	}
}
