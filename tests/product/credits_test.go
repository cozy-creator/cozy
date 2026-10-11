package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// creditHub is Tensorhub's embedded billing (OpenRails at /billing, th-103) as
// the CLI meets it: the balance, the credit history, the catalog's deposit, and
// an order paid on Stripe's hosted page, which this stand-in settles once the
// buyer has been sent there.
type creditHub struct {
	mu      sync.Mutex
	balance int64
	// orders are the order requests' bodies; ordered and payments the idempotency keys of
	// the orders and their payments.
	orders   []map[string]any
	ordered  []string
	paidWith []map[string]any
	payments []string
	amount   int64
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
	refuse := func(w http.ResponseWriter, status int, code, message string) {
		reply(w, status, map[string]any{"error": map[string]string{"type": "invalid_request_error", "code": code, "message": message}})
	}
	signedIn := func(w http.ResponseWriter, r *http.Request) bool {
		if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer user-") {
			reply(w, 401, map[string]any{"error": map[string]string{"code": "authentication_required", "message": "authentication required"}})
			return false
		}
		return true
	}
	// settled is the purchase's state as its buyer reads it: awaiting them at Stripe, then
	// paid a moment later, when the webhook lands the lot.
	settled := func() bool {
		h.reads++
		if h.reads > 1 && !h.paid {
			h.balance += h.amount
			h.paid = true
		}
		return h.paid
	}
	order := func(status, payment string, next any) map[string]any {
		options := []any{map[string]any{"psp_id": "psp_1", "rail": "stripe", "accepts": []string{"saved_card", "new_card", "redirect"}}}
		if status == "complete" {
			options = []any{}
		}
		return map[string]any{"id": "ord_1", "origin": "customer", "status": status, "currency": "USD",
			"total": strconv.FormatInt(h.amount, 10), "payment_options": options,
			"payment":    map[string]any{"status": payment, "next_action": next, "last_payment_error": nil},
			"expires_at": time.Now().Add(24 * time.Hour).UTC().Format(time.RFC3339)}
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
					"currency": "USD", "source": "grant", "source_id": "ord_1", "created_at": "2026-10-09T05:00:00Z"})
			}
			reply(w, 200, map[string]any{"data": data, "next_cursor": nil})
		})
		mux.HandleFunc("GET /billing/v1/catalog/products", func(w http.ResponseWriter, r *http.Request) {
			// The catalog is a lookup by product key.
			if keys := r.URL.Query()["keys"]; len(keys) != 1 || keys[0] != "api-credit" || len(r.URL.Query()) != 1 {
				refuse(w, 400, "invalid_query", "keys names 1 to 100 products")
				return
			}
			reply(w, 200, map[string]any{"next_cursor": nil, "data": []any{map[string]any{
				"key": "api-credit", "prices": []any{map[string]any{"id": "price_1", "key": "deposit", "archived": false, "currency": "USD",
					"unit_amount": "0", "customer_amount": map[string]string{"min_amount": "10000000", "max_amount": "500000000"}}},
			}}})
		})
		mux.HandleFunc("POST /billing/v1/me/orders", func(w http.ResponseWriter, r *http.Request) {
			if !signedIn(w, r) {
				return
			}
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			h.mu.Lock()
			defer h.mu.Unlock()
			h.orders, h.ordered = append(h.orders, body), append(h.ordered, r.Header.Get("Idempotency-Key"))
			if r.Header.Get("Idempotency-Key") == "" {
				refuse(w, 400, "idempotency_key_required", "The operation needs an Idempotency-Key header.")
				return
			}
			lines, _ := body["lines"].([]any)
			line, _ := lines[0].(map[string]any)
			amount, _ := line["amount"].(string)
			if len(lines) != 1 || line["price_id"] != "price_1" || len(body) != 1 {
				refuse(w, 422, "order_line_unavailable", "The price is not for sale.")
				return
			}
			h.amount, _ = strconv.ParseInt(amount, 10, 64)
			reply(w, 201, order("open", "requires_payment_method", nil))
		})
		mux.HandleFunc("POST /billing/v1/me/orders/ord_1/pay", func(w http.ResponseWriter, r *http.Request) {
			if !signedIn(w, r) {
				return
			}
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			h.mu.Lock()
			defer h.mu.Unlock()
			h.paidWith, h.payments = append(h.paidWith, body), append(h.payments, r.Header.Get("Idempotency-Key"))
			payment, _ := body["payment"].(map[string]any)
			switch {
			case r.Header.Get("Idempotency-Key") == "":
				refuse(w, 400, "idempotency_key_required", "The operation needs an Idempotency-Key header.")
			case len(payment) != 1 || payment["return_url"] != base()+"/credits/return":
				refuse(w, 400, "invalid_param", "return_url is not an allowed origin")
			case body["expected_total"] != strconv.FormatInt(h.amount, 10):
				refuse(w, 409, "order_total_changed", "The order's total changed.")
			default:
				reply(w, 200, order("open", "requires_action", map[string]any{"type": "redirect_to_url",
					"url": "https://checkout.stripe.test/c/pay/cs_test_1", "transactions": []any{}, "psp_id": "psp_1", "payload": map[string]any{}}))
			}
		})
		mux.HandleFunc("GET /billing/v1/me/orders/ord_1", func(w http.ResponseWriter, r *http.Request) {
			if !signedIn(w, r) {
				return
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			if len(h.paidWith) > 0 && settled() {
				reply(w, 200, order("complete", "succeeded", nil))
				return
			}
			reply(w, 200, order("open", "requires_action", nil))
		})
	}
}

// creditCLI runs cozy against hub in its own home.
func creditCLI(t *testing.T, hub string) func(stdin string, args ...string) (int, string) {
	root := t.TempDir()
	return func(stdin string, args ...string) (int, string) {
		t.Helper()
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, append(args, "--tensorhub="+hub)...)...)
		cmd.Env = childEnv(t, root)
		cmd.Stdin = strings.NewReader(stdin) //cozy:stdin-value test login code and account name
		var out bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &out
		_ = cmd.Run()
		return cmd.ProcessState.ExitCode(), out.String()
	}
}

// TestCreditsReadAndBuy is cl-077's surface: `cozy credits` and its history read
// the signed-in user's prepaid credit; `cozy credits buy` refuses an amount
// outside the hub's deposit bounds before any order exists, and otherwise
// orders the deposit at that amount, pays it on Stripe's hosted page and waits
// until the lot lands.
func TestCreditsReadAndBuy(t *testing.T) {
	credits := &creditHub{}
	var hubURL string
	hub := newAccountHubWith(t, credits.routes(t, func() string { return hubURL }))
	hubURL = hub.URL
	run := creditCLI(t, hub.URL)

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
	ordered := len(credits.orders)
	credits.mu.Unlock()
	if ordered != 0 {
		t.Fatalf("a refused amount opened an order")
	}

	code, out := run("", "credits", "buy", "25.50", "--no-browser")
	if code != 0 || !strings.Contains(out, "https://checkout.stripe.test/c/pay/cs_test_1") ||
		!strings.Contains(out, "paid") || !strings.Contains(out, "available: $25.50") {
		t.Fatalf("buy $25.50: exit %d\n%s", code, out)
	}
	credits.mu.Lock()
	orders, paidWith := credits.orders, credits.paidWith
	keys := append(append([]string{}, credits.ordered...), credits.payments...)
	credits.mu.Unlock()
	want := map[string]any{"lines": []any{map[string]any{"price_id": "price_1", "amount": "25500000"}}}
	if len(orders) != 1 || !reflect.DeepEqual(orders[0], want) {
		t.Fatalf("the order asked for %v, want %v", orders, want)
	}
	pay := map[string]any{"payment": map[string]any{"return_url": hub.URL + "/credits/return"}, "expected_total": "25500000"}
	if len(paidWith) != 1 || !reflect.DeepEqual(paidWith[0], pay) {
		t.Fatalf("the order was paid with %v, want %v", paidWith, pay)
	}
	if len(keys) != 2 || keys[0] == "" || keys[1] == "" || keys[0] == keys[1] {
		t.Fatalf("the order and its payment carried idempotency keys %q", keys)
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
	if code, out := run("", "credits", "buy", "10", "--no-browser", "--no-wait", "--json", "--full"); code != 0 ||
		!strings.Contains(out, `"order":"ord_1"`) || !strings.Contains(out, `"status":"awaiting payment"`) {
		t.Fatalf("buy --no-wait: exit %d\n%s", code, out)
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
	run := creditCLI(t, hub.URL)
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
