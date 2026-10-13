package hub

import (
	"context"
	"crypto/rand"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The hub's embedded billing (OpenRails at /billing, th-103). Money is integer
// micros of USD; the wire carries it as decimal strings.
const (
	CreditCurrency = "USD"
	// CreditReturnPath is the hub page Stripe Checkout returns a buyer to.
	CreditReturnPath = "/credits/return"
)

// Balance is the signed-in user's prepaid credit: Available is Balance less
// what running work holds.
type Balance struct {
	Currency  string `json:"currency"`
	Balance   int64  `json:"balance_amount,string"`
	Held      int64  `json:"held_amount,string"`
	Available int64  `json:"available_amount,string"`
	Owed      int64  `json:"owed_amount,string"`
}

// Balance reads the user's credit from their billing account, which lists a balance
// per currency held: none before the first.
func (c *Client) Balance(ctx context.Context) (Balance, *exit.Error) {
	var out struct {
		Balances []Balance `json:"balances"`
	}
	if problem := c.do(ctx, call{method: http.MethodGet, path: "/billing/v1/me", auth: true}, &out); problem != nil {
		return Balance{}, problem
	}
	for _, b := range out.Balances {
		if b.Currency == CreditCurrency {
			return b, nil
		}
	}
	return Balance{Currency: CreditCurrency}, nil
}

// Credit is the caller's net balance (balance less every hold and anything owed) against
// the Hub's warning and its overdraft floor: rentals keep running into debt down to the
// floor and are shut off past it (th-242). Burn is what running rentals charge an hour
// together, and DepletesAt when that reaches the floor (th-252); zero and nil from an older
// Hub or with nothing charging. nil for an account that never touches credit.
type Credit struct {
	Available  int64      `json:"available_usd_micros"`
	Warning    int64      `json:"warning_usd_micros"`
	Floor      int64      `json:"floor_usd_micros"`
	Burn       int64      `json:"burn_usd_micros_per_hour"`
	DepletesAt *time.Time `json:"depletes_at"`
}

// Low is whether the owner is to be warned.
func (c *Credit) Low() bool { return c != nil && c.Available < c.Warning }

// Credit reads the caller's credit.
func (c *Client) Credit(ctx context.Context) (*Credit, *exit.Error) {
	var out struct {
		Credit *Credit `json:"credit"`
	}
	problem := c.do(ctx, call{method: http.MethodGet, path: "/v1/credit", auth: true}, &out)
	return out.Credit, problem
}

// CreditTransaction is one movement of the user's credit.
type CreditTransaction struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Amount    int64     `json:"amount,string"`
	Currency  string    `json:"currency"`
	Source    string    `json:"source"`
	SourceID  string    `json:"source_id"`
	CreatedAt time.Time `json:"created_at"`
}

// CreditHistory is one page of the user's credit movements, newest first, and
// the cursor of the next ("" at the end).
func (c *Client) CreditHistory(ctx context.Context, limit int, cursor string) ([]CreditTransaction, string, *exit.Error) {
	query := url.Values{"currency": {CreditCurrency}, "limit": {strconv.Itoa(limit)}}
	if cursor != "" {
		query.Set("cursor", cursor)
	}
	var out struct {
		Data       []CreditTransaction `json:"data"`
		NextCursor *string             `json:"next_cursor"`
	}
	problem := c.do(ctx, call{method: http.MethodGet, path: "/billing/v1/me/balance/transactions?" + query.Encode(), auth: true}, &out)
	next := ""
	if out.NextCursor != nil {
		next = *out.NextCursor
	}
	return out.Data, next, problem
}

// CreditDeposit is the credit deposit as the hub's catalog sells it: its price, and the
// inclusive range one purchase may be, in micros.
type CreditDeposit struct {
	PriceID string
	Min     int64
	Max     int64
}

// CreditDeposit reads what one credit purchase may be from the hub's deposit endpoint.
func (c *Client) CreditDeposit(ctx context.Context) (CreditDeposit, *exit.Error) {
	var out struct {
		PriceID string `json:"price_id"`
		Min     int64  `json:"min_amount,string"`
		Max     int64  `json:"max_amount,string"`
	}
	if problem := c.do(ctx, call{method: http.MethodGet, path: "/v1/credit/deposit"}, &out); problem != nil {
		return CreditDeposit{}, problem
	}
	return CreditDeposit{PriceID: out.PriceID, Min: out.Min, Max: out.Max}, nil
}

// CreditCheckout is a credit purchase awaiting the buyer on Stripe's hosted page. ID
// names the purchase's order.
type CreditCheckout struct {
	ID        string
	URL       string
	ExpiresAt time.Time
	// status reads where the purchase stands.
	status func(context.Context) (CreditPurchase, *exit.Error)
}

// CreditPurchase is where a credit purchase stands: Paid once its credit landed, or
// Ended, naming how it closed unpaid (canceled, expired, ...).
type CreditPurchase struct {
	Paid    bool
	Ended   string
	Message string
}

// Status reads where the purchase stands.
func (c CreditCheckout) Status(ctx context.Context) (CreditPurchase, *exit.Error) {
	return c.status(ctx)
}

// creditOrder is the hub's order as a credit purchase reads it.
type creditOrder struct {
	ID        string     `json:"id"`
	Status    string     `json:"status"`
	Total     int64      `json:"total,string"`
	ExpiresAt *time.Time `json:"expires_at"`
	Payment   struct {
		NextAction *struct {
			Type string  `json:"type"`
			URL  *string `json:"url"`
		} `json:"next_action"`
		LastPaymentError *struct {
			Message string `json:"message"`
		} `json:"last_payment_error"`
	} `json:"payment"`
	PaymentOptions []struct {
		Accepts []string `json:"accepts"`
	} `json:"payment_options"`
}

// BuyCredit buys amount micros of credit: an order of the deposit's price at that
// amount, paid on the hub's hosted Stripe page. It answers where the buyer pays.
func (c *Client) BuyCredit(ctx context.Context, deposit CreditDeposit, amount int64) (CreditCheckout, *exit.Error) {
	var order creditOrder
	problem := c.do(ctx, call{method: http.MethodPost, path: "/billing/v1/me/orders", auth: true, idempotency: rand.Text(),
		body: map[string]any{"lines": []map[string]string{{"price_id": deposit.PriceID, "amount": strconv.FormatInt(amount, 10)}}}}, &order)
	if problem != nil {
		return CreditCheckout{}, problem
	}
	hosted := false
	for _, option := range order.PaymentOptions {
		hosted = hosted || slices.Contains(option.Accepts, "redirect")
	}
	if !hosted {
		return CreditCheckout{}, exit.Named(exit.Unavailable, "credits.no_hosted_checkout",
			"this hub offers no hosted checkout for credit")
	}
	path := "/billing/v1/me/orders/" + url.PathEscape(order.ID)
	var paid creditOrder
	if problem := c.do(ctx, call{method: http.MethodPost, path: path + "/pay", auth: true, idempotency: rand.Text(), body: map[string]any{
		"payment":        map[string]string{"return_url": c.base + CreditReturnPath},
		"expected_total": strconv.FormatInt(order.Total, 10),
	}}, &paid); problem != nil {
		return CreditCheckout{}, problem
	}
	next := paid.Payment.NextAction
	if next == nil || next.Type != "redirect_to_url" || next.URL == nil || *next.URL == "" {
		return CreditCheckout{}, exit.Named(exit.Internal, "credits.checkout_without_url",
			"the hub answered order %s with no payment page (status %s)", order.ID, paid.Status)
	}
	expires := time.Now().Add(24 * time.Hour)
	if order.ExpiresAt != nil {
		expires = *order.ExpiresAt
	}
	return CreditCheckout{ID: order.ID, URL: *next.URL, ExpiresAt: expires, status: func(ctx context.Context) (CreditPurchase, *exit.Error) {
		var now creditOrder
		if problem := c.do(ctx, call{method: http.MethodGet, path: path, auth: true}, &now); problem != nil {
			return CreditPurchase{}, problem
		}
		switch now.Status {
		case "complete":
			return CreditPurchase{Paid: true}, nil
		case "open", "processing":
			return CreditPurchase{}, nil
		}
		purchase := CreditPurchase{Ended: now.Status}
		if failure := now.Payment.LastPaymentError; failure != nil {
			purchase.Message = failure.Message
		}
		return purchase, nil
	}}, nil
}
