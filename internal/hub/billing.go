package hub

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The hub's embedded billing (OpenRails at /billing, th-103). Money is integer
// micros of USD; the wire carries it as decimal strings.
const (
	CreditCurrency = "USD"
	creditProduct  = "api-credit"
	creditPrice    = "deposit"
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
// floor and are shut off past it (th-242). nil for an account that never touches credit.
type Credit struct {
	Available int64 `json:"available_usd_micros"`
	Warning   int64 `json:"warning_usd_micros"`
	Floor     int64 `json:"floor_usd_micros"`
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

// DepositBounds is the inclusive range one credit purchase may be, in micros.
type DepositBounds struct {
	Min int64 `json:"min_amount,string"`
	Max int64 `json:"max_amount,string"`
}

// CreditDepositBounds reads what one purchase may be from the hub's catalog.
func (c *Client) CreditDepositBounds(ctx context.Context) (DepositBounds, *exit.Error) {
	var out struct {
		Data []struct {
			Key    string `json:"key"`
			Prices []struct {
				Key            string         `json:"key"`
				Archived       bool           `json:"archived"`
				Currency       string         `json:"currency"`
				CustomerAmount *DepositBounds `json:"customer_amount"`
			} `json:"prices"`
		} `json:"data"`
	}
	if problem := c.do(ctx, call{method: http.MethodGet, path: "/billing/v1/catalog/products", optionalAuth: true}, &out); problem != nil {
		return DepositBounds{}, problem
	}
	for _, product := range out.Data {
		if product.Key != creditProduct {
			continue
		}
		for _, price := range product.Prices {
			if price.Key == creditPrice && !price.Archived && price.Currency == CreditCurrency && price.CustomerAmount != nil {
				return *price.CustomerAmount, nil
			}
		}
	}
	return DepositBounds{}, exit.Named(exit.NotFound, "credits.not_sold",
		"this hub sells no credit (%s/%s)", creditProduct, creditPrice)
}

// CreditCheckout is a credit purchase awaiting the buyer at Stripe Checkout.
type CreditCheckout struct {
	SessionID string    `json:"session_id"`
	URL       string    `json:"url"`
	ExpiresAt time.Time `json:"expires_at"`
}

// BuyCredit opens a checkout session for amount micros of credit and pays it
// through the hub's redirect option, answering where the buyer pays.
func (c *Client) BuyCredit(ctx context.Context, amount int64) (CreditCheckout, *exit.Error) {
	var minted struct {
		ID        string    `json:"id"`
		ExpiresAt time.Time `json:"expires_at"`
	}
	if problem := c.do(ctx, call{method: http.MethodPost, path: "/billing/v1/me/checkout-sessions", auth: true, body: map[string]string{
		"product_key": creditProduct, "price_key": creditPrice,
		"amount": strconv.FormatInt(amount, 10), "success_url": c.base + CreditReturnPath,
	}}, &minted); problem != nil {
		return CreditCheckout{}, problem
	}
	session, problem := c.creditSession(ctx, minted.ID)
	if problem != nil {
		return CreditCheckout{}, problem
	}
	option := ""
	for _, o := range session.Options {
		if o.Driver == "redirect" {
			option = o.ID
			break
		}
	}
	if option == "" {
		return CreditCheckout{}, exit.Named(exit.Unavailable, "credits.no_hosted_checkout",
			"this hub offers no hosted checkout for credit")
	}
	var paid struct {
		Status     string `json:"status"`
		NextAction *struct {
			URL *string `json:"url"`
		} `json:"next_action"`
	}
	if problem := c.do(ctx, call{method: http.MethodPost, path: "/billing/v1/checkout-sessions/" + url.PathEscape(minted.ID) + "/pay",
		body: map[string]string{"option_id": option}}, &paid); problem != nil {
		return CreditCheckout{}, problem
	}
	if paid.NextAction == nil || paid.NextAction.URL == nil || *paid.NextAction.URL == "" {
		return CreditCheckout{}, exit.Named(exit.Internal, "credits.checkout_without_url",
			"the hub answered checkout %s with no payment page (status %s)", minted.ID, paid.Status)
	}
	return CreditCheckout{SessionID: minted.ID, URL: *paid.NextAction.URL, ExpiresAt: minted.ExpiresAt}, nil
}

// CreditSession is where a credit purchase stands: created, requires_action,
// processing, succeeded, failed, blocked, expired or canceled.
type CreditSession struct {
	Status         string  `json:"status"`
	FailureMessage *string `json:"failure_message"`
	Options        []struct {
		ID     string `json:"id"`
		Driver string `json:"driver"`
	} `json:"options"`
}

// CreditSessionStatus reads a purchase's checkout session.
func (c *Client) CreditSessionStatus(ctx context.Context, id string) (CreditSession, *exit.Error) {
	return c.creditSession(ctx, id)
}

func (c *Client) creditSession(ctx context.Context, id string) (CreditSession, *exit.Error) {
	var out CreditSession
	problem := c.do(ctx, call{method: http.MethodGet, path: "/billing/v1/checkout-sessions/" + url.PathEscape(id)}, &out)
	return out, problem
}
