package hub

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Temporary: a hub before OpenRails v0.237 sells credit through a checkout session, and
// its orders take no amount the buyer chooses. Delete this file and its one use in
// BuyCredit once every hub sells credit as an order.

// hubSellsCreditByCheckoutSession is whether the hub's answer to the credit order says
// it cannot take one: it has no order route (its mux's own 404, not a typed one), or its
// orders refuse a line's amount as a field they do not know.
func hubSellsCreditByCheckoutSession(status int, problem *exit.Error) bool {
	if problem == nil {
		return false
	}
	return status == http.StatusNotFound && problem.ErrName() == "hub.untyped_refusal" ||
		status == http.StatusBadRequest && problem.ErrName() == "unknown_field"
}

// buyCreditByCheckoutSession opens a checkout session for amount micros of credit and
// pays it through the hub's redirect option.
func (c *Client) buyCreditByCheckoutSession(ctx context.Context, amount int64) (CreditCheckout, *exit.Error) {
	type session struct {
		Status         string  `json:"status"`
		FailureMessage *string `json:"failure_message"`
		Options        []struct {
			ID     string `json:"id"`
			Driver string `json:"driver"`
		} `json:"options"`
	}
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
	path := "/billing/v1/checkout-sessions/" + url.PathEscape(minted.ID)
	read := func(ctx context.Context) (session, *exit.Error) {
		var out session
		problem := c.do(ctx, call{method: http.MethodGet, path: path}, &out)
		return out, problem
	}
	opened, problem := read(ctx)
	if problem != nil {
		return CreditCheckout{}, problem
	}
	option := ""
	for _, o := range opened.Options {
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
	if problem := c.do(ctx, call{method: http.MethodPost, path: path + "/pay", body: map[string]string{"option_id": option}}, &paid); problem != nil {
		return CreditCheckout{}, problem
	}
	if paid.NextAction == nil || paid.NextAction.URL == nil || *paid.NextAction.URL == "" {
		return CreditCheckout{}, exit.Named(exit.Internal, "credits.checkout_without_url",
			"the hub answered checkout %s with no payment page (status %s)", minted.ID, paid.Status)
	}
	return CreditCheckout{ID: minted.ID, URL: *paid.NextAction.URL, ExpiresAt: minted.ExpiresAt, status: func(ctx context.Context) (CreditPurchase, *exit.Error) {
		now, problem := read(ctx)
		if problem != nil {
			return CreditPurchase{}, problem
		}
		switch now.Status {
		case "succeeded":
			return CreditPurchase{Paid: true}, nil
		case "failed", "blocked", "expired", "canceled":
			purchase := CreditPurchase{Ended: now.Status}
			if now.FailureMessage != nil {
				purchase.Message = *now.FailureMessage
			}
			return purchase, nil
		}
		return CreditPurchase{}, nil
	}}, nil
}
