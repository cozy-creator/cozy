package cli

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

// cl-077: the user's prepaid Tensorhub credit, read and bought through the hub's
// embedded billing.

const creditHistoryLimit = 50

func handleCredits(ctx *Context) *exit.Error {
	hctx, cancel := hub.Context()
	balance, problem := client(ctx).Balance(hctx)
	cancel()
	if problem != nil {
		return problem
	}
	return emit(ctx, balanceRecord(balance))
}

func balanceRecord(b hub.Balance) output.Record {
	fields := []output.Field{
		{K: "balance", V: usd(b.Balance)},
		{K: "held", V: usd(b.Held)},
		{K: "available", V: usd(b.Available)},
		{K: "owed", V: usd(b.Owed)},
		{K: "currency", V: b.Currency},
		{K: "balance_micros", V: strconv.FormatInt(b.Balance, 10)},
		{K: "held_micros", V: strconv.FormatInt(b.Held, 10)},
		{K: "available_micros", V: strconv.FormatInt(b.Available, 10)},
		{K: "owed_micros", V: strconv.FormatInt(b.Owed, 10)},
	}
	compact := []string{"balance", "held", "available"}
	if b.Owed != 0 {
		compact = append(compact, "owed")
	}
	record := compactRecord(fields, compact...)
	if b.Available <= 0 {
		record.Next = []string{"cozy credits buy <usd>"}
	}
	return record
}

func handleCreditsHistory(ctx *Context) *exit.Error {
	hctx, cancel := hub.Context()
	moves, next, problem := client(ctx).CreditHistory(hctx, creditHistoryLimit, ctx.Inv.Value("--cursor"))
	cancel()
	if problem != nil {
		return problem
	}
	rows := make([]map[string]string, 0, len(moves))
	for _, m := range moves {
		rows = append(rows, map[string]string{
			"when": m.CreatedAt.UTC().Format(time.RFC3339), "type": m.Type, "amount": usd(m.Amount),
			"source": m.Source, "id": m.ID, "amount_micros": strconv.FormatInt(m.Amount, 10), "source_id": m.SourceID,
		})
	}
	list := output.List{
		Name: "credit history", Fields: []string{"when", "type", "amount", "source"},
		AllFields: []string{"when", "type", "amount", "source", "id", "amount_micros", "source_id"},
		Rows:      rows, Uncapped: true,
	}
	if next != "" {
		list.More = "cozy credits history --cursor " + next
	}
	return emit(ctx, list)
}

func handleCreditsBuy(ctx *Context) *exit.Error {
	amount, problem := parseUSD(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	c := client(ctx)
	hctx, cancel := hub.Context()
	bounds, problem := c.CreditDepositBounds(hctx)
	cancel()
	if problem != nil {
		return problem
	}
	if amount < bounds.Min || amount > bounds.Max {
		return exit.Named(exit.Validation, "credits.amount_out_of_range",
			"%s is outside what one purchase may be: %s to %s", usd(amount), usd(bounds.Min), usd(bounds.Max)).
			WithRemedy("buy between %s and %s", usd(bounds.Min), usd(bounds.Max))
	}
	hctx, cancel = hub.Context()
	before, problem := c.Balance(hctx)
	cancel()
	if problem != nil {
		return problem
	}
	hctx, cancel = hub.Context()
	checkout, problem := c.BuyCredit(hctx, amount)
	cancel()
	if problem != nil {
		return problem
	}
	opened := !ctx.Inv.Bool("--no-browser") && openBrowser(checkout.URL)
	if ctx.Inv.Bool("--no-wait") {
		record := compactRecord([]output.Field{
			{K: "status", V: "awaiting payment"},
			{K: "amount", V: usd(amount)},
			{K: "checkout_url", V: checkout.URL},
			{K: "session", V: checkout.SessionID},
			{K: "expires_at", V: checkout.ExpiresAt.UTC().Format(time.RFC3339)},
		}, "status", "amount", "checkout_url")
		record.Next = []string{"cozy credits"}
		return emit(ctx, record)
	}
	if !ctx.Mode().JSON {
		if opened {
			fmt.Fprintf(ctx.Err, "Opened Stripe Checkout in your browser. If it did not open, pay here:\n  %s\n", checkout.URL)
		} else {
			fmt.Fprintf(ctx.Err, "Pay here:\n  %s\n", checkout.URL)
		}
		fmt.Fprintln(ctx.Err, "Waiting for the payment to land (Ctrl-C stops waiting; a payment still lands)...")
	}
	return waitForCredit(ctx, c, checkout, amount, before)
}

// waitForCredit follows the purchase until its lot lands or the session ends.
func waitForCredit(ctx *Context, c *hub.Client, checkout hub.CreditCheckout, amount int64, before hub.Balance) *exit.Error {
	wait, cancel := context.WithDeadline(context.Background(), checkout.ExpiresAt)
	defer cancel()
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-wait.Done():
			return exit.Named(exit.Deadline, "credits.checkout_expired",
				"checkout %s expired before a payment landed", checkout.SessionID).
				WithRemedy("nothing was charged; buy again").WithNext("cozy credits buy " + dollars(amount))
		case <-tick.C:
		}
		hctx, done := hub.Context()
		session, problem := c.CreditSessionStatus(hctx, checkout.SessionID)
		done()
		if problem != nil {
			continue // a transient read; the session's own deadline bounds the wait
		}
		switch session.Status {
		case "succeeded":
			hctx, done := hub.Context()
			after, problem := c.Balance(hctx)
			done()
			if problem != nil {
				return problem
			}
			record := balanceRecord(after)
			record.Fields = append([]output.Field{{K: "status", V: "paid"}, {K: "bought", V: usd(amount)}}, record.Fields...)
			record.AllFields = append([]output.Field{{K: "status", V: "paid"}, {K: "bought", V: usd(amount)},
				{K: "session", V: checkout.SessionID}, {K: "balance_before", V: usd(before.Balance)}}, record.AllFields...)
			record.Next = nil
			return emit(ctx, record)
		case "failed", "blocked", "expired", "canceled":
			message := "the payment did not complete"
			if session.FailureMessage != nil && *session.FailureMessage != "" {
				message = *session.FailureMessage
			}
			return exit.Named(exit.Failed, "credits.checkout_"+session.Status,
				"checkout %s %s: %s", checkout.SessionID, session.Status, message).
				WithNext("cozy credits buy " + dollars(amount))
		}
	}
}

// parseUSD reads a dollar amount, such as 10 or 25.50, as exact micros.
func parseUSD(text string) (int64, *exit.Error) {
	raw := strings.TrimPrefix(strings.TrimSpace(text), "$")
	whole, frac, _ := strings.Cut(raw, ".")
	bad := exit.Usagef("%q is not a dollar amount", text).WithRemedy("write dollars, such as 10 or 25.50")
	if whole == "" || len(frac) > 6 || strings.Trim(whole+frac, "0123456789") != "" {
		return 0, bad
	}
	dollars, err := strconv.ParseInt(whole, 10, 64)
	if err != nil || dollars > 1_000_000_000 {
		return 0, bad
	}
	micros, _ := strconv.ParseInt((frac + "000000")[:6], 10, 64)
	return dollars*1_000_000 + micros, nil
}

// usd renders micros as dollars: cents always, finer digits only when present.
func usd(micros int64) string {
	sign := ""
	if micros < 0 {
		sign, micros = "-", -micros
	}
	return sign + "$" + dollars(micros)
}

func dollars(micros int64) string {
	frac := strings.TrimRight(fmt.Sprintf("%06d", micros%1_000_000), "0")
	for len(frac) < 2 {
		frac += "0"
	}
	return strconv.FormatInt(micros/1_000_000, 10) + "." + frac
}

// openBrowser asks the desktop to open url; false when it cannot.
func openBrowser(url string) bool {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	return cmd.Start() == nil
}
