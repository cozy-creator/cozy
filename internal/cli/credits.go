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
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// cl-077: the user's prepaid Tensorhub credit, read and bought through the hub's
// embedded billing.

const creditHistoryLimit = 50

func handleCredits(ctx *Context) *exit.Error {
	c := client(ctx)
	// A Hub that cannot say now leaves the balance shown and no warning.
	hctx, cancel := hub.Context()
	credit, creditProblem := c.Credit(hctx)
	cancel()
	if creditProblem == nil {
		rememberCredit(ctx, credit)
		if credit == nil {
			record := compactRecord([]output.Field{{K: "billing", V: "unmetered"}, {K: "hub", V: c.Base()}}, "billing", "hub")
			record.Notes = []string{"This Hub does not meter your account: rentals hold no credit and are never stopped for balance."}
			return emit(ctx, record)
		}
	}
	hctx, cancel = hub.Context()
	balance, problem := c.Balance(hctx)
	cancel()
	if problem != nil {
		return problem
	}
	record := balanceRecord(balance)
	burnFacts(ctx, &record, c, credit)
	if credit.Low() {
		record.Notes = append(record.Notes, lowCreditMessage(credit))
		record.Next = []string{"cozy credits buy <usd>"}
	}
	return emit(ctx, record)
}

// burnFacts adds what running rentals charge an hour and when that runs the credit to the
// floor, as far as the Hub says (th-252); an older Hub says neither.
func burnFacts(ctx *Context, record *output.Record, c *hub.Client, credit *hub.Credit) {
	if credit == nil {
		return
	}
	var shown, typed []output.Field
	if credit.Burn > 0 {
		typed = append(typed, output.Field{K: "burn_usd_micros_per_hour", V: strconv.FormatInt(credit.Burn, 10)})
	}
	if at := credit.DepletesAt; at != nil {
		typed = append(typed, output.Field{K: "depletes_at", V: at.UTC().Format(time.RFC3339)})
	}
	if mode := ctx.Mode(); !mode.Human || mode.JSON {
		record.Fields, record.AllFields = append(record.Fields, typed...), append(record.AllFields, typed...)
		return
	}
	if credit.Burn > 0 {
		burn := usdPerHour(credit.Burn)
		switch n := chargingRentals(c); {
		case n == 1:
			burn += " across 1 rental"
		case n > 1:
			burn += fmt.Sprintf(" across %d rentals", n)
		}
		shown = append(shown, output.Field{K: "burn", V: burn})
	}
	if at := credit.DepletesAt; at != nil {
		shown = append(shown, output.Field{K: "runs out", V: clock(*at)})
	}
	record.Fields = append(record.Fields, shown...)
	record.AllFields = append(append(record.AllFields, shown...), typed...)
}

// chargingRentals is how many of the account's rentals the Hub is charging now; 0 when the
// listing cannot say.
func chargingRentals(c *hub.Client) int {
	hctx, cancel := hub.Context()
	defer cancel()
	rentals, _ := c.Rentals(hctx)
	n := 0
	for _, r := range rentals {
		if r.SpendBasis == "charged" && (r.State == hub.RentalReady || r.State == hub.RentalDegraded) {
			n++
		}
	}
	return n
}

// creditWarningAge is how long a Hub's stated credit warns run and rental commands.
const creditWarningAge = 15 * time.Minute

// lowCreditMessage tells the owner their credit is under the Hub's warning (th-242).
func lowCreditMessage(credit *hub.Credit) string {
	return fmt.Sprintf("Tensorhub credit is low: %s available. Running rentals continue until your balance passes %s, "+
		"then they are shut off, even mid-run. Buy credit: cozy credits buy <usd>", usd(credit.Available), usd(credit.Floor))
}

// stateCredit records what one Hub stated about the account's credit.
func stateCredit(st *records.Store, origin string, credit *hub.Credit) *exit.Error {
	if credit == nil {
		return st.StateHubCredit(origin, nil)
	}
	return st.StateHubCredit(origin, &records.HubCredit{Available: credit.Available, Warning: credit.Warning,
		Floor: credit.Floor, Observed: time.Now()})
}

func rememberCredit(ctx *Context, credit *hub.Credit) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return
	}
	st, problem := records.Open(layout.DB)
	if problem != nil {
		return
	}
	defer st.Close()
	_ = stateCredit(st, ctx.Cfg.HubURL, credit)
}

// warnLowCredit warns a run or rental command whose Hub last stated the account's credit
// under its warning, recently. It asks the Hub nothing: Calcifer's rental listing and
// `cozy credits` keep the statement current.
func warnLowCredit(ctx *Context) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return
	}
	st, problem := records.Open(layout.DB)
	if problem != nil {
		return
	}
	defer st.Close()
	stated, problem := st.HubCredit(ctx.Cfg.HubURL)
	if problem != nil || stated == nil || time.Since(stated.Observed) > creditWarningAge {
		return
	}
	credit := &hub.Credit{Available: stated.Available, Warning: stated.Warning, Floor: stated.Floor}
	if credit.Low() {
		ctx.warn(records.Warning{Code: "credit.low", Message: lowCreditMessage(credit)})
	}
}

// creditCommand is whether argv runs or rents: the commands that spend credit.
func creditCommand(argv []string) bool {
	for i := 0; i < len(argv); i++ {
		switch arg := argv[i]; {
		case arg == "--tensorhub" || arg == "--fields":
			i++
		case strings.HasPrefix(arg, "-"):
		default:
			return arg == "run" || arg == "rental" || arg == "rent"
		}
	}
	return false
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
	deposit, problem := c.CreditDeposit(hctx)
	cancel()
	if problem != nil {
		return problem
	}
	if amount < deposit.Min || amount > deposit.Max {
		return exit.Named(exit.Validation, "credits.amount_out_of_range",
			"%s is outside what one purchase may be: %s to %s", usd(amount), usd(deposit.Min), usd(deposit.Max)).
			WithRemedy("buy between %s and %s", usd(deposit.Min), usd(deposit.Max))
	}
	hctx, cancel = hub.Context()
	before, problem := c.Balance(hctx)
	cancel()
	if problem != nil {
		return problem
	}
	hctx, cancel = hub.Context()
	checkout, problem := c.BuyCredit(hctx, deposit, amount)
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
			{K: "order", V: checkout.ID},
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

// waitForCredit follows the purchase until its lot lands or it ends unpaid.
func waitForCredit(ctx *Context, c *hub.Client, checkout hub.CreditCheckout, amount int64, before hub.Balance) *exit.Error {
	wait, cancel := context.WithDeadline(context.Background(), checkout.ExpiresAt)
	defer cancel()
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-wait.Done():
			return exit.Named(exit.Deadline, "credits.checkout_expired",
				"checkout %s expired before a payment landed", checkout.ID).
				WithRemedy("nothing was charged; buy again").WithNext("cozy credits buy " + dollars(amount))
		case <-tick.C:
		}
		hctx, done := hub.Context()
		purchase, problem := checkout.Status(hctx)
		done()
		if problem != nil {
			continue // a transient read; the purchase's own deadline bounds the wait
		}
		switch {
		case purchase.Paid:
			hctx, done := hub.Context()
			after, problem := c.Balance(hctx)
			done()
			if problem != nil {
				return problem
			}
			record := balanceRecord(after)
			record.Fields = append([]output.Field{{K: "status", V: "paid"}, {K: "bought", V: usd(amount)}}, record.Fields...)
			record.AllFields = append([]output.Field{{K: "status", V: "paid"}, {K: "bought", V: usd(amount)},
				{K: "order", V: checkout.ID}, {K: "balance_before", V: usd(before.Balance)}}, record.AllFields...)
			record.Next = nil
			return emit(ctx, record)
		case purchase.Ended != "":
			message := "the payment did not complete"
			if purchase.Message != "" {
				message = purchase.Message
			}
			return exit.Named(exit.Failed, "credits.checkout_"+purchase.Ended,
				"checkout %s %s: %s", checkout.ID, purchase.Ended, message).
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
