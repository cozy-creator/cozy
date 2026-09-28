package cli

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

var pinPattern = regexp.MustCompile(`^sha-256 ((?:[0-9A-F]{2}:){31}[0-9A-F]{2})$`)

// handleRunPlay prints a link that plays one output of a run in any browser, straight from
// the rented machine running it. The machine's address and pinned certificate come from the
// Hub's rental view; the capability names this run's output and is signed by this origin's
// device key. All of it rides the fragment, which never leaves the browser.
func handleRunPlay(ctx *Context) *exit.Error {
	daemon, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	life, problem := daemon.Request(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	run, name := runReference(life.Number, life.RequestID), strings.TrimSpace(ctx.Inv.Value("--output"))
	expires, err := time.ParseDuration(ctx.Inv.Value("--expires"))
	switch execution := life.MachineExecution; {
	case name == "" || strings.ContainsAny(name, "#&= "):
		return exit.Named(exit.Usage, "play.output_invalid", "--output names one output: name, or name/index")
	case err != nil || expires <= 0:
		return exit.Named(exit.Usage, "play.expires_invalid", "--expires is a positive duration such as 24h")
	case !life.Rental || life.RentalID == "":
		return exit.Named(exit.Validation, "play.not_rented", "run %s ran on %s: browser playback reaches rented machines", run, life.Machine).
			WithRemedy("open its output file in a desktop player")
	case execution == nil || !execution.Accepted:
		return exit.Named(exit.Conflict, "play.not_started", "run %s has not reached its machine yet", run).WithNext("cozy run watch " + run)
	case execution.Number == 0:
		return exit.Named(exit.Conflict, "play.runtime_predates", "the Runtime on %s predates browser playback: it does not number its runs", life.Machine).
			WithRemedy("update the rental's Runtime, or play a run of a newer rental")
	}
	scoped := ctx.forHub(life.Hub)
	hctx, cancel := hub.Context()
	rental, problem := client(scoped).RentalView(hctx, life.RentalID)
	cancel()
	if problem != nil {
		return problem
	}
	pin := pinPattern.FindStringSubmatch(stringOr(rental.WebRTC))
	switch {
	case rental.State != hub.RentalReady:
		return exit.Named(exit.Conflict, "play.rental_ended", "rental %s is %s: nothing serves run %s's outputs", life.Machine, rental.State, run).
			WithRemedy("open the run's output file in a desktop player")
	case rental.WebRTC == nil && rental.MediaAddress != "":
		return exit.Named(exit.Conflict, "play.image_predates", "rental %s runs an image that predates browser playback (not a cozy.machine image)", life.Machine)
	case rental.WebRTC == nil:
		return exit.Named(exit.Conflict, "play.hub_predates", "rental %s offers no browser playback: its Hub (%s) or its machine's daemon predates it", life.Machine, scoped.Cfg.HubURL)
	case pin == nil:
		return exit.Named(exit.Conflict, "play.pin_unreadable", "the Hub named rental %s's certificate as %q, not sha-256 AB:CD:…", life.Machine, rental.WebRTC.Fingerprint)
	}
	grant := capability.Grant{Machine: life.MachineExecution.Worker, Run: strconv.FormatUint(life.MachineExecution.Number, 10),
		Outputs: []string{name}, Expires: time.Now().Add(expires).Unix()}
	token, problem := scoped.AccountAuth.MintCapability(grant)
	if problem != nil {
		return problem
	}
	link := fmt.Sprintf("%s#v=1&a=%s&f=%s&c=%s&r=%s&o=%s", ctx.Cfg.PlayerURL, rental.WebRTC.Address,
		strings.ToLower(strings.ReplaceAll(pin[1], ":", "")), token, grant.Run, name)
	return emit(ctx, compactRecord([]output.Field{{K: "run", V: run}, {K: "output", V: name},
		{K: "expires", V: time.Unix(grant.Expires, 0).UTC().Format(time.RFC3339)}, {K: "link", V: link}}, "link"))
}

func stringOr(w *hub.RentalWebRTC) string {
	if w == nil {
		return ""
	}
	return w.Fingerprint
}
