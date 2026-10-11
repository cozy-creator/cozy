package hub

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// RentalBootLog is one page of a rental attempt's provider boot log: every line the provider
// wrote while the attempt booted, kept by the Hub after the pod is gone, for operators only.
// Attempts is the rental's latest attempt; Booting is a rental still coming up; Next
// continues this attempt.
type RentalBootLog struct {
	Attempt  int           `json:"attempt"`
	Attempts int           `json:"attempts"`
	Booting  bool          `json:"booting"`
	Lines    []BootLogLine `json:"lines"`
	Next     int           `json:"next"`
}

// BootLogLine is one line as the provider wrote it, with the boot step it records.
type BootLogLine struct {
	At   time.Time `json:"at"`
	Step string    `json:"step,omitempty"`
	Line string    `json:"line"`
}

// RentalBootLog reads one attempt's boot log after its first `after` lines; attempt 0 is the
// latest. It is the operator's: the Hub refuses anyone but an admin.
func (c *Client) RentalBootLog(ctx context.Context, id string, attempt, after int) (RentalBootLog, *exit.Error) {
	var out RentalBootLog
	if e := validateRentalID(id); e != nil {
		return out, e
	}
	query := url.Values{"attempt": {strconv.Itoa(attempt)}, "after": {strconv.Itoa(after)}}
	e := c.do(ctx, call{method: http.MethodGet, auth: true, responseBytes: maxRentalResponseBytes,
		path: "/v1/admin/rentals/" + url.PathEscape(id) + "/boot-log?" + query.Encode()}, &out)
	if e != nil && e.ErrName() == "auth.admin_required" {
		return out, exit.Named(exit.Credential, "auth.admin_required", "a rental's boot log is the operator's: only a Tensorhub admin reads it").
			WithRemedy("`cozy rental show` names the boot's progress; `cozy rental logs <rental> --tensorfs` reads the machine's own log")
	}
	return out, e
}
