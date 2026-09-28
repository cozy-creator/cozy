package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// handleRentalLogs is `cozy rental logs <rental>`: every line the provider logged while the
// rental's latest (or --attempt) attempt booted, kept by its Hub after the pod is gone. -f
// follows a rental still coming up, onto each attempt a replan starts, until it is up or over.
func handleRentalLogs(ctx *Context) *exit.Error {
	attempt := 0
	if text := ctx.Inv.Value("--attempt"); text != "" {
		n, err := strconv.Atoi(text)
		if err != nil || n < 1 {
			return exit.Usagef("--attempt is an attempt number from 1, not %q", text)
		}
		attempt = n
	}
	daemon, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	inventory, problem := daemon.RentalInventory(context.Background(), true, true)
	if problem != nil {
		return problem
	}
	subject := strings.TrimSpace(ctx.Inv.Args[0])
	id, origin := "", ""
	for _, row := range append(append(inventory.Rentals, inventory.Unrecorded...), inventory.Pending...) {
		if row.ID != "" && (row.ID == subject || row.MachineName == subject) {
			id, origin = row.ID, row.Hub
			break
		}
	}
	if id == "" {
		return exit.Named(exit.NotFound, "rental.unknown", "no listed rental is named %q", subject).
			WithNext("cozy rental list --all-hubs")
	}
	hubClient := client(ctx.forHub(origin))
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(interrupt)
	follow, human := ctx.Inv.Bool("--follow"), !ctx.Mode().JSON
	var lines []hub.BootLogLine
	for after := 0; ; {
		call, cancel := hub.Context()
		page, problem := hubClient.RentalBootLog(call, id, attempt, after)
		cancel()
		if problem != nil {
			return problem
		}
		if attempt == 0 && page.Attempt > 0 {
			attempt = page.Attempt
			if human && page.Attempts > 1 {
				fmt.Fprintf(ctx.Out, "attempt %d of %d\n", attempt, page.Attempts)
			}
		}
		for _, line := range page.Lines {
			if human {
				fmt.Fprintf(ctx.Out, "%s  %-18s  %s\n", line.At.UTC().Format(time.RFC3339), line.Step, line.Line)
			} else if follow {
				_ = json.NewEncoder(ctx.Out).Encode(map[string]any{"attempt": attempt, "at": line.At, "step": line.Step, "line": line.Line})
			}
		}
		lines, after = append(lines, page.Lines...), page.Next
		switch {
		case len(page.Lines) > 0:
			continue
		case !follow && human:
			if len(lines) == 0 {
				fmt.Fprintf(ctx.Err, "the hub keeps no boot-log lines for attempt %d of %s\n", page.Attempt, subject)
			}
			return nil
		case !follow:
			if err := json.NewEncoder(ctx.Out).Encode(map[string]any{"rental": id, "attempt": page.Attempt,
				"attempts": page.Attempts, "booting": page.Booting, "lines": lines}); err != nil {
				return exit.Internalf("cannot write the boot log: %s", err)
			}
			return nil
		case attempt > 0 && page.Attempts > attempt:
			attempt, after = attempt+1, 0
			if human {
				fmt.Fprintf(ctx.Out, "attempt %d of %d\n", attempt, page.Attempts)
			}
			continue
		case !page.Booting:
			return nil
		}
		select {
		case <-interrupt:
			return nil
		case <-time.After(pollCadence):
		}
	}
}
