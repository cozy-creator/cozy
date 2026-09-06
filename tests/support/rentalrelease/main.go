// rentalrelease is a one-shot operator of the existing authenticated Hub API.
// It starts no daemon and leaves local rental/request records to normal reconciliation.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

func main() {
	id := flag.String("rental", "", "exact existing rental ID")
	boot := flag.String("boot", "", "expected worker boot from prior authenticated readiness")
	release := flag.Bool("release", false, "request deletion and wait for the Hub to report release")
	flag.Parse()
	fail := func(name string) { fmt.Fprintln(os.Stderr, name); os.Exit(1) }
	if *id == "" || *boot == "" || flag.NArg() != 0 {
		fail("rental_release.exact_subject_required")
	}
	cfg, problem := config.Load()
	if problem != nil {
		fail(problem.ErrName())
	}
	client := hub.New(cfg, "source-custody-operator").WithTokenSource(accountauth.New(cfg))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	boundWorker := ""
	inspect := func(afterRelease bool) (hub.Rental, bool) {
		call, cancel := hub.Context()
		stopped := context.AfterFunc(ctx, cancel)
		result, problem := client.Rental(call, *id)
		stopped()
		cancel()
		if problem != nil {
			if problem.Code == exit.NotFound && afterRelease {
				return hub.Rental{ID: *id, State: "absent"}, true
			}
			fail(problem.ErrName())
		}
		if afterRelease && ((result.WorkerBootID != "" && result.WorkerBootID != *boot) || (result.WorkerID != "" && result.WorkerID != boundWorker)) {
			fail("rental_release.worker_identity_changed")
		}
		if result.State == hub.RentalReleased && afterRelease {
			return result, true
		}
		switch result.State {
		case hub.RentalReady, hub.RentalDegraded, hub.RentalFailed, hub.RentalReleaseRequested, hub.RentalReleased:
		default:
			fail("rental_release.unexpected_state")
		}
		if result.WorkerBootID != *boot || result.WorkerID == "" || (boundWorker != "" && result.WorkerID != boundWorker) {
			fail("rental_release.worker_boot_changed")
		}
		boundWorker = result.WorkerID
		return result, result.State == hub.RentalReleased
	}
	emit := func(result hub.Rental) {
		_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"rental_id": result.ID, "state": result.State, "worker_boot_id": result.WorkerBootID, "provider_state": result.ProviderState, "container_state": result.ContainerState})
	}
	current, gone := inspect(false)
	if !*release || gone {
		emit(current)
		return
	}
	call, cancel := hub.Context()
	stopped := context.AfterFunc(ctx, cancel)
	problem = client.Release(call, *id, "source custody complete; operator replacement")
	stopped()
	cancel()
	if problem != nil && problem.Code != exit.NotFound {
		fail(problem.ErrName())
	}
	ticker := time.NewTicker(2 * time.Second) // existing rental CLI observation cadence
	defer ticker.Stop()
	for {
		current, gone = inspect(true)
		if gone {
			emit(current)
			return
		}
		select {
		case <-ctx.Done():
			fail("rental_release.interrupted_before_absence")
		case <-ticker.C:
		}
	}
}
