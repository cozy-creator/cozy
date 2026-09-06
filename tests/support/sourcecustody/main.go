// sourcecustody runs the operator-only handoff against an existing stopped owner.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
)

func main() {
	request := flag.String("request", "", "existing queued producer request")
	rental := flag.String("rental", "", "exact existing rental")
	boot := flag.String("boot", "", "exact pinned worker boot")
	check := flag.Bool("check", false, "read eligibility and progress without claiming the daemon lock or contacting the pod")
	flag.Parse()
	cfg, problem := config.Load()
	if problem != nil {
		fmt.Fprintln(os.Stderr, problem.ErrName())
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	emit := func(progress *cli.SourceCustodyResult) {
		if err := json.NewEncoder(os.Stdout).Encode(map[string]any{"request_id": *request, "rental_id": *rental, "worker_boot_id": *boot, "custody": progress}); err != nil {
			fmt.Fprintln(os.Stderr, "source custody result could not be written")
		}
	}
	var progress *cli.SourceCustodyResult
	if *check {
		progress, problem = cli.InspectStoredSourceCustody(cfg, *request, *rental, *boot)
	} else {
		progress, problem = cli.SyncStoredSourceCustody(ctx, cfg, *request, *rental, *boot, os.Stderr, accountauth.New(cfg), func(held context.Context, result *cli.SourceCustodyResult) {
			emit(result)
			fmt.Fprintln(os.Stderr, "source custody acknowledged; holding control and owner lock until stopped")
			<-held.Done()
		})
	}
	if problem != nil {
		fmt.Fprintln(os.Stderr, problem.ErrName())
		os.Exit(1)
	}
	if *check {
		emit(progress)
	}
}
