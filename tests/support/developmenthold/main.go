// developmenthold owns the stopped local daemon's idle-pod control during a dev update.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
)

func main() {
	rental := flag.String("rental", "", "exact existing development rental")
	boot := flag.String("boot", "", "exact pinned worker boot")
	check := flag.Bool("check", false, "inspect local eligibility without claiming the daemon lock or contacting the pod")
	flag.Parse()
	cfg, problem := config.Load()
	if problem != nil {
		fmt.Fprintln(os.Stderr, problem)
		os.Exit(1)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()
	emit := func(state cli.DevelopmentHoldResult) {
		if err := json.NewEncoder(os.Stdout).Encode(state); err != nil {
			fmt.Fprintln(os.Stderr, "development hold state could not be written")
			cancel()
		}
	}
	if *check {
		var result *cli.DevelopmentHoldResult
		result, problem = cli.InspectStoredDevelopmentHold(cfg, *rental, *boot)
		if problem == nil {
			emit(*result)
		}
	} else {
		problem = cli.HoldStoredDevelopmentWorker(ctx, cfg, *rental, *boot, os.Stderr, emit)
	}
	if problem != nil {
		fmt.Fprintln(os.Stderr, problem)
		os.Exit(1)
	}
}
