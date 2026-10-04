// cozy-machine-reclaim is the bounded owner qualification client. It reads an existing
// rental record/key/pin and performs only explicit idle GPU memory maintenance.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	name := flag.String("rental", "", "existing owned ready rental name or ID")
	root := flag.String("home", "", "existing controller home; default ordinary Cozy home")
	flag.Parse()
	if *name == "" {
		return fmt.Errorf("-rental names one existing owned ready rental")
	}
	cfg, problem := config.Load()
	if problem != nil {
		return problem
	}
	if *root != "" {
		cfg.Home = *root
	}
	layout, problem := home.Open(cfg.Home)
	if problem != nil {
		return problem
	}
	if _, err := os.Stat(layout.DB); err != nil {
		return fmt.Errorf("recorded controller database is unavailable: %w", err)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return problem
	}
	defer store.Close()
	subject, problem := rental.Resolve(store, *name)
	if problem != nil {
		return problem
	}
	row := subject.Row
	if row == nil || row.State != "ready" {
		return fmt.Errorf("reclaim requires an existing ready rental")
	}
	pin, err := workertls.LoadPin(row.CertPath)
	if err != nil {
		return err
	}
	owner, problem := rental.CreatorIdentityFor(layout, row.ID)
	if problem != nil {
		return problem
	}
	client, err := machinev1.Dial(row.Address, pin.TLSConfig(), row.ExpectedWorkerID, owner.Signer())
	if err != nil {
		return err
	}
	defer client.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	state, err := client.Status(ctx)
	if err != nil {
		return err
	}
	if state.WorkerId != row.ExpectedWorkerID || row.ExpectedWorkerBootID != "" && state.BootId != row.ExpectedWorkerBootID {
		return fmt.Errorf("recorded rental names a different machine or boot")
	}
	receipt, err := client.ReclaimIdleMemory(ctx)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(receipt)
}
