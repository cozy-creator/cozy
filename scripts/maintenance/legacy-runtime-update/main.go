// This one-off maintenance command updates a pre-bootstrap rental's Python pair.
// It preserves the running agent and never changes the normal CLI's admission contract.
package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"slices"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

type pair struct {
	Runtime  string `json:"runtime"`
	TensorFS string `json:"tensorfs"`
}

type operation struct {
	Operation string `json:"operation"`
	State     string `json:"state"`
	Error     string `json:"error,omitempty"`
	From      pair   `json:"from"`
	To        pair   `json:"to"`
}

type observation struct {
	Capabilities []string `json:"capabilities"`
	pair
	Error     string     `json:"error,omitempty"`
	Update    *operation `json:"update"`
	Bootstrap struct {
		ABI string `json:"abi"`
	} `json:"bootstrap"`
}

func terminal(state string) bool {
	return state == "succeeded" || state == "failed" || state == "rolled_back"
}

func read(ctx context.Context, client *machines.Maintenance) (observation, error) {
	var state observation
	_, problem := client.Do(ctx, http.MethodGet, "/v1/machine/runtime", nil, &state)
	if problem != nil {
		return state, problem
	}
	if state.Error != "" {
		return state, errors.New(state.Error)
	}
	if !slices.Contains(state.Capabilities, machines.RuntimeUpdateCapability) || state.Runtime == "" || state.TensorFS == "" {
		return state, errors.New("machine does not report the legacy Runtime update contract")
	}
	if state.Bootstrap.ABI != "" {
		return state, errors.New("machine has a service bootstrap; use ordinary cozy rental update")
	}
	return state, nil
}

func run(ctx context.Context) error {
	name := flag.String("machine", "", "Existing rental name")
	runtimeVersion := flag.String("runtime-version", "", "Exact published Runtime version")
	tensorfsVersion := flag.String("tensorfs-version", "", "Exact published TensorFS version")
	apply := flag.Bool("apply", false, "Submit or observe the stable update operation; default only reports the plan")
	expectAgent := flag.String("expect-agent-version", "", "Agent version from the reviewed plan; required with --apply")
	expectStarted := flag.Uint64("expect-agent-started-unix-ms", 0, "Agent start time from the reviewed plan; required with --apply")
	flag.Parse()
	version := regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)
	if *name == "" || !version.MatchString(*runtimeVersion) || !version.MatchString(*tensorfsVersion) || flag.NArg() != 0 {
		return errors.New("require --machine, --runtime-version and --tensorfs-version; --apply is optional")
	}
	cfg, problem := config.Load()
	if problem != nil {
		return problem
	}
	layout := home.Paths(cfg.Home)
	store, problem := records.OpenReadOnly(layout.DB)
	if problem != nil {
		return problem
	}
	defer store.Close()
	row, problem := store.RentalByMachine(*name)
	if problem != nil {
		return problem
	}
	if row == nil || !records.RentalReadyState(row.State) || row.ExpectedWorkerID == "" || row.ExpectedWorkerBootID == "" {
		return errors.New("rental is not attached with a complete machine identity")
	}
	key, problem := rental.CreatorIdentityFor(layout, row.ID)
	if problem != nil {
		return problem
	}
	public, err := base64.RawURLEncoding.DecodeString(key.PublicKey())
	if err != nil || len(public) != ed25519.PublicKeySize {
		return errors.New("rental owner key is invalid")
	}
	pin, err := workertls.LoadPin(row.CertPath)
	if err != nil {
		return fmt.Errorf("cannot read pinned machine identity: %w", err)
	}
	transport := &http.Transport{TLSClientConfig: pin.TLSConfig()}
	defer transport.CloseIdleConnections()
	client := &machines.Maintenance{Base: "https://" + row.Address, Machine: row.ExpectedWorkerID,
		Public: public, Sign: key.Sign, Client: &http.Client{Transport: transport, Timeout: 30 * time.Second}}
	connection := &orchestrator.WorkerConnection{RentalID: row.ID, Addr: row.Address, CACert: row.CertPath,
		WorkerID: row.ExpectedWorkerID, WorkerBootID: row.ExpectedWorkerBootID}
	proof, problem := rental.ClaimProof(layout)(connection, orchestrator.RecordOwnerEpoch)
	if problem != nil {
		return problem
	}
	channel, err := grpc.NewClient(row.Address, grpc.WithTransportCredentials(credentials.NewTLS(pin.TLSConfig())))
	if err != nil {
		return err
	}
	defer channel.Close()
	peer, err := pb.NewPodHostClient(channel).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	if err != nil {
		return err
	}
	if peer.MinimumWireMinor == 0 || max(peer.MinimumWireMinor, pb.MinCompatibleWireMinor) > min(peer.WireMinor, pb.WireMinor) {
		return errors.New("machine and maintenance client have no shared worker protocol range")
	}
	claim := &pb.Claim{RecordOwnerId: orchestrator.RecordOwnerID, RecordOwnerEpoch: orchestrator.RecordOwnerEpoch,
		WorkerId: row.ExpectedWorkerID, WorkerBootId: row.ExpectedWorkerBootID, WireMinor: min(peer.WireMinor, pb.WireMinor), Proof: proof}
	describe := func() (*pb.MachineDescription, error) {
		bounded, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		got, err := pb.NewWorkerControlClient(channel).DescribeMachine(bounded, &pb.DescribeMachineQuery{Claim: claim})
		if err != nil {
			return nil, err
		}
		if got.WorkerId != row.ExpectedWorkerID || got.WorkerBootId != row.ExpectedWorkerBootID || got.Host == nil {
			return nil, errors.New("authenticated machine identity differs from the retained rental")
		}
		return got, nil
	}
	before, err := describe()
	if err != nil {
		return err
	}
	if *apply && (*expectAgent == "" || *expectStarted == 0 || before.Host.Version != *expectAgent || before.Host.StartedAtUnixMs != *expectStarted) {
		return errors.New("--apply requires matching --expect-agent-version and --expect-agent-started-unix-ms from the reviewed plan")
	}
	state, err := read(ctx, client)
	if err != nil {
		return err
	}
	want := pair{*runtimeVersion, *tensorfsVersion}
	sum := sha256.Sum256([]byte(row.ID + "\x00" + row.ExpectedWorkerID + "\x00" + row.ExpectedWorkerBootID + "\x00" + want.Runtime + "\x00" + want.TensorFS))
	id := "legacy-runtime-" + hex.EncodeToString(sum[:12])
	queued, running, problem := store.RentalRunCounts(row.ID)
	if problem != nil {
		return problem
	}
	emit := func(status string) error {
		return json.NewEncoder(os.Stdout).Encode(map[string]any{
			"status": status, "machine": *name, "rental_id": row.ID, "worker_id": row.ExpectedWorkerID,
			"boot_id": row.ExpectedWorkerBootID, "agent_version": before.Host.Version, "agent_started_unix_ms": before.Host.StartedAtUnixMs,
			"installed": state.pair, "requested": want, "operation": id, "update": state.Update,
			"queued": queued, "running": running, "agent_action": "preserve existing process; bootstrap migration remains separate"})
	}
	if !*apply {
		return emit("plan_only")
	}
	if state.Update != nil && !terminal(state.Update.State) && state.Update.Operation != id {
		return fmt.Errorf("another update %s is pending; no update submitted", state.Update.Operation)
	}
	if state.Update != nil && state.Update.Operation == id {
		if state.Update.To != (pair{}) && state.Update.To != want {
			return errors.New("existing operation targets another software pair")
		}
	} else if state.pair != want {
		if queued != 0 || running != 0 {
			return errors.New("rental has queued or running work; no update submitted")
		}
		if before.Runtime == nil || before.RuntimeAbsent != "" || before.Host.Phase != "ready" {
			return errors.New("installed Runtime is not ready; no update submitted")
		}
		body, _ := json.Marshal(map[string]any{"operation": id, "runtime": map[string]string{"version": want.Runtime}, "tensorfs": map[string]string{"version": want.TensorFS}})
		if _, problem := client.Do(ctx, http.MethodPost, "/v1/machine/runtime/update", bytes.NewReader(body), nil); problem != nil {
			return fmt.Errorf("update %s may have been accepted; rerun this same command to observe it: %w", id, problem)
		}
		state.Update = &operation{Operation: id, State: "waiting"}
	}
	for state.Update != nil && state.Update.Operation == id && !terminal(state.Update.State) {
		select {
		case <-ctx.Done():
			return fmt.Errorf("observation detached; update %s remains owned by the machine", id)
		case <-time.After(time.Second):
		}
		state, err = read(ctx, client)
		if err != nil {
			return fmt.Errorf("observation interrupted; rerun for update %s: %w", id, err)
		}
		if state.Update == nil || state.Update.Operation != id {
			return errors.New("machine no longer reports this operation; outcome requires inspection")
		}
	}
	if state.Update != nil && state.Update.Operation == id && state.Update.State != "succeeded" {
		_ = emit("update_failed")
		return fmt.Errorf("update %s %s: %s", id, state.Update.State, state.Update.Error)
	}
	state, err = read(ctx, client)
	if err != nil {
		return err
	}
	after, err := describe()
	if err != nil {
		return err
	}
	if state.pair != want || after.Runtime == nil || after.Runtime.Version != want.Runtime || after.Runtime.TensorfsVersion != want.TensorFS || after.RuntimeAbsent != "" || after.Host.Phase != "ready" {
		return errors.New("update lacks authenticated readiness at the requested Runtime/TensorFS versions")
	}
	if after.Host.Version != before.Host.Version || after.Host.StartedAtUnixMs != before.Host.StartedAtUnixMs {
		return errors.New("machine agent identity changed during Runtime-only maintenance")
	}
	return emit("verified")
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if err := run(ctx); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
