package main

import (
	"encoding/json"
	"fmt"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	"os"
	"strings"
)

func main() {
	if len(os.Args) < 3 {
		panic("authority|attach HOME [FACTS]")
	}
	layout, e := home.Open(os.Args[2])
	if e != nil {
		panic(e.Message)
	}
	if os.Args[1] == "snapshot" {
		store, problem := records.Open(layout.DB)
		if problem != nil {
			panic(problem.Message)
		}
		defer store.Close()
		row, problem := store.RequestByIdempotencyKey(os.Args[3])
		if problem != nil {
			panic(problem.Message)
		}
		if row == nil {
			panic("request absent")
		}
		json.NewEncoder(os.Stdout).Encode(row)
		return
	}
	identity, e := rental.PendingCreatorIdentity(layout, "upgrade55-proof")
	if e != nil {
		panic(e.Message)
	}
	token, e := rental.PendingMediaToken(layout, "upgrade55-proof")
	if e != nil {
		panic(e.Message)
	}
	if os.Args[1] == "authority" {
		json.NewEncoder(os.Stdout).Encode(map[string]any{"worker_id": "old55-worker", "control_public_key_ed25519_b64url": identity.PublicKey(), "media_token_sha256": []string{secret.HashHex(token)}})
		return
	}
	raw, err := os.ReadFile(os.Args[3])
	if err != nil {
		panic(err)
	}
	var facts map[string]string
	if err = json.Unmarshal(raw, &facts); err != nil {
		panic(err)
	}
	cert, err := os.ReadFile(facts["certificate"])
	if err != nil {
		panic(err)
	}
	store, e := records.Open(layout.DB)
	if e != nil {
		panic(e.Message)
	}
	defer store.Close()
	row := records.Rental{ID: "pr-11111111111111111111", MachineName: "old55-worker", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "ready", Hub: facts["hub"], Address: facts["control"], MediaAddress: strings.TrimPrefix(facts["media"], "https://"), ExpectedWorkerID: "old55-worker", ExpectedWorkerBootID: facts["boot"]}
	if e = rental.Attach(layout, store, row, string(cert), token, identity); e != nil {
		panic(e.Message)
	}
	fmt.Println("old55 schema39 rental custody recorded")
}
