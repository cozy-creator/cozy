package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalInventoryDiscoversOtherAccountsAndKeepsTheirLastCensus(t *testing.T) {
	for _, named := range []bool{true, false} {
		t.Run(map[bool]string{true: "configured-hub", false: "unnamed-login"}[named], func(t *testing.T) {
			root, hubA, standA := rentalEndRoot(t, "inventory-discovery")
			const rentalA, rentalB = "pr-a4a4a4a4a4a4a4a4a4a4", "pr-b4b4b4b4b4b4b4b4b4b4"
			standA.add(rentalA, "alpha")
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			var unavailable, ended atomic.Bool
			var crossed atomic.Int32
			hubB := httptest.NewServer(machineKeyLogin("machine-b", public, "token-b", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer token-b" {
					crossed.Add(1)
					w.WriteHeader(http.StatusUnauthorized)
					return
				}
				if r.Method != http.MethodGet || r.URL.Path != "/v1/rentals" {
					t.Errorf("listing attempted a non-inventory operation: %s %s", r.Method, r.URL.Path)
					w.WriteHeader(http.StatusMethodNotAllowed)
					return
				}
				if unavailable.Load() {
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = w.Write([]byte(`{"error":{"code":"rental.list_unavailable","message":"B cannot refresh its census"}}`))
					return
				}
				rows := []map[string]any{}
				if !ended.Load() {
					rows = append(rows, map[string]any{"rental_id": rentalB, "name": "bravo", "state": "ready",
						"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 200_000})
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"rentals": rows})
			})))
			defer hubB.Close()
			plantMachineKey(t, root, hubB.URL, "machine-b", private)
			configPath := filepath.Join(root, config.FileName)
			before, err := os.ReadFile(configPath)
			must(t, err)
			if named {
				before = append(before, []byte("hubs:\n  second: "+hubB.URL+"\n")...)
				must(t, os.WriteFile(configPath, before, 0o600))
			}
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			fatal(t, store.RecordRental(records.Rental{ID: rentalA, MachineName: "alpha", State: "ready", Hub: hubA,
				AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000}))
			store.Close()

			type inventory struct {
				Rentals []struct {
					ID         string `json:"rental_id"`
					Hub        string `json:"hub"`
					Unverified bool   `json:"unverified"`
				} `json:"rentals"`
				Live       bool `json:"live"`
				Count      *int `json:"machines_running"`
				Unreadable []struct {
					Hub string `json:"hub"`
				} `json:"unreadable_hubs"`
			}
			list := func(code int, args ...string) inventory {
				t.Helper()
				got, output := runCozy(t, root, append([]string{"rental", "list", "--json"}, args...)...)
				if got != code {
					t.Fatalf("list exit %d, want %d: %s", got, code, output)
				}
				var result inventory
				must(t, json.Unmarshal([]byte(output), &result))
				return result
			}
			for _, args := range [][]string{nil, {"--tensorhub=" + hubB.URL}} {
				found := list(0, args...)
				if !found.Live || found.Count == nil || *found.Count != 2 || len(found.Rentals) != 2 {
					t.Fatalf("account-side rental hidden by Hub selection: %+v", found)
				}
			}
			unavailable.Store(true)
			stale := list(1)
			if stale.Live || stale.Count != nil || len(stale.Rentals) != 2 || len(stale.Unreadable) != 1 || stale.Unreadable[0].Hub != hubB.URL {
				t.Fatalf("partial failure discarded a known rental or claimed current totals: %+v", stale)
			}
			for _, row := range stale.Rentals {
				if row.Unverified != (row.ID == rentalB) || row.Hub != map[string]string{rentalA: hubA, rentalB: hubB.URL}[row.ID] {
					t.Fatalf("staleness or owner Hub crossed rows: %+v", row)
				}
			}
			ended.Store(true)
			unavailable.Store(false)
			if fresh := list(0); len(fresh.Rentals) != 1 || fresh.Rentals[0].ID != rentalA {
				t.Fatalf("a successful empty census did not replace stale rows: %+v", fresh)
			}
			after, err := os.ReadFile(configPath)
			must(t, err)
			if !bytes.Equal(before, after) || crossed.Load() != 0 {
				t.Fatal("listing changed configuration or sent another Hub's credential")
			}
			store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			if row, problem := store.RentalRow(rentalB); problem != nil || row != nil {
				t.Fatalf("observing another client's rental enrolled it locally: %+v %v", row, problem)
			}
		})
	}
}
