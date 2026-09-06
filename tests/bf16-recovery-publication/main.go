// Task-owned recovery action harness. Credentials remain in Creator accountauth;
// bounded grants go only to the controlling process over its private pipe.
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const manifest = "sha256:c3b72104aff017e77fc9f5b427c8cd0c5f8dc7c4cbe604584e89f9fc24f1025a"
const header = "sha256:0cffc4f8dbc8681350195db70f92b16ea1b8ccb76d37a45b95e8a4e97c5e3402"
const operation = "bf16-recovery-20260906"

func main() {
	cfg := config.Config{HubURL: "http://127.0.0.1:8819", Home: "/home/fidika/.cozy"}
	client := hub.New(cfg, "bf16-recovery").WithTokenSource(accountauth.New(cfg))
	ref, _ := hub.ParseRef("paul/minimax-h3")
	output := json.NewEncoder(os.Stdout)
	root := os.Args[1]
	decoder := json.NewDecoder(os.Stdin)
	for {
		var cmd struct {
			Action string   `json:"action"`
			IDs    []string `json:"ids"`
		}
		if err := decoder.Decode(&cmd); err != nil {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		var result any
		switch cmd.Action {
		case "open":
			f, err := os.Open(filepath.Join(root, "objects.jsonl"))
			if err != nil {
				panic("object roster absent")
			}
			scan := bufio.NewScanner(f)
			objects := []hub.Object{}
			for scan.Scan() {
				var row hub.Object
				if json.Unmarshal(scan.Bytes(), &row) != nil {
					panic("invalid roster")
				}
				objects = append(objects, row)
			}
			f.Close()
			if scan.Err() != nil || len(objects) != 5424 {
				panic("invalid object census")
			}
			value, problem := client.OpenPublication(ctx, ref, operation, objects, "recover exact BF16 checkpoint from verified NFS sources")
			if problem != nil {
				result = map[string]string{"error": problem.Name}
			} else {
				result = value
			}
		case "grants":
			value, problem := client.GrantKnownTransfers(ctx, ref, operation, cmd.IDs, "bank exact recovered BF16 metadata")
			if problem != nil {
				result = map[string]string{"error": problem.Name}
			} else {
				result = value
			}
		case "verify":
			value, problem := client.VerifyPublicationObjects(ctx, ref, operation, cmd.IDs)
			if problem != nil {
				result = map[string]string{"error": problem.Name}
			} else {
				result = value
			}
		case "finalize":
			value, problem := client.FinalizePublication(ctx, ref, operation, hub.FinalizePublicationRequest{ManifestID: manifest, ManifestLength: 164}, "recover original complete BF16 checkpoint; evaluation pending")
			if problem != nil {
				result = map[string]string{"error": problem.Name}
			} else {
				result = value
			}
		case "metadata":
			value, problem := client.ReadPublicationObjects(ctx, ref, operation, []string{manifest, header})
			if problem != nil {
				result = map[string]string{"error": problem.Name}
				break
			}
			files := map[string]string{manifest: "manifest.json", header: "header.cbor"}
			transport := &http.Client{Timeout: 45 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return fmt.Errorf("redirect refused") }}
			for _, read := range value.Reads {
				if read.Length <= 0 || read.Length > 1048576 || files[read.ObjectID] == "" {
					panic("unexpected metadata grant")
				}
				req, err := http.NewRequestWithContext(ctx, "GET", read.URL, nil)
				if err != nil {
					panic("invalid read grant")
				}
				response, err := transport.Do(req)
				if err != nil {
					panic("metadata GET failed")
				}
				body, err := io.ReadAll(io.LimitReader(response.Body, read.Length+1))
				response.Body.Close()
				sum := sha256.Sum256(body)
				if err != nil || response.StatusCode != 200 || int64(len(body)) != read.Length || "sha256:"+hex.EncodeToString(sum[:]) != read.ObjectID {
					panic("metadata integrity failure")
				}
				if os.WriteFile(filepath.Join(root, files[read.ObjectID]), body, 0600) != nil {
					panic("metadata write failed")
				}
			}
			result = map[string]any{"metadata_readback": len(value.Reads)}
		default:
			panic("unsupported action")
		}
		cancel()
		if output.Encode(result) != nil {
			return
		}
	}
}
