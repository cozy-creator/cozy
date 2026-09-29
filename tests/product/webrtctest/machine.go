// Package webrtctest supplies public Runtime read views and a real standalone agent.
// No machine-server implementation is imported into the controller repository.
package webrtctest

import (
	"crypto/ed25519"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"fmt"
	"net"
	"net/netip"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	_ "modernc.org/sqlite"
)

type Entry struct{ Seq uint64 }
type output struct {
	data  []byte
	parts []*pb.RunProductPart
}

// Machine writes only the Runtime's public read-view contract and immutable blobs.
// It acts as the journal writer while the actual agent serves every byte and event.
type Machine struct {
	root, store string
	db          *sql.DB
	mu          sync.Mutex
	keys        []ed25519.PublicKey
	sequence    map[uint64]uint64
	outputs     map[string]*output
}

func NewMachine(dir string, keys ...ed25519.PublicKey) *Machine {
	m := &Machine{root: filepath.Join(dir, "root"), keys: keys, sequence: map[uint64]uint64{}, outputs: map[string]*output{}}
	m.store = filepath.Join(m.root, "var/lib/tensorfs")
	tfs, err := exec.LookPath("tfs")
	check(err)
	if body, err := exec.Command(tfs, "store", "init", m.store).CombinedOutput(); err != nil {
		panic(fmt.Sprintf("initialize fixture Store: %v: %s", err, body))
	}
	path := filepath.Join(m.store, ".cozy-workspace/journal.sqlite3")
	check(os.MkdirAll(filepath.Dir(path), 0755))
	m.db, err = sql.Open("sqlite", path)
	check(err)
	_, err = m.db.Exec(`
 PRAGMA journal_mode=WAL;
 CREATE VIEW machine_workspace_v1 AS SELECT 'workspace' AS workspace_id;
 CREATE TABLE runs(workspace_id TEXT,run_number INTEGER PRIMARY KEY,request_id TEXT,attempt_ordinal INTEGER,generation INTEGER,state TEXT,collected INTEGER,sequence INTEGER,compacted_through INTEGER,accepted_at_ms INTEGER,finished_at_ms INTEGER,worker_id TEXT,worker_boot_id TEXT);
 CREATE VIEW machine_runs_v1 AS SELECT 'cozy-local-client' AS record_namespace,* FROM runs;
 CREATE TABLE events(request_id TEXT,sequence INTEGER,attempt_ordinal INTEGER,at_ms INTEGER,kind TEXT,body_json BLOB,invocation_spec_digest BLOB,outcome_id TEXT,outcome_digest BLOB,outcome_json BLOB);
 CREATE VIEW machine_events_v1 AS SELECT 'cozy-local-client' AS record_namespace,* FROM events;`)
	check(err)
	return m
}
func check(err error) {
	if err != nil {
		panic(err)
	}
}
func (m *Machine) blob(data []byte) *pb.Ref {
	sum := sha256.Sum256(data)
	name := hex.EncodeToString(sum[:])
	path := filepath.Join(m.store, "blobs", name[:2], name[2:4], name)
	check(os.MkdirAll(filepath.Dir(path), 0755))
	check(os.WriteFile(path, data, 0600))
	return &pb.Ref{Digest: sum[:], Length: uint64(len(data))}
}
func (m *Machine) Append(run uint64, name string, index int, data []byte, duration uint64) Entry {
	return m.product(run, name, index, data, duration, false)
}
func (m *Machine) Replace(run uint64, name string, index int, data []byte, duration uint64) Entry {
	return m.product(run, name, index, data, duration, true)
}
func (m *Machine) product(run uint64, name string, index int, data []byte, duration uint64, replace bool) Entry {
	m.mu.Lock()
	defer m.mu.Unlock()
	key := fmt.Sprintf("%d/%s/%d", run, name, index)
	current := m.outputs[key]
	if current == nil || replace {
		current = &output{}
		m.outputs[key] = current
	}
	current.data = append(current.data, data...)
	current.parts = append(current.parts, &pb.RunProductPart{Content: m.blob(data), DurationUs: duration})
	product := &pb.RunProduct{Output: name, Op: pb.RunProductOp_RUN_PRODUCT_OP_SET, Content: m.blob(current.data), Parts: current.parts, MediaType: "video/mp4", Label: name}
	if index >= 0 {
		product.Op = pb.RunProductOp_RUN_PRODUCT_OP_APPEND
		product.Index = uint32(index)
	}
	raw, err := canonical.Bytes(product)
	check(err)
	return m.event(run, "product", raw, nil)
}
func (m *Machine) event(run uint64, kind string, body, outcome []byte) Entry {
	m.sequence[run]++
	seq := m.sequence[run]
	request := fmt.Sprintf("fixture-%d", run)
	tx, err := m.db.Begin()
	check(err)
	defer tx.Rollback()
	_, err = tx.Exec(`INSERT OR IGNORE INTO runs VALUES ('workspace',?,?,1,1,'running',0,0,0,100,0,'wk-test','fixture-boot')`, run, request)
	check(err)
	state, finished := "running", 0
	if outcome != nil {
		state, finished = "succeeded", 200
	}
	_, err = tx.Exec(`UPDATE runs SET sequence=?,state=?,finished_at_ms=? WHERE run_number=?`, seq, state, finished, run)
	check(err)
	digest := sha256.Sum256(outcome)
	_, err = tx.Exec(`INSERT INTO events VALUES (?,?,1,100,?,?,?,?,?,?)`, request, seq, kind, body, make([]byte, 32), "outcome", digest[:], outcome)
	check(err)
	check(tx.Commit())
	return Entry{Seq: seq}
}
func (m *Machine) End(run uint64, status string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	value := pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED
	if status == "failed" {
		value = pb.OutcomeStatus_OUTCOME_STATUS_FAILED
	}
	if status == "canceled" {
		value = pb.OutcomeStatus_OUTCOME_STATUS_CANCELED
	}
	body, err := canonical.Bytes(&pb.AttemptOutcomeBody{Status: value})
	check(err)
	m.event(run, "outcome", []byte(`{}`), body)
}

type Server struct {
	Addr                 netip.AddrPort
	Fingerprint, Machine string
}

func Serve(t testing.TB, host string, m *Machine, agent, python string) Server {
	t.Helper()
	t.Cleanup(func() { m.db.Close() })
	bin := filepath.Join(m.root, "usr/local/bin")
	check(os.MkdirAll(bin, 0755))
	tfs, err := exec.LookPath("tfs")
	check(err)
	check(os.Symlink(tfs, filepath.Join(bin, "tfs")))
	check(os.MkdirAll(filepath.Join(m.root, "opt/cozy"), 0755))
	check(os.Symlink(filepath.Dir(filepath.Dir(python)), filepath.Join(m.root, "opt/cozy/python")))
	free := func() int {
		ln, err := net.Listen("tcp", net.JoinHostPort(host, "0")) //cozy:allow choose a free port for the standalone test agent
		check(err)
		port := ln.Addr().(*net.TCPAddr).Port
		check(ln.Close())
		return port
	}
	worker, webrtc := free(), free()
	keys := make([]string, len(m.keys))
	for i, key := range m.keys {
		keys[i] = base64.RawURLEncoding.EncodeToString(key)
	}
	command := exec.Command(agent)
	command.Env = []string{"COZY_MACHINE_ROOT=" + m.root, "COZY_MACHINE_LIFETIME=persistent", "COZY_LISTEN_HOST=" + host, "COZY_WORKER_ID=wk-test",
		"COZY_WORKER_INTERNAL_PORT=" + strconv.Itoa(worker), "COZY_WEBRTC_INTERNAL_PORT=" + strconv.Itoa(webrtc), "COZY_AUTHORIZED_KEYS=" + strings.Join(keys, ",")}
	log, err := os.OpenFile(filepath.Join(m.root, "fixture-agent.log"), os.O_CREATE|os.O_WRONLY, 0600)
	check(err)
	command.Stdout, command.Stderr = log, log
	check(command.Start())
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	t.Cleanup(func() { _ = command.Process.Signal(syscall.SIGTERM); <-done; log.Close() })
	deadline := time.Now().Add(30 * time.Second)
	for {
		select {
		case err := <-done:
			done <- err
			body, _ := os.ReadFile(log.Name())
			t.Fatalf("fixture agent exited: %v: %s", err, body)
		default:
		}
		raw, err := os.ReadFile(filepath.Join(m.root, "run/cozy/bootstrap/tls.crt"))
		if err == nil {
			if leaf, _ := pem.Decode(raw); leaf != nil {
				if conn, err := net.DialTimeout("tcp", net.JoinHostPort(host, strconv.Itoa(webrtc)), time.Second); err == nil {
					conn.Close()
					return Server{Addr: netip.AddrPortFrom(netip.MustParseAddr(host), uint16(webrtc)), Fingerprint: Fingerprint(leaf.Bytes), Machine: "wk-test"}
				}
			}
		}
		if time.Now().After(deadline) {
			t.Fatal("standalone media fixture did not become ready")
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func Fingerprint(der []byte) string {
	sum := sha256.Sum256(der)
	parts := make([]string, len(sum))
	for i, b := range sum {
		parts[i] = fmt.Sprintf("%02X", b)
	}
	return "sha-256 " + strings.Join(parts, ":")
}
