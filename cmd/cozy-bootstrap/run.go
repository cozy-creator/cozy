package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/mediawire"
)

const (
	adapterPath         = "/opt/cozy/bin/cozy-materialize-launch"
	mediaPath           = "/usr/local/bin/cozy-media"
	stateDir            = "/run/cozy/bootstrap"
	mediaRoot           = "/var/lib/cozy/media"
	plansDir            = "/run/cozy/binding-plans"
	podBootIDPath       = stateDir + "/pod-boot-id"
	certificatePath     = stateDir + "/tls.crt"
	privateKeyPath      = stateDir + "/tls.key"
	receiptPayloadPath  = stateDir + "/readiness-payload"
	receiptEnvelopePath = stateDir + "/readiness-envelope.json"
	tlsServerName       = "cozy-worker"
)

var receiptDomain = []byte("cozy.pod-readiness/1\x00")

type child struct {
	name string
	cmd  *exec.Cmd
	done chan struct{}
	err  error
}

// run performs the generic pod bootstrap. It dials nothing: the whole launch surface is
// the injected environment, and the readiness payload the adapter drops is opaque bytes
// this process only authenticates (cr-048/th-067 retired the two provision documents).
func run(parent context.Context) error {
	cfg, err := parseConfig()
	if err != nil {
		return err
	}
	readinessCtx, cancelReadiness := context.WithDeadline(parent, cfg.deadline)
	defer cancelReadiness()
	if err := prepareState(cfg); err != nil {
		return err
	}
	media, err := startMedia(cfg)
	if err != nil {
		return err
	}
	children := []*child{media}
	defer stopChildren(children)
	adapter, err := startAdapter(cfg)
	if err != nil {
		return err
	}
	children = append(children, adapter)
	if err := publishReceipt(readinessCtx, cfg.hmacKey, children); err != nil {
		return classifyContext(parent, readinessCtx, err)
	}
	cancelReadiness()
	return supervise(parent, children)
}

func prepareState(cfg config) error {
	for _, dir := range []string{stateDir, mediaRoot, plansDir} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return fmt.Errorf("create private runtime directory: %w", err)
		}
		if err := os.Chmod(dir, 0o700); err != nil {
			return fmt.Errorf("protect private runtime directory: %w", err)
		}
	}
	for _, path := range []string{receiptPayloadPath, receiptEnvelopePath} {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("remove stale readiness handoff: %w", err)
		}
	}
	if err := mintPodBootID(); err != nil {
		return err
	}
	return mintCertificate(cfg.deadline)
}

func mintPodBootID() error {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return fmt.Errorf("mint pod_boot_id: %w", err)
	}
	// Every entrypoint bootstrap is a new container boot observation. Replace
	// any stale handoff atomically; only children of this process reuse it.
	return atomicWrite(podBootIDPath, []byte(base64.RawURLEncoding.EncodeToString(raw)+"\n"), 0o400)
}

func mintCertificate(deadline time.Time) error {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return fmt.Errorf("mint TLS key: %w", err)
	}
	serialLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serial, err := rand.Int(rand.Reader, serialLimit)
	if err != nil {
		return fmt.Errorf("mint TLS serial: %w", err)
	}
	now := time.Now()
	template := certificateTemplate(serial, now, deadline)
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return fmt.Errorf("mint TLS certificate: %w", err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return fmt.Errorf("encode TLS key: %w", err)
	}
	if err := atomicWrite(privateKeyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o400); err != nil {
		return err
	}
	return atomicWrite(certificatePath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o444)
}

func certificateTemplate(serial *big.Int, now, deadline time.Time) *x509.Certificate {
	return &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: "cozy-pod-bootstrap"},
		NotBefore:    now.Add(-5 * time.Minute),
		NotAfter:     deadline.Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:     []string{tlsServerName},
	}
}

func startMedia(cfg config) (*child, error) {
	listen := net.JoinHostPort("0.0.0.0", strconv.Itoa(int(cfg.mediaPort)))
	cmd := exec.Command(mediaPath,
		"--listen", listen,
		"--root", mediaRoot,
		"--plans", plansDir,
		"--token-sha256", mediaTokenGrant(cfg.tokenHashes),
		"--tls-cert", certificatePath,
		"--tls-key", privateKeyPath,
		"--bootstrap-receipt", receiptEnvelopePath,
	)
	cmd.Env = childEnvironment()
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return startChild("cozy-media", cmd)
}

// mediaTokenGrant spells the validated hash set the way cozy-media takes it. These are
// DIGESTS, not credentials — the raw token exists only on the renter's host — so the
// process list carries nothing an observer could present.
func mediaTokenGrant(hashes []string) string {
	lines := make([]string, len(hashes))
	for i, hash := range hashes {
		lines[i] = "sha256:" + hash
	}
	return strings.Join(lines, ",")
}

func startAdapter(cfg config) (*child, error) {
	cmd := exec.Command(adapterPath)
	env := childEnvironment()
	hashes, _ := json.Marshal(cfg.tokenHashes)
	env = append(env,
		"COZY_ADAPTER_ACQUISITION_ATTEMPT_ID="+cfg.attemptID,
		"COZY_ADAPTER_ACQUISITION_ATTEMPT_ORDINAL="+strconv.FormatInt(cfg.attemptOrdinal, 10),
		"COZY_ADAPTER_RENTAL_ID="+cfg.rentalID,
		"COZY_ADAPTER_POD_BOOT_ID_PATH="+podBootIDPath,
		"COZY_ADAPTER_TLS_CERT_PATH="+certificatePath,
		"COZY_ADAPTER_TLS_KEY_PATH="+privateKeyPath,
		"COZY_ADAPTER_RECEIPT_PAYLOAD_PATH="+receiptPayloadPath,
		"COZY_ADAPTER_WORKER_INTERNAL_PORT="+strconv.Itoa(int(cfg.workerPort)),
		"COZY_ADAPTER_MEDIA_INTERNAL_PORT="+strconv.Itoa(int(cfg.mediaPort)),
		"COZY_ADAPTER_RENTER_TOKEN_SHA256_JSON="+string(hashes),
		"COZY_ADAPTER_READINESS_DEADLINE_UNIX="+strconv.FormatInt(cfg.deadline.Unix(), 10),
	)
	cmd.Env = env
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return startChild("materialize/launch adapter", cmd)
}

func startChild(name string, cmd *exec.Cmd) (*child, error) {
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start %s: %w", name, err)
	}
	c := &child{name: name, cmd: cmd, done: make(chan struct{})}
	go func() {
		c.err = cmd.Wait()
		close(c.done)
	}()
	return c, nil
}

func publishReceipt(ctx context.Context, key []byte, children []*child) error {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		payload, found, err := readReceiptPayload()
		if err != nil {
			return err
		}
		if found {
			mac := hmac.New(sha256.New, key)
			_, _ = mac.Write(receiptDomain)
			_, _ = mac.Write(payload)
			envelope, err := json.Marshal(struct {
				Payload    []byte `json:"payload"`
				HMACSHA256 string `json:"hmac_sha256"`
			}{Payload: payload, HMACSHA256: hex.EncodeToString(mac.Sum(nil))})
			if err != nil {
				return fmt.Errorf("encode readiness envelope: %w", err)
			}
			if len(envelope) > mediawire.MaxReceiptBytes {
				return fmt.Errorf("readiness envelope exceeds the media plane's published ceiling")
			}
			return atomicWrite(receiptEnvelopePath, envelope, 0o400)
		}
		for _, c := range children {
			select {
			case <-c.done:
				return childExit(c.name, c.err)
			default:
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func readReceiptPayload() ([]byte, bool, error) {
	return readStableRegularFile(receiptPayloadPath, mediawire.MaxReceiptBytes)
}

func readStableRegularFile(path string, maxBytes int64) ([]byte, bool, error) {
	pathInfo, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("inspect adapter readiness payload path: %w", err)
	}
	if !pathInfo.Mode().IsRegular() {
		return nil, false, fmt.Errorf("adapter readiness payload path is not one regular file")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, false, fmt.Errorf("open adapter readiness payload: %w", err)
	}
	defer file.Close()
	openInfo, err := file.Stat()
	if err != nil {
		return nil, false, fmt.Errorf("inspect opened adapter readiness payload: %w", err)
	}
	if !openInfo.Mode().IsRegular() || !os.SameFile(pathInfo, openInfo) {
		return nil, false, fmt.Errorf("adapter readiness payload path changed while it was opened")
	}
	payload, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, false, fmt.Errorf("read adapter readiness payload: %w", err)
	}
	if len(payload) == 0 || int64(len(payload)) > maxBytes {
		return nil, false, fmt.Errorf("adapter readiness payload is empty or exceeds its envelope bound")
	}
	return payload, true, nil
}

func supervise(ctx context.Context, children []*child) error {
	for {
		for _, c := range children {
			select {
			case <-c.done:
				return childExit(c.name, c.err)
			default:
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func childExit(name string, err error) error {
	if err == nil {
		return fmt.Errorf("%s exited unexpectedly", name)
	}
	return fmt.Errorf("%s failed: %w", name, err)
}

func stopChildren(children []*child) {
	for _, c := range children {
		if c != nil && c.cmd.Process != nil {
			_ = c.cmd.Process.Signal(syscall.SIGTERM)
		}
	}
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	for _, c := range children {
		if c == nil {
			continue
		}
		select {
		case <-c.done:
		case <-deadline.C:
			for _, remaining := range children {
				select {
				case <-remaining.done:
				default:
					_ = remaining.cmd.Process.Kill()
				}
			}
			return
		}
	}
}

func atomicWrite(target string, data []byte, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return err
	}
	temp, err := os.CreateTemp(filepath.Dir(target), ".atomic-*")
	if err != nil {
		return err
	}
	tempName := temp.Name()
	defer os.Remove(tempName)
	if err := temp.Chmod(mode); err != nil {
		temp.Close()
		return err
	}
	if _, err := temp.Write(data); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempName, target); err != nil {
		return err
	}
	return syncDirectory(filepath.Dir(target))
}

func syncDirectory(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}

func classifyContext(parent, readiness context.Context, fallback error) error {
	if parent.Err() != nil {
		return parent.Err()
	}
	if errors.Is(readiness.Err(), context.DeadlineExceeded) {
		return errors.New("frozen readiness deadline elapsed")
	}
	return fallback
}
