package executionowner

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"io"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// Authority is delivered only through a Host-owned descriptor. These paths are
// privileged bootstrap configuration, never members of the signed client capsule.
type Authority struct {
	Format                   string `json:"format"`
	SignedGrant              []byte `json:"signed_grant"`
	ExecutionPrivateKey      []byte `json:"execution_private_key,omitempty"`
	CreatorHome              string `json:"creator_home"`
	PackageStageRoot         string `json:"package_stage_root"`
	WorkerAddress            string `json:"worker_address"`
	WorkerTLSCertificatePath string `json:"worker_tls_certificate_path"`
	TensorFSRoot             string `json:"tensorfs_root"`
	MediaAddress             string `json:"media_address"`
	MediaBearer              string `json:"media_bearer,omitempty"`
}

type Bootstrap struct {
	Capsule   *Validated
	Authority Authority
	Grant     *pb.SignedExecutionOwnerGrant
	Wheels    map[string][]string
}

// ForSubmission is called only after the Host authenticates a new submission
// against this accepted generation. The initial grant remains unchanged; every
// new captured root receives its own ordinary request and capsule identity.
func (b *Bootstrap) ForSubmission(raw []byte) (*Bootstrap, *exit.Error) {
	capsule, problem := Decode(raw)
	if problem != nil {
		return nil, problem
	}
	next := &Bootstrap{Capsule: capsule, Authority: b.Authority, Grant: b.Grant, Wheels: map[string][]string{}}
	if problem := next.verifyWheels(); problem != nil {
		return nil, problem
	}
	return next, nil
}

// ReadBootstrap checks Host configuration, exact capture identity and every
// uploaded wheel before a coordinator may import this root. Host already verifies
// the laptop signature; Creator verifies its actual worker/capsule/key bindings.
func ReadBootstrap(capsuleReader, authorityReader io.Reader, serving bool) (*Bootstrap, *exit.Error) {
	raw, err := io.ReadAll(io.LimitReader(capsuleReader, pb.MaxInlineControlBytes+1))
	if err != nil {
		return nil, invalid("cannot read capsule descriptor")
	}
	capsule, problem := Decode(raw)
	if problem != nil {
		return nil, problem
	}
	raw, err = io.ReadAll(io.LimitReader(authorityReader, pb.MaxInlineControlBytes+1))
	if err != nil || len(raw) > pb.MaxInlineControlBytes {
		return nil, invalid("authority descriptor exceeds its bound")
	}
	var authority Authority
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&authority) != nil || decoder.Decode(&struct{}{}) != io.EOF || authority.Format != AuthorityFormat {
		return nil, invalid("authority descriptor differs from its closed schema")
	}
	for _, path := range []string{authority.CreatorHome, authority.PackageStageRoot, authority.WorkerTLSCertificatePath, authority.TensorFSRoot} {
		if !filepath.IsAbs(path) || filepath.Clean(path) != path {
			return nil, invalid("authority contains an invalid Host path")
		}
	}
	host, port, err := net.SplitHostPort(authority.WorkerAddress)
	if err != nil || port == "" || net.ParseIP(host) == nil || !net.ParseIP(host).IsLoopback() {
		return nil, invalid("execution worker address must be Host loopback")
	}
	media, err := url.Parse(authority.MediaAddress)
	if err != nil || media.Scheme != "https" || media.User != nil || media.Path != "" || media.RawQuery != "" || media.Fragment != "" ||
		media.Port() == "" || net.ParseIP(media.Hostname()) == nil || !net.ParseIP(media.Hostname()).IsLoopback() {
		return nil, invalid("execution media address must be pinned Host loopback")
	}
	grant := &pb.SignedExecutionOwnerGrant{}
	if proto.Unmarshal(authority.SignedGrant, grant) != nil || len(grant.ProtoReflect().GetUnknown()) != 0 ||
		grant.Grant == nil || len(grant.Grant.ProtoReflect().GetUnknown()) != 0 || len(grant.Signature) != ed25519.SignatureSize {
		return nil, invalid("authority omitted its exact signed execution grant")
	}
	g := grant.Grant
	if g.RecordOwnerEpoch == 0 || g.RecordOwnerId == "" || g.WorkerId == "" || g.WorkerBootId == "" ||
		len(g.ExecutionPublicKey) != ed25519.PublicKeySize || !bytes.Equal(g.InitialCapsuleDigest, canonical.Digest(capsule.Raw)) {
		return nil, invalid("execution grant differs from the initial capsule or owner")
	}
	certificate, err := os.ReadFile(authority.WorkerTLSCertificatePath)
	block, _ := pem.Decode(certificate)
	if err != nil || block == nil || block.Type != "CERTIFICATE" {
		return nil, invalid("Host TLS certificate is unavailable")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil || !bytes.Equal(g.WorkerTlsCertificateDigest, canonical.Digest(cert.Raw)) {
		return nil, invalid("execution grant names a different worker certificate")
	}
	if serving {
		if len(authority.ExecutionPrivateKey) != ed25519.PrivateKeySize || !bytes.Equal(ed25519.PrivateKey(authority.ExecutionPrivateKey).Public().(ed25519.PublicKey), g.ExecutionPublicKey) {
			return nil, invalid("coordinator key differs from its execution grant")
		}
		if len(authority.MediaBearer) < 32 || len(authority.MediaBearer) > 4096 || strings.TrimSpace(authority.MediaBearer) != authority.MediaBearer {
			return nil, invalid("coordinator media credential is unavailable")
		}
	} else if len(authority.ExecutionPrivateKey) != 0 || authority.MediaBearer != "" {
		return nil, invalid("validation must not receive execution credentials")
	}
	bootstrap := &Bootstrap{Capsule: capsule, Authority: authority, Grant: grant, Wheels: map[string][]string{}}
	if problem := bootstrap.verifyWheels(); problem != nil {
		return nil, problem
	}
	return bootstrap, nil
}

// UploadOperation is independent of laptop directory names and install IDs. It
// uses the existing PodHost operation directory and its durable upload ledger.
func UploadOperation(revision string) string {
	return "execution-" + strings.TrimPrefix(revision, "sha256:")
}

func (b *Bootstrap) verifyWheels() *exit.Error {
	root, err := os.OpenRoot(b.Authority.PackageStageRoot)
	if err != nil {
		return invalid("Host package staging is unavailable")
	}
	defer root.Close()
	for id, revision := range b.Capsule.Packages {
		projects := 0
		for _, file := range revision.List("files") {
			relative := filepath.Join(UploadOperation(id), "wheels", file.Str("filename"))
			info, err := root.Lstat(relative)
			if err != nil || !info.Mode().IsRegular() || info.Size() != file.Int("length") {
				return invalid("captured wheel is absent, partial or not a regular file")
			}
			input, err := root.Open(relative)
			if err != nil {
				return invalid("captured wheel cannot be opened")
			}
			hash := sha256.New()
			length, err := io.Copy(hash, input)
			input.Close()
			if err != nil || length != file.Int("length") || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != file.Str("digest") {
				return invalid("uploaded wheel differs from its captured identity")
			}
			path := filepath.Join(b.Authority.PackageStageRoot, relative)
			fact, problem := wheel.InspectIdentity(path)
			if problem != nil || fact.Filename != file.Str("filename") || fact.Length != length {
				return invalid("uploaded wheel has an invalid distribution identity")
			}
			if fact.Distribution == strings.TrimPrefix(revision.Str("package"), "local/") && fact.Version == revision.Str("release") {
				projects++
			}
			b.Wheels[id] = append(b.Wheels[id], path)
		}
		if projects != 1 {
			return invalid("captured revision requires exactly one matching project wheel")
		}
	}
	return nil
}
