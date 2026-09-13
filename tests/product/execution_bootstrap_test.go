package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/executionowner"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestExecutionBootstrapRequiresExactUploadedCustody(t *testing.T) {
	capsule, problem := executionowner.Encode(executionCapsule(t))
	fatal(t, problem)
	parsed, problem := executionowner.Decode(capsule)
	fatal(t, problem)
	root := t.TempDir()
	stage := filepath.Join(root, "stage")
	must(t, os.Mkdir(stage, 0700))
	var files []string
	for id, doc := range parsed.Packages {
		name := strings.TrimPrefix(doc.Str("package"), "local/")
		dir := filepath.Join(stage, executionowner.UploadOperation(id), "wheels")
		must(t, os.MkdirAll(dir, 0700))
		path := filepath.Join(dir, doc.List("files")[0].Str("filename"))
		must(t, os.WriteFile(path, capsuleWheel(t, name), 0400))
		files = append(files, path)
	}
	public, key, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour)}
	certificate, err := x509.CreateCertificate(rand.Reader, template, template, public, key)
	must(t, err)
	certPath := filepath.Join(root, "worker.pem")
	must(t, os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificate}), 0400))
	grant := &pb.SignedExecutionOwnerGrant{Grant: &pb.ExecutionOwnerGrant{
		RecordOwnerEpoch: 1, RecordOwnerId: "pod-owner", WorkerId: "private-worker", WorkerBootId: "private-boot",
		WorkerTlsCertificateDigest: canonical.Digest(certificate), InitialCapsuleDigest: canonical.Digest(capsule), ExecutionPublicKey: public,
	}}
	grantDocument, _, err := canonical.Identity(grant.Grant)
	must(t, err)
	grant.Signature = ed25519.Sign(key, grantDocument)
	signed, err := proto.Marshal(grant)
	must(t, err)
	authority := executionowner.Authority{Format: executionowner.AuthorityFormat, SignedGrant: signed,
		CreatorHome: filepath.Join(root, "creator"), PackageStageRoot: stage, WorkerAddress: "127.0.0.1:19781",
		WorkerTLSCertificatePath: certPath, TensorFSRoot: filepath.Join(root, "tensorfs")}
	read := func(a executionowner.Authority, serving bool) (*executionowner.Bootstrap, error) {
		raw, err := json.Marshal(a)
		must(t, err)
		b, problem := executionowner.ReadBootstrap(bytes.NewReader(capsule), bytes.NewReader(raw), serving)
		if problem != nil {
			return nil, problem
		}
		return b, nil
	}
	b, err := read(authority, false)
	must(t, err)
	if len(b.Wheels) != 2 || b.Capsule.Digest != parsed.Digest {
		t.Fatal("bootstrap lost exact captured inventory")
	}
	// Exercise the fixed Creator command with inherited descriptors, the same
	// interface the trusted Host launcher uses. It starts no daemon or Python.
	meta, err := json.Marshal(authority)
	must(t, err)
	paths := []string{filepath.Join(root, "capsule.json"), filepath.Join(root, "authority.json")}
	var inherited []*os.File
	for index, body := range [][]byte{capsule, meta} {
		must(t, os.WriteFile(paths[index], body, 0400))
		file, err := os.Open(paths[index])
		must(t, err)
		defer file.Close()
		inherited = append(inherited, file)
	}
	command := exec.Command(cozyBin, "execution-owner", "validate", "--capsule-fd=3", "--authority-fd=4", "--json")
	command.Env, command.ExtraFiles = childEnv(t, filepath.Join(root, "cli")), inherited
	output, err := command.CombinedOutput()
	if err != nil || !bytes.Contains(output, []byte(`"capsule_digest":"`+parsed.Digest+`"`)) {
		t.Fatalf("fixed Creator validation command failed: %v %s", err, output)
	}
	if _, err := read(authority, true); err == nil {
		t.Fatal("serving admitted no pod private key")
	}
	authority.ExecutionPrivateKey = key
	if _, err := read(authority, false); err == nil {
		t.Fatal("validation unnecessarily received a private key")
	}
	_, err = read(authority, true)
	must(t, err)
	_, wrongKey, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	changed := authority
	changed.ExecutionPrivateKey = wrongKey
	if _, err := read(changed, true); err == nil {
		t.Fatal("foreign execution key accepted")
	}
	changed = authority
	wrongGrant := proto.Clone(grant).(*pb.SignedExecutionOwnerGrant)
	wrongGrant.Grant.InitialCapsuleDigest = canonical.Digest([]byte("different capture"))
	changed.SignedGrant, err = proto.Marshal(wrongGrant)
	must(t, err)
	if _, err := read(changed, true); err == nil {
		t.Fatal("grant accepted a different capsule")
	}
	original, err := os.ReadFile(files[0])
	must(t, err)
	must(t, os.Chmod(files[0], 0600))
	corrupt := append([]byte(nil), original...)
	corrupt[len(corrupt)/2] ^= 1
	must(t, os.WriteFile(files[0], corrupt, 0400))
	if _, err := read(authority, true); err == nil {
		t.Fatal("same-length corrupted wheel passed custody validation")
	}
	must(t, os.WriteFile(files[0], original, 0400))
	_, err = read(authority, true)
	must(t, err)
	must(t, os.Remove(files[0]))
	if _, err := read(authority, true); err == nil {
		t.Fatal("absent uploaded wheel passed custody validation")
	}
}
