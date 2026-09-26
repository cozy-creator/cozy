package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"google.golang.org/protobuf/proto"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Pinned TLS and the real owner validate complete revision bounds before upload.
func TestUnpublishedWheelCapacityBeforeTransfer(t *testing.T) {
	for _, tc := range []struct {
		name    string
		count   int
		refusal string
	}{
		{"current34", 34, ""},
		{"current129", 129, ""},
		{"too-many130", 130, "local_package_revision_invalid"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public}
			root := t.TempDir()
			connection, _ := startFakePod(t, root, pod)
			revision := stageLocalRevision(t, root)
			revision.DependencyRequirements = []byte("torch @ https://files.pythonhosted.org/torch-2.13.0-py3-none-any.whl --hash=sha256:" + strings.Repeat("a", 64) + "\n")
			for i := len(revision.Files); i < tc.count; i++ {
				name := fmt.Sprintf("helper%d-1.0.0-py3-none-any.whl", i)
				body := []byte(fmt.Sprintf("captured dependency %d", i))
				path := filepath.Join(root, name)
				must(t, os.WriteFile(path, body, 0600))
				digest, _ := canonical.Spell(sha256Of(body))
				revision.Files = append(revision.Files, localpackage.File{Digest: digest, Filename: name, Kind: "dependency", Path: path, Length: int64(len(body))})
			}
			sort.Slice(revision.Files, func(i, j int) bool { return revision.Files[i].Digest < revision.Files[j].Digest })
			o := hostOwner(t, "capacity-"+tc.name, rentalWiring(connection, private), func(opt *orchestrator.Options) { opt.Packages = localLauncher{revision: revision} })
			instance, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			problem = o.c.ConvergeLocalPackage(instance, "capacity-"+tc.name, revision, "", nil)
			if tc.refusal != "" {
				if problem == nil || problem.ErrName() != tc.refusal {
					t.Fatalf("refusal=%v; want %s", problem, tc.refusal)
				}
			} else {
				fatal(t, problem)
			}
			if tc.refusal == "" {
				waitUntil(t, "unpublished package revision prepared", func() bool {
					pod.mu.Lock()
					defer pod.mu.Unlock()
					return len(pod.localPrepares) == 1
				})
			}
			pod.mu.Lock()
			defer pod.mu.Unlock()
			if tc.refusal != "" {
				if len(pod.uploads) != 0 || len(pod.localPrepares) != 0 {
					t.Fatalf("refused revision caused upload/preparation: %d/%d", len(pod.uploads), len(pod.localPrepares))
				}
			} else if !bytes.Equal(pod.localPrepares[0].LocalPackageSet.DependencyRequirements, revision.DependencyRequirements) {
				t.Fatal("Host preparation lost the captured dependency requirements")
			} else if len(pod.uploads) != tc.count || len(pod.localPrepares) != 1 {
				t.Fatalf("got %d uploads/%d preparations; want %d/1", len(pod.uploads), len(pod.localPrepares), tc.count)
			}
		})
	}
}

// Even the maximum capability-bearing frame remains below the control ceiling.
func TestUnpublishedWheelMaximumGrantFrameFitsControl(t *testing.T) {
	request := &pb.LocalPackageFetchRequest{RecordOwnerEpoch: ^uint64(0), ControlStreamEpoch: ^uint64(0), WorkerBootId: strings.Repeat("b", 256), OperationId: strings.Repeat("o", 256)}
	for i := 0; i < pb.MaxLocalPackageFiles; i++ {
		request.Files = append(request.Files, &pb.LocalPackageFileGrant{Digest: bytes.Repeat([]byte{byte(i)}, 32), Filename: strings.Repeat("f", pb.MaxLocalPackageFilenameBytes), Length: ^uint64(0), Url: strings.Repeat("u", pb.MaxLocalPackageGrantURLBytes)})
	}
	frame := &pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_LocalPackageFetchRequest{LocalPackageFetchRequest: request}}
	encoded, err := proto.Marshal(frame)
	must(t, err)
	if len(encoded) > pb.MaxInlineControlBytes {
		t.Fatalf("%d-byte frame exceeds %d-byte control ceiling", len(encoded), pb.MaxInlineControlBytes)
	}
	t.Logf("maximum captured wheel grant frame: %d bytes of %d", len(encoded), pb.MaxInlineControlBytes)
}
