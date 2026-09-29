package cli

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
)

type reopenTimingHost struct {
	pb.PodHostClient
	installed *pb.InstalledPackage
	calls     int
}

func (h *reopenTimingHost) PrepareLocalPackage(context.Context, *pb.PrepareLocalPackageCall, ...grpc.CallOption) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
	h.calls++
	return &reopenTimingStream{installed: h.installed}, nil
}

type reopenTimingStream struct {
	grpc.ClientStream
	installed *pb.InstalledPackage
}

func (s *reopenTimingStream) Recv() (*pb.PrepareEvent, error) {
	// A retained installation can still take time to answer. Its sole PREPARED
	// event supplies no intermediate progress from which to infer that wait.
	time.Sleep(20 * time.Millisecond)
	return &pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, InstalledPackage: s.installed}, nil
}

func TestPackageReopenRecordsElapsedTimeAndPreservesRefusal(t *testing.T) {
	for _, invalid := range []bool{false, true} {
		t.Run(map[bool]string{false: "reused", true: "invalid_reply"}[invalid], func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
			if problem != nil {
				t.Fatal(problem)
			}
			defer store.Close()
			request := records.Request{ID: "req-timing", IdemKey: "timing", BodyDigest: "sha256:" + strings.Repeat("1", 64), Package: "local/timing", Entrypoint: "generate", Payload: []byte("{}")}
			if _, _, problem := store.Submit(request); problem != nil {
				t.Fatal(problem)
			}
			revision := localpackage.Installation{ID: "installation-timing", Package: request.Package, Release: "1.0.0", Files: []localpackage.File{{Filename: "source.tar", Kind: "source", Length: 1}}}
			installed := &pb.InstalledPackage{InstallationId: revision.ID, Package: revision.Package, Release: revision.Release, PackageInterface: []byte("{}")}
			if invalid {
				installed.InstallationId = "another-installation"
			}
			host := &reopenTimingHost{installed: installed}
			connection := &machineConnection{Machine: &machines.Machine{Host: host, Claim: &pb.Claim{WorkerBootId: "boot"}}, runs: &machineRuns{store: store}, installed: map[string]*pb.InstalledPackage{}, placements: map[string]*pb.DesiredPlacementSet{}}
			problem = connection.prepare(context.Background(), request.ID, revision)
			if (problem != nil) != invalid {
				t.Fatalf("invalid=%v, result=%v", invalid, problem)
			}
			if host.calls != 1 {
				t.Fatalf("reopen made %d preparation calls", host.calls)
			}
			if (connection.installed[revision.ID] != nil) == invalid {
				t.Fatal("timing changed installation acceptance")
			}
			events, problem := store.EventsAfter(request.ID, 0, 100)
			if problem != nil {
				t.Fatal(problem)
			}
			count := 0
			for _, event := range events {
				if event.Type != "request.preparing" || event.Payload["stage"] != "package preparation total" {
					continue
				}
				count++
				ms, ok := event.Payload["ms"].(float64)
				if !ok || ms < 20 {
					t.Fatalf("stream wait absent from elapsed evidence: %v", event.Payload)
				}
				detail, _ := event.Payload["detail"].(string)
				if !strings.Contains(detail, "reopen") || !strings.Contains(detail, "including nested") || strings.Contains(detail, "refused") != invalid {
					t.Fatalf("ambiguous preparation outcome: %v", event.Payload)
				}
				if event.Payload["started_unix_ms"] == nil {
					t.Fatal("elapsed interval has no start")
				}
			}
			if count != 1 {
				t.Fatalf("got %d elapsed events", count)
			}
		})
	}
}
