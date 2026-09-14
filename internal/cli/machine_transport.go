package cli

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/processtree"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

func (m *machineRuns) connect(ctx context.Context, machine string) (*machineConnection, *exit.Error) {
	if machine == "local" {
		return m.connectLocalMachine(ctx)
	}
	target, problem := rental.Resolver(m.layout, m.store)(machine)
	if problem != nil {
		return nil, problem
	}
	identity := target.Connection
	pin, err := workertls.LoadPin(identity.CACert)
	if err != nil {
		return nil, exit.New(exit.Credential, "machine TLS identity cannot be read")
	}
	connection, err := grpc.NewClient(identity.Addr, grpc.WithTransportCredentials(credentials.NewTLS(pin.TLSConfig())), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(16<<20), grpc.MaxCallSendMsgSize(16<<20)))
	if err != nil {
		return nil, machineTransport(err)
	}
	host := pb.NewPodHostClient(connection)
	proof, problem := rental.ClaimProof(m.layout)(identity, 1)
	if problem != nil {
		connection.Close()
		return nil, problem
	}
	claim := &pb.Claim{RecordOwnerEpoch: 1, RecordOwnerId: "cozy-local-client", WorkerId: identity.WorkerID, WorkerBootId: identity.WorkerBootID, WireMinor: pb.WireMinor, Proof: proof}
	probeContext, cancel := context.WithTimeout(ctx, 10*time.Second)
	info, err := host.ProtocolInfo(probeContext, &pb.ProtocolInfoRequest{})
	cancel()
	if err != nil || info.WireMinor < 51 {
		connection.Close()
		if err != nil {
			return nil, machineTransport(err)
		}
		return nil, exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "Runtime-owned execution requires worker protocol 51; this worker reports %d", info.WireMinor)
	}
	m.mu.Lock()
	alreadyClaimed := m.claimed[machine] == claim.WorkerBootId
	m.mu.Unlock()
	if !alreadyClaimed {
		if problem := authenticateMachine(ctx, pb.NewWorkerControlClient(connection), claim); problem != nil {
			connection.Close()
			return nil, problem
		}
		m.mu.Lock()
		m.claimed[machine] = claim.WorkerBootId
		m.mu.Unlock()
	}
	result := &machineConnection{connection: connection, client: host, claim: claim, wireMinor: info.WireMinor, certificateDigest: pin.Digest()}
	result.preparePublished = func(ctx context.Context, request records.Request) (*pb.DesiredPlacementSet, *exit.Error) {
		ref := &pb.DownloadPackageRef{Package: request.Package, Release: request.Release}
		facts, problem := rental.PrepareFactsSource(client(m.context))(ctx, identity, ref)
		if problem != nil {
			return nil, problem
		}
		result.publicOrigin = rental.PublicOrigin(facts.LockedRequirements, request.Package)
		downloads, problem := rental.DownloadSet([]*pb.DownloadPackageRef{ref}, nil)
		if problem != nil {
			return nil, problem
		}
		stream, err := host.PreparePackageSet(ctx, &pb.PreparePackageSetCall{
			Claim: claim, PackageSet: &pb.DesiredPackageSet{DownloadDelegation: downloads},
			Application: facts.Application, ModelSlotPaths: facts.ModelSlotPaths,
			ImageInventory: facts.ImageInventory, LockedRequirements: facts.LockedRequirements,
		})
		if err != nil {
			return nil, machineTransport(err)
		}
		return readMachinePreparedSet(stream)
	}

	result.retainModel = func(ctx context.Context, request *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error) {
		return host.RetainDerivedResult(ctx, &pb.DerivedRetentionCall{Claim: claim, Request: request})
	}
	result.releaseModel = func(ctx context.Context, request *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error) {
		return host.ReleaseDerivedRetention(ctx, &pb.DerivedRetentionCall{Claim: claim, Request: request})
	}
	result.retainBytes = func(ctx context.Context, request *pb.NativeByteRetentionRequest) (*pb.NativeByteRetentionResult, error) {
		return host.RetainByteTree(ctx, &pb.NativeByteRetentionCall{Claim: claim, Request: request})
	}
	result.releaseBytes = func(ctx context.Context, request *pb.NativeByteRetentionRequest) (*pb.NativeByteRetentionResult, error) {
		return host.ReleaseByteTree(ctx, &pb.NativeByteRetentionCall{Claim: claim, Request: request})
	}
	result.importInputTree = func(ctx context.Context) (grpc.ClientStreamingClient[pb.InputTreeImportFrame, pb.NativeByteRetentionResult], error) {
		return host.ImportInputTree(ctx)
	}
	result.readBytes = func(ctx context.Context, source *pb.NativeByteRetentionRequest, object *pb.Ref) (machineByteStream, error) {
		return host.ReadByteTreeObject(ctx, &pb.NativeByteReadCall{Claim: claim, Source: source, Object: object})
	}
	result.prepare = func(ctx context.Context, request string, revision localpackage.Revision) *exit.Error {
		uploaded, problem := m.store.MachinePackageUploaded(request, claim.WorkerBootId, revision.Digest)
		if problem != nil {
			return problem
		}
		operation := machinePackageOperation(request, revision)
		if !uploaded {
			// A prior request's Host receipt does not establish this Runtime's
			// process-local prepared registry after a worker restart. New roots use
			// their own normal upload/prepare; only this root's transfer may replay.
			if problem := uploadMachinePackage(ctx, host, claim, operation, revision); problem != nil {
				return problem
			}
			// Preparation may remove wheel carriers after installing their bytes.
			// Freeze the verified upload before crossing that boundary, so recovery
			// replays preparation rather than trying to reopen a terminal upload.
			if problem := m.store.AppendEvent(request, "machine.package_uploaded", 0, map[string]any{"worker_boot_id": claim.WorkerBootId, "revision": revision.Digest}); problem != nil {
				return problem
			}
		}
		selected, problem := orchestrator.LocalPackageSelection(operation, revision)
		if problem != nil {
			return problem
		}
		stream, err := host.PrepareLocalPackage(ctx, &pb.PrepareLocalPackageCall{Claim: claim, LocalPackageSet: selected})
		if err != nil {
			return machineTransport(err)
		}
		return readMachinePreparation(stream)
	}
	return result, nil
}

func machinePackageOperation(request string, revision localpackage.Revision) string {
	return request + "." + strings.TrimPrefix(revision.Digest, "sha256:")
}

// The ordinary Host upload has durable verified prefixes. Private code moves
// directly from this client to its machine, without a package repository.
func uploadMachinePackage(ctx context.Context, host pb.PodHostClient, claim *pb.Claim, operation string, revision localpackage.Revision) *exit.Error {
	source, err := canonical.Raw(revision.SourceDigest)
	if err != nil {
		return exit.New(exit.Conflict, "captured source identity is invalid")
	}
	for _, file := range revision.Files {
		digest, err := canonical.Raw(file.Digest)
		if err != nil {
			return exit.New(exit.Conflict, "captured wheel identity is invalid")
		}
		stream, err := host.LocalPackageUpload(ctx)
		if err != nil {
			return machineTransport(err)
		}
		header := &pb.LocalPackageUploadHeader{Claim: claim, OperationId: operation, SourceDigest: source, File: &pb.LocalPackageFileRef{Digest: digest, Length: uint64(file.Length), Filename: file.Filename}}
		if err := stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: header}}); err != nil {
			stream.CloseSend()
			return machineTransport(err)
		}
		held, err := stream.Recv()
		if err != nil {
			stream.CloseSend()
			return machineTransport(err)
		}
		if held.ReceivedBytes > uint64(file.Length) {
			stream.CloseSend()
			return exit.New(exit.Conflict, "machine upload returned an impossible verified prefix")
		}
		if held.State == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED {
			stream.CloseSend()
			continue
		}
		if held.State == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED {
			stream.CloseSend()
			return exit.New(exit.Conflict, "machine refused captured wheel: %s", held.SafeDetail)
		}
		reader, err := os.Open(file.Path)
		if err != nil {
			stream.CloseSend()
			return exit.New(exit.NotFound, "captured wheel is unavailable: %s", err)
		}
		offset := held.ReceivedBytes
		if _, err := reader.Seek(int64(offset), io.SeekStart); err != nil {
			reader.Close()
			stream.CloseSend()
			return exit.Internalf("cannot resume captured wheel upload: %s", err)
		}
		buffer := make([]byte, 1<<20)
		for offset < uint64(file.Length) {
			n, err := io.ReadFull(reader, buffer[:min(uint64(len(buffer)), uint64(file.Length)-offset)])
			if err != nil {
				reader.Close()
				stream.CloseSend()
				return exit.New(exit.Conflict, "captured wheel changed during upload: %s", err)
			}
			if err := stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Chunk{Chunk: &pb.LocalPackageUploadChunk{Offset: offset, Data: buffer[:n]}}}); err != nil {
				reader.Close()
				stream.CloseSend()
				return machineTransport(err)
			}
			next, err := stream.Recv()
			if err != nil {
				reader.Close()
				stream.CloseSend()
				return machineTransport(err)
			}
			offset += uint64(n)
			if next.ReceivedBytes != offset {
				reader.Close()
				stream.CloseSend()
				return exit.New(exit.Conflict, "machine upload changed its acknowledged prefix")
			}
			held = next
		}
		reader.Close()
		stream.CloseSend()
		if held.State != pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED {
			return exit.New(exit.Conflict, "captured wheel has no verified upload completion")
		}
	}
	return nil
}

type localMachineProcess struct {
	PID      int    `json:"pid"`
	WorkerID string `json:"worker_id"`
}

func (m *machineRuns) connectLocalMachine(ctx context.Context) (*machineConnection, *exit.Error) {
	if runtime.GOOS == "windows" {
		return nil, exit.Named(exit.Structural, "machine_execution.local_platform_unsupported", "Runtime-owned local execution requires the local Runtime IPC launcher on this platform")
	}
	m.localMu.Lock()
	defer m.localMu.Unlock()
	root := filepath.Join(m.layout.Root, "runtime")
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, exit.Internalf("cannot create local Runtime state: %s", err)
	}
	bootstrapPath := filepath.Join(root, "bootstrap")
	bootstrap, err := os.ReadFile(bootstrapPath)
	if os.IsNotExist(err) {
		random := make([]byte, 32)
		if _, err := rand.Read(random); err != nil {
			return nil, exit.Internalf("cannot mint local Runtime authentication: %s", err)
		}
		bootstrap = []byte(base64.RawURLEncoding.EncodeToString(random))
		if err := os.WriteFile(bootstrapPath, bootstrap, 0o600); err != nil {
			return nil, exit.Internalf("cannot retain local Runtime authentication: %s", err)
		}
	} else if err != nil || len(bootstrap) != 43 {
		return nil, exit.New(exit.Credential, "local Runtime authentication is unreadable")
	}
	workerID := "local-" + strings.TrimPrefix(spellLocalDigest([]byte(m.layout.Root)), "sha256:")[:24]
	socket := filepath.Join(root, "control.sock")
	processPath := filepath.Join(root, "process.json")
	var process localMachineProcess
	if raw, err := os.ReadFile(processPath); err == nil {
		if json.Unmarshal(raw, &process) != nil || process.WorkerID != workerID {
			return nil, exit.New(exit.Conflict, "local Runtime process record has a different identity")
		}
	}
	if process.PID == 0 || !processtree.Alive(process.PID) {
		tool, problem := hostruntime.Path(m.context.Cfg.Tool())
		if problem != nil {
			return nil, problem
		}
		resolved, err := filepath.EvalSymlinks(tool)
		if err != nil {
			return nil, exit.New(exit.NotFound, "local Runtime tool cannot be resolved")
		}
		python := filepath.Join(filepath.Dir(resolved), "python")
		if _, err := os.Stat(python); err != nil {
			return nil, exit.Named(exit.Structural, "machine_execution.runtime_python_missing", "the installed Runtime must provide its environment Python beside cozy-runtime")
		}
		backend, devices := "none", strings.Join(m.resolver.Devices, ",")
		if devices != "" {
			backend = "cuda"
		}
		command := exec.Command(tool, "serve", "--socket", socket, "--out", root,
			"--worker-id", workerID, "--install-root", filepath.Join(root, "environments"),
			"--artifact-cache", filepath.Join(root, "artifacts"), "--tensorfs-root", m.context.Cfg.TensorFSRoot,
			"--environment-python", python, "--accelerator-backend", backend, "--devices", devices,
			"--grant-root", filepath.Join(root, "grants"))
		command.Env = m.context.Cfg.Child("COZY_HOME="+root, "COZY_BOOTSTRAP_CREDENTIAL="+string(bootstrap), "CUDA_VISIBLE_DEVICES="+devices, "COZY_DEPENDENCY_CACHE="+m.layout.DependencyCache())
		command.Dir = root
		log, err := os.OpenFile(filepath.Join(root, "worker.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			return nil, exit.Internalf("cannot open local Runtime log: %s", err)
		}
		// Direct inherited descriptors, not client-owned pipes: Runtime survives
		// this client daemon disconnecting or exiting.
		command.Stdout, command.Stderr = log, log
		processtree.Prepare(command)
		if err := command.Start(); err != nil {
			log.Close()
			return nil, exit.Internalf("cannot start local Runtime: %s", err)
		}
		log.Close()
		process = localMachineProcess{PID: command.Process.Pid, WorkerID: workerID}
		raw, _ := json.Marshal(process)
		if err := os.WriteFile(processPath, raw, 0o600); err != nil {
			return nil, exit.Internalf("cannot retain local Runtime process identity: %s", err)
		}
		go func() { _ = command.Wait() }()
	}
	connection, err := grpc.NewClient("unix://"+socket, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(16<<20), grpc.MaxCallSendMsgSize(16<<20)))
	if err != nil {
		return nil, machineTransport(err)
	}
	preparation := pb.NewRuntimePreparationClient(connection)
	ready, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	info, err := preparation.ProtocolInfo(ready, &pb.ProtocolInfoRequest{}, grpc.WaitForReady(true))
	if err != nil || info.WireMinor < 51 {
		connection.Close()
		if err != nil {
			return nil, machineTransport(err)
		}
		return nil, exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "local Runtime requires protocol 51; installed Runtime reports %d", info.WireMinor)
	}
	client := pb.NewWorkerControlClient(connection)
	claim := &pb.Claim{RecordOwnerEpoch: 1, RecordOwnerId: "cozy-local-client", WorkerId: workerID, WireMinor: pb.WireMinor, Proof: bootstrap}
	if m.localPID != process.PID {
		if problem := authenticateMachine(ctx, client, claim); problem != nil {
			connection.Close()
			return nil, problem
		}
		m.localPID = process.PID
	}
	result := &machineConnection{connection: connection, client: client, claim: claim, wireMinor: info.WireMinor}
	result.preparePublished = func(ctx context.Context, request records.Request) (*pb.DesiredPlacementSet, *exit.Error) {
		facts, problem := m.resolver.installFacts(request.InstallID)
		if problem != nil {
			return nil, problem
		}
		if facts.Install.SourceKind != "tensorhub" || facts.Install.Package != request.Package || facts.Install.Version != request.Release {
			return nil, exit.New(exit.Conflict, "published local preparation changed its immutable install")
		}
		spec, problem := facts.PreparationSpec(m.resolver.Devices)
		if problem != nil {
			return nil, problem
		}
		locked, err := os.ReadFile(spec.Preparation.LockedRequirements)
		if err != nil {
			return nil, exit.Internalf("cannot read retained published requirements: %s", err)
		}
		downloads, problem := rental.DownloadSet([]*pb.DownloadPackageRef{{Package: request.Package, Release: request.Release}}, nil)
		if problem != nil {
			return nil, problem
		}
		prepared, err := preparation.PreparePackageSet(ctx, &pb.PreparePackageSetRequest{
			InstallRoot: filepath.Join(root, "environments"), DownloadDelegation: downloads,
			Application: facts.PackageInterface.Application, LockedRequirements: locked,
			ModelSlotPaths: spec.Preparation.ModelSlotPaths,
		})
		if err != nil {
			return nil, machineTransport(err)
		}
		return prepared.PlacementSet, validateMachinePrepared(prepared.PlacementSet)
	}
	result.retainModel = func(ctx context.Context, request *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error) {
		return preparation.WorkspaceRetainDerivedResult(ctx, &pb.DerivedRetentionCall{Claim: claim, Request: request})
	}
	result.releaseModel = func(ctx context.Context, request *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error) {
		return preparation.WorkspaceReleaseDerivedRetention(ctx, &pb.DerivedRetentionCall{Claim: claim, Request: request})
	}
	result.retainBytes = func(ctx context.Context, request *pb.NativeByteRetentionRequest) (*pb.NativeByteRetentionResult, error) {
		return preparation.WorkspaceRetainByteTree(ctx, &pb.NativeByteRetentionCall{Claim: claim, Request: request})
	}
	result.releaseBytes = func(ctx context.Context, request *pb.NativeByteRetentionRequest) (*pb.NativeByteRetentionResult, error) {
		return preparation.WorkspaceReleaseByteTree(ctx, &pb.NativeByteRetentionCall{Claim: claim, Request: request})
	}
	result.importInputTree = func(ctx context.Context) (grpc.ClientStreamingClient[pb.InputTreeImportFrame, pb.NativeByteRetentionResult], error) {
		return preparation.ImportInputTree(ctx)
	}
	result.readBytes = func(ctx context.Context, source *pb.NativeByteRetentionRequest, object *pb.Ref) (machineByteStream, error) {
		return preparation.WorkspaceReadByteTreeObject(ctx, &pb.NativeByteReadCall{Claim: claim, Source: source, Object: object})
	}
	result.prepare = func(ctx context.Context, requestID string, revision localpackage.Revision) *exit.Error {
		selected, problem := orchestrator.LocalPackageSelection(machinePackageOperation(requestID, revision), revision)
		if problem != nil {
			return problem
		}
		request := &pb.PrepareLocalPackageRequest{OperationId: selected.OperationId, Package: selected.Package, InstallRoot: filepath.Join(root, "environments")}
		wheelRoot := filepath.Join(request.InstallRoot, ".stage", selected.OperationId, "wheels")
		if err := os.MkdirAll(wheelRoot, 0o700); err != nil {
			return exit.Internalf("cannot stage local Runtime wheels: %s", err)
		}
		for _, file := range revision.Files {
			path, problem := stageMachineWheel(wheelRoot, file)
			if problem != nil {
				return problem
			}
			digest, _ := canonical.Raw(file.Digest)
			request.Wheels = append(request.Wheels, &pb.LocalPackageWheel{Digest: digest, Filename: file.Filename, Length: uint64(file.Length), Path: path})
		}
		prepared, err := preparation.PrepareLocalPackage(ctx, request)
		if err != nil {
			return machineTransport(err)
		}
		return validateMachinePrepared(prepared.PlacementSet)
	}
	return result, nil
}

func authenticateMachine(ctx context.Context, client pb.WorkerControlClient, claim *pb.Claim) *exit.Error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	stream, err := client.Control(ctx)
	if err != nil {
		return machineTransport(err)
	}
	defer stream.CloseSend()
	if err := stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}); err != nil {
		return machineTransport(err)
	}
	for {
		frame, err := stream.Recv()
		if err != nil {
			return machineTransport(err)
		}
		ack := frame.GetClaimAck()
		if ack == nil {
			continue
		}
		if !ack.Accepted || ack.WorkerId != claim.WorkerId || ack.WorkerBootId == "" || claim.WorkerBootId != "" && claim.WorkerBootId != ack.WorkerBootId {
			return exit.New(exit.Credential, "Runtime refused the expected machine identity")
		}
		claim.WorkerBootId = ack.WorkerBootId
		return nil // no SnapshotAck, DesiredState, or AttemptOffer on this stream
	}
}

func stageMachineWheel(root string, file localpackage.File) (string, *exit.Error) {
	if filepath.Base(file.Filename) != file.Filename {
		return "", exit.New(exit.Conflict, "captured wheel filename is not one component")
	}
	target := filepath.Join(root, file.Filename)
	reader, err := os.Open(file.Path)
	if err != nil {
		return "", exit.New(exit.NotFound, "captured local wheel is unavailable: %s", err)
	}
	defer reader.Close()
	output, err := os.CreateTemp(root, ".wheel-")
	if err != nil {
		return "", exit.Internalf("cannot stage local wheel: %s", err)
	}
	defer os.Remove(output.Name())
	defer output.Close()
	hash := sha256.New()
	count, err := io.Copy(io.MultiWriter(output, hash), reader)
	if err != nil || count != file.Length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != file.Digest {
		return "", exit.New(exit.Conflict, "captured wheel changed before local preparation")
	}
	if err := output.Sync(); err != nil {
		return "", exit.Internalf("cannot retain local wheel: %s", err)
	}
	if err := output.Close(); err != nil {
		return "", exit.Internalf("cannot close local wheel: %s", err)
	}
	if err := os.Rename(output.Name(), target); err != nil {
		return "", exit.Internalf("cannot finalize local wheel staging: %s", err)
	}
	return target, nil
}

func spellLocalDigest(raw []byte) string {
	spelling, _ := canonical.Spell(canonical.Digest(raw))
	return spelling
}
