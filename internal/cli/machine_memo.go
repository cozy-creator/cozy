package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// answerMemoLookups answers each memo.lookup in a page from the owner's own completed
// results, before the observer reads again: a machine that asked waits only until then. The
// results never leave this machine; a Hub only confirms that a recorded checkpoint still
// exists. A machine without memo_lookup is never answered.
func (m *machineRuns) answerMemoLookups(ctx context.Context, request records.Request, connection *machineConnection,
	query *pb.MachineExecutionQuery, page *pb.MachineExecutionEventPage,
) *exit.Error {
	for _, event := range page.Events {
		if event.GetKind() != "memo.lookup" {
			continue
		}
		workspace, problem := m.workspace(ctx, connection)
		if problem != nil || !workspace.MemoLookup {
			return problem
		}
		var lookup struct {
			Operation string `json:"operation"`
			Key       string `json:"computation_digest"`
		}
		_ = json.Unmarshal(event.BodyCanonicalBytes, &lookup)
		digest, err := hex.DecodeString(strings.TrimPrefix(lookup.Key, "sha256:"))
		if err != nil || len(digest) != sha256.Size || lookup.Operation == "" {
			continue
		}
		result, problem := m.ownerMemoHit(ctx, request.Hub, lookup.Operation, lookup.Key)
		if problem != nil {
			return problem
		}
		// A refusal (the lookup already settled as a miss) changes nothing here.
		_, _ = connection.Host.ControlMachineExecution(ctx, &pb.MachineExecutionControl{
			Execution: query, CommandId: fmt.Sprintf("memo-%d", event.Sequence),
			Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_ANSWER_MEMO,
			Memo:   &pb.MachineMemoAnswer{LookupSequence: event.Sequence, ComputationDigest: digest, ResultCanonicalBytes: result},
		})
	}
	return nil
}

// ownerMemoHit is the newest recorded result for the key whose checkpoint the owner's Hub
// still serves with the recorded manifest, or nil: a miss.
func (m *machineRuns) ownerMemoHit(ctx context.Context, origin, operation, key string) ([]byte, *exit.Error) {
	results, problem := m.store.OwnerMemoResults(origin, operation, key)
	if problem != nil {
		return nil, problem
	}
	owner := client(m.context.forHub(origin))
	for _, result := range results {
		var checkpoint struct {
			Destination string `json:"destination"`
			Checkpoint  string `json:"checkpoint"`
			Manifest    struct {
				Digest string `json:"digest"`
				Length int64  `json:"length"`
			} `json:"manifest"`
		}
		if json.Unmarshal(result, &checkpoint) != nil || checkpoint.Checkpoint == "" {
			continue
		}
		ref, problem := hub.ParseRef(checkpoint.Destination)
		if problem != nil {
			continue
		}
		manifest, problem := owner.CheckpointManifest(ctx, ref, checkpoint.Checkpoint)
		sum := sha256.Sum256(manifest)
		if problem == nil && int64(len(manifest)) == checkpoint.Manifest.Length &&
			checkpoint.Manifest.Digest == "sha256:"+hex.EncodeToString(sum[:]) {
			return result, nil
		}
	}
	return nil, nil
}
