package rental

import (
	"crypto/rand"
	"encoding/hex"
	"sort"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const artifactDelegationTTL = time.Hour

// ArtifactDelegations derives every logical identity from Tensorhub's exact persisted
// control snapshot. The orchestrator supplies only worker/boot facts learned on ClaimAck.
func ArtifactDelegations(l home.Layout, st *records.Store) orchestrator.ArtifactDelegationSource {
	return func(connection *orchestrator.WorkerConnection,
		request orchestrator.ArtifactDelegationRequest) (orchestrator.ArtifactDelegation, *exit.Error) {
		if connection == nil || connection.RentalID == "" || request.WorkerID == "" ||
			request.WorkerBootID == "" {
			return orchestrator.ArtifactDelegation{}, exit.Named(exit.Credential,
				"rental.artifact_delegation_authority_missing",
				"artifact delegation requires one rental, worker, boot, and Creator identity")
		}
		row, problem := st.RentalRow(connection.RentalID)
		if problem != nil {
			return orchestrator.ArtifactDelegation{}, problem
		}
		if row == nil {
			return orchestrator.ArtifactDelegation{}, unknown(connection.RentalID)
		}
		facts, problem := controlFacts(*row)
		if problem != nil {
			return orchestrator.ArtifactDelegation{}, problem
		}
		if row.PlacementRevision == 0 || facts.Placement.PackageReleaseID == "" {
			return orchestrator.ArtifactDelegation{}, exit.Named(exit.Conflict,
				"rental.artifact_delegation_identity_missing",
				"the rental control snapshot has no exact package or placement revision")
		}
		identity, problem := CreatorIdentityFor(l, row.ID)
		if problem != nil {
			return orchestrator.ArtifactDelegation{}, problem
		}
		pin, err := workertls.LoadPin(connection.CACert)
		if err != nil {
			return orchestrator.ArtifactDelegation{}, exit.New(exit.Credential,
				"the worker certificate pin is unreadable: %s", err)
		}
		models := append([]string(nil), facts.ModelRootDigests...)
		sort.Strings(models)
		for index := 1; index < len(models); index++ {
			if models[index] == models[index-1] {
				return orchestrator.ArtifactDelegation{}, exit.Named(exit.Conflict,
					"rental.artifact_delegation_identity_duplicate",
					"the rental control snapshot repeats model checkpoint %s", models[index])
			}
		}
		delegationID, problem := mintDelegationID()
		if problem != nil {
			return orchestrator.ArtifactDelegation{}, problem
		}
		expires := uint64(time.Now().Add(artifactDelegationTTL).Unix())
		document := &pb.ArtifactDelegation{
			RentalId: row.ID, WorkerId: request.WorkerID, WorkerBootId: request.WorkerBootID,
			WorkerTlsCertificateDigest: pin.Digest(), Revision: row.PlacementRevision,
			PackageReleaseIds:  []string{facts.Placement.PackageReleaseID},
			ModelCheckpointIds: models, DelegationId: delegationID, ExpiresAtUnix: expires,
		}
		canonicalBytes, err := canonical.Bytes(document)
		if err != nil {
			return orchestrator.ArtifactDelegation{}, exit.Internalf(
				"cannot author the artifact delegation: %s", err)
		}
		return orchestrator.ArtifactDelegation{
			CanonicalBytes: canonicalBytes, Signature: identity.Sign(canonicalBytes),
			DelegationID: delegationID, Revision: row.PlacementRevision, ExpiresAtUnix: expires,
		}, nil
	}
}

func mintDelegationID() (string, *exit.Error) {
	var random [16]byte
	if _, err := rand.Read(random[:]); err != nil {
		return "", exit.Internalf("cannot mint an artifact delegation id: %s", err)
	}
	return "dlg-" + hex.EncodeToString(random[:]), nil
}
