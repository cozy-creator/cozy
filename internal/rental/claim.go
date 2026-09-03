package rental

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ClaimProof reconstructs worker-protocol ClaimProof/1 from the exact attach projection
// and signs it with the per-rental Creator key. That key survives the delegation
// deletion: it still signs ClaimProof, which is how a pod knows this owner.
func ClaimProof(l home.Layout) orchestrator.RentalClaimProofSource {
	return func(connection *orchestrator.WorkerConnection, epoch uint64) ([]byte, *exit.Error) {
		if connection == nil || connection.RentalID == "" || epoch == 0 ||
			connection.WorkerID == "" || connection.WorkerBootID == "" {
			return nil, exit.Named(exit.Credential, "rental.claim_identity_missing",
				"private rental Claim requires exact rental, worker, boot, and Creator identity")
		}
		identity, problem := CreatorIdentityFor(l, connection.RentalID)
		if problem != nil {
			return nil, problem
		}
		pin, err := workertls.LoadPin(connection.CACert)
		if err != nil {
			return nil, exit.New(exit.Credential, "the worker certificate pin is unreadable: %s", err)
		}
		canonicalBytes, err := canonical.Bytes(&pb.ClaimProof{
			RecordOwnerEpoch: epoch, WorkerId: connection.WorkerID,
			WorkerBootId: connection.WorkerBootID, WorkerTlsCertificateDigest: pin.Digest(),
		})
		if err != nil {
			return nil, exit.Internalf("cannot author the rental ClaimProof: %s", err)
		}
		return identity.Sign(canonicalBytes), nil
	}
}
