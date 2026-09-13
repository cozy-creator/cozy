package orchestrator

import (
	"bytes"
	"crypto/ed25519"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// PrivateExecutionOwner is supplied only by the trusted pod bootstrap. The user
// script never receives it. Its key can claim only the worker/boot/generation the
// signed grant names; it is not a rental or Tensorhub account credential.
type PrivateExecutionOwner struct {
	Authorization *pb.SignedExecutionOwnerGrant
	PrivateKey    ed25519.PrivateKey
}

func freezePrivateExecution(value *PrivateExecutionOwner) (*PrivateExecutionOwner, *exit.Error) {
	if value == nil || value.Authorization == nil || value.Authorization.Grant == nil || len(value.PrivateKey) != ed25519.PrivateKeySize {
		return nil, exit.New(exit.Credential, "private execution authority is incomplete")
	}
	auth := proto.Clone(value.Authorization).(*pb.SignedExecutionOwnerGrant)
	g := auth.Grant
	if len(auth.ProtoReflect().GetUnknown()) != 0 || len(g.ProtoReflect().GetUnknown()) != 0 || len(auth.Signature) != ed25519.SignatureSize ||
		g.RecordOwnerEpoch == 0 || g.RecordOwnerEpoch > 1<<53-1 || g.RecordOwnerId == "" || len(g.RecordOwnerId) > 256 || g.WorkerId == "" || g.WorkerBootId == "" || len(g.WorkerTlsCertificateDigest) != 32 || len(g.InitialCapsuleDigest) != 32 {
		return nil, exit.New(exit.Credential, "private execution authority has invalid fields")
	}
	key := ed25519.NewKeyFromSeed(value.PrivateKey.Seed())
	if !bytes.Equal(key, value.PrivateKey) || !bytes.Equal(key.Public().(ed25519.PublicKey), g.ExecutionPublicKey) {
		return nil, exit.New(exit.Credential, "private execution key differs from its signed grant")
	}
	if _, err := canonical.Bytes(g); err != nil {
		return nil, exit.New(exit.Credential, "private execution grant is not canonical")
	}
	return &PrivateExecutionOwner{Authorization: auth, PrivateKey: key}, nil
}

func (c *Orchestrator) ownerID() string {
	if c.opt.PrivateExecution != nil {
		return c.opt.PrivateExecution.Authorization.Grant.RecordOwnerId
	}
	return defaultRecordOwnerID
}
func (c *Orchestrator) ownerEpoch() uint64 {
	if c.opt.PrivateExecution != nil {
		return c.opt.PrivateExecution.Authorization.Grant.RecordOwnerEpoch
	}
	return defaultRecordOwnerEpoch
}

func (c *Orchestrator) executionClaimProof(connection *WorkerConnection) ([]byte, *exit.Error) {
	owner := c.opt.PrivateExecution
	if owner == nil || connection == nil {
		return nil, exit.New(exit.Credential, "private execution requires its granted worker connection")
	}
	g := owner.Authorization.Grant
	if connection.WorkerID != g.WorkerId || connection.WorkerBootID != g.WorkerBootId {
		return nil, exit.New(exit.Credential, "private execution cannot claim another worker or boot")
	}
	pin, err := workertls.LoadPin(connection.CACert)
	if err != nil || !bytes.Equal(pin.Digest(), g.WorkerTlsCertificateDigest) {
		return nil, exit.New(exit.Credential, "private execution cannot claim another worker TLS identity")
	}
	raw, err := canonical.Bytes(g)
	if err != nil {
		return nil, exit.New(exit.Credential, "private execution grant is not canonical")
	}
	proof, err := canonical.Bytes(&pb.ExecutionClaimProof{ExecutionGrantDigest: canonical.Digest(raw), RecordOwnerEpoch: g.RecordOwnerEpoch, RecordOwnerId: g.RecordOwnerId, WorkerId: g.WorkerId, WorkerBootId: g.WorkerBootId, WorkerTlsCertificateDigest: g.WorkerTlsCertificateDigest})
	if err != nil {
		return nil, exit.New(exit.Credential, "private execution claim is not canonical")
	}
	return ed25519.Sign(owner.PrivateKey, proof), nil
}
