package rental

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"os"
	"runtime"

	"github.com/cozy-creator/cozy/internal/machinev1"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
)

// CreatorIdentity is one rental's private RecordOwner identity. Its fields stay
// private so formatting or JSON cannot accidentally expose key material.
type CreatorIdentity struct {
	pem     []byte
	private ed25519.PrivateKey
	public  ed25519.PublicKey
}

func (i CreatorIdentity) PublicKey() string {
	return base64.RawURLEncoding.EncodeToString(i.public)
}

func (i CreatorIdentity) Sign(message []byte) []byte {
	return ed25519.Sign(i.private, message)
}

// Signer is this identity as the signer of a machine's Cozy-Caps.
func (i CreatorIdentity) Signer() machinev1.Signer {
	return machinev1.Signer{Public: i.public, Sign: i.Sign}
}

func CreatorIdentityFor(l home.Layout, rentalID string) (CreatorIdentity, *exit.Error) {
	if problem := validID(rentalID); problem != nil {
		return CreatorIdentity{}, problem
	}
	return loadCreatorIdentity(l.RentalCreatorIdentity(rentalID))
}

// PendingCreatorIdentity mints before the paid POST and reuses the same key on an
// operation retry. Failed generation leaves no partial credential.
func PendingCreatorIdentity(l home.Layout, operationKey string) (CreatorIdentity, *exit.Error) {
	if err := os.MkdirAll(l.Rentals, 0o700); err != nil {
		return CreatorIdentity{}, exit.Internalf("cannot create the rental credential root: %s", err)
	}
	path := l.PendingRentalCreatorIdentity(operationKey)
	if _, err := os.Stat(path); err == nil {
		return loadCreatorIdentity(path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return CreatorIdentity{}, exit.Internalf("cannot inspect the pending Creator identity: %s", err)
	}
	identity, problem := mintCreatorIdentity()
	if problem != nil {
		return CreatorIdentity{}, problem
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		return loadCreatorIdentity(path)
	}
	if err != nil {
		return CreatorIdentity{}, exit.Internalf("cannot create the pending Creator identity: %s", err)
	}
	ok := false
	defer func() {
		_ = f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	if err := f.Chmod(0o600); err == nil {
		_, err = f.Write(identity.pem)
	}
	if err == nil {
		err = f.Sync()
	}
	if err != nil {
		return CreatorIdentity{}, exit.Internalf("cannot persist the pending Creator identity: %s", err)
	}
	ok = true
	return identity, nil
}

// OwnerIdentityAt is a machine's owner key at path, minted 0600 on first use. The local
// machine keeps one for its lifetime, as a rental keeps its per-rental key.
func OwnerIdentityAt(path string) (CreatorIdentity, *exit.Error) {
	if _, err := os.Stat(path); err == nil {
		return loadCreatorIdentity(path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return CreatorIdentity{}, exit.Internalf("cannot inspect the machine owner key: %s", err)
	}
	identity, problem := mintCreatorIdentity()
	if problem != nil {
		return CreatorIdentity{}, problem
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		return loadCreatorIdentity(path)
	}
	if err != nil {
		return CreatorIdentity{}, exit.Internalf("cannot create the machine owner key: %s", err)
	}
	_, err = f.Write(identity.pem)
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(path)
		return CreatorIdentity{}, exit.Internalf("cannot persist the machine owner key: %s", err)
	}
	return identity, nil
}

// ExistingOwnerIdentityAt never creates or changes a controller's signing identity.
func ExistingOwnerIdentityAt(path string) (CreatorIdentity, *exit.Error) {
	return loadCreatorIdentity(path)
}

func mintCreatorIdentity() (CreatorIdentity, *exit.Error) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return CreatorIdentity{}, exit.Internalf("cannot generate the rental Creator key: %s", err)
	}
	key, err := x509.MarshalPKCS8PrivateKey(private)
	if err != nil {
		return CreatorIdentity{}, exit.Internalf("cannot encode the rental Creator key: %s", err)
	}
	encoded := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key})
	return CreatorIdentity{pem: encoded, private: private, public: public}, nil
}

func loadCreatorIdentity(path string) (CreatorIdentity, *exit.Error) {
	info, err := os.Stat(path)
	if err != nil {
		return CreatorIdentity{}, exit.New(exit.NotFound, "the rental Creator identity is unavailable: %s", err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return CreatorIdentity{}, exit.Named(exit.Credential, "rental.creator_identity_permissions",
			"%s is mode %#o; a rental Creator identity must be 0600", path, info.Mode().Perm())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return CreatorIdentity{}, exit.New(exit.Credential, "the rental Creator identity is unreadable: %s", err)
	}
	block, trailing := pem.Decode(data)
	if block == nil || block.Type != "PRIVATE KEY" || len(trailing) != 0 {
		return CreatorIdentity{}, exit.New(exit.Credential, "the rental Creator identity is malformed")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return CreatorIdentity{}, exit.New(exit.Credential, "the rental Creator identity is malformed")
	}
	private, privateOK := parsed.(ed25519.PrivateKey)
	if !privateOK {
		return CreatorIdentity{}, exit.New(exit.Credential, "the rental Creator identity is not one Ed25519 keypair")
	}
	public := private.Public().(ed25519.PublicKey)
	return CreatorIdentity{pem: data, private: private, public: public}, nil
}
