// Package rental is the local side of a rented pod (cl-015): where its facts live, where
// its credential lives, and the ONE function that turns a rental id into the dial triple
// the coordinator attaches a remote worker with.
//
// The split is deliberate and is the same one cl-006 already made for the CLI's own
// credential: the FACTS are rows in the records authority (they are durable lifecycle
// state and every reader may see them), and the OWNER TOKEN is a 0600 file (every reader
// of the database may NOT). The pinned certificate is public and sits beside it as a file
// only because that is what crypto/x509 wants to be handed.
package rental

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/cozy-creator/cozy-creator-v2/internal/coord"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/media"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

// Attach persists everything a later process needs to dial this rental: the row, the
// pinned certificate, and the owner token under 0600. It is written in that order on
// purpose — the row is what `cozy rent ls` and `release` work from, so a crash between
// the credential and the row would leave a pod nobody can name or tear down.
func Attach(l home.Layout, st *records.Store, row records.Rental, cert string, token secret.Value) *exit.Error {
	if err := os.MkdirAll(l.Rentals, 0o700); err != nil {
		return exit.Internalf("cannot create the rental credential root %s: %s", l.Rentals, err)
	}
	if err := os.WriteFile(l.RentalCert(row.ID), []byte(cert), 0o644); err != nil {
		return exit.Internalf("cannot pin the rental's certificate: %s", err)
	}
	if e := write0600(l.RentalToken(row.ID), secret.FileBody(token)); e != nil {
		return e
	}
	row.CertPath = l.RentalCert(row.ID)
	return st.RecordRental(row)
}

// Forget removes the local half. The pod is the hub's to destroy; this is what stops
// this host from holding a credential for something that no longer exists.
func Forget(l home.Layout, st *records.Store, id string) (bool, *exit.Error) {
	_ = os.Remove(l.RentalToken(id))
	_ = os.Remove(l.RentalCert(id))
	return st.ForgetRental(id)
}

// Token reads one rental's owner token back, refusing a file whose mode widened. Reading
// a credential that became group- or world-readable would be this client agreeing to a
// leak it created the file to prevent — cl-006's rule for the CLI credential, and the
// same one here because it is the same class of file.
func Token(l home.Layout, id string) (secret.Value, *exit.Error) {
	path := l.RentalToken(id)
	info, err := os.Stat(path)
	if err != nil {
		return secret.Value{}, exit.New(exit.NotFound,
			"rental %s has no owner token on this host", id).
			WithRemedy("`cozy rent` writes it when the pod comes ready; a rental rented elsewhere is not this host's").
			WithNext("cozy rent ls")
	}
	// Windows reports 0666 for every file: the boundary there is the user profile's ACL,
	// which already scopes COZY_HOME to the user, so the bits are not consulted.
	if perm := info.Mode().Perm(); perm&0o077 != 0 && runtime.GOOS != "windows" {
		return secret.Value{}, exit.New(exit.Credential,
			"%s is mode %#o; a rental's owner token is 0600 or it is not used", path, perm).
			WithRemedy("release this rental and rent again: every rental provisions its own token").
			WithNext("cozy rent release " + id + " --yes")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return secret.Value{}, exit.New(exit.Credential,
			"rental %s's owner token is unreadable: %s", id, err)
	}
	v := secret.New(string(data))
	if !v.Present() {
		return secret.Value{}, exit.New(exit.Credential,
			"rental %s's owner token file is empty", id).WithNext("cozy rent ls")
	}
	return v, nil
}

// Known is the EXISTENCE question, and only that. The HTTP layer asks it so a submission
// naming a pod this host does not hold is refused before a request row exists; it must
// not resolve the credential, because a check that loaded the owner token would expose it
// a second time to answer a question the row already answers.
func Known(st *records.Store) func(string) *exit.Error {
	return func(id string) *exit.Error {
		row, e := st.RentalRow(id)
		if e != nil {
			return e
		}
		if row == nil {
			return unknown(id)
		}
		if row.Address == "" {
			return noAddress(id, row.State)
		}
		return nil
	}
}

func unknown(id string) *exit.Error {
	return exit.New(exit.NotFound, "no rental %s on this host", id).
		WithRemedy("`cozy rent ls` names the pods this host holds").
		WithNext("cozy rent ls")
}

func noAddress(id, state string) *exit.Error {
	return exit.Unavailablef("rental %s is %s and carries no address yet", id, state).
		WithRemedy("`cozy rent ls` shows its state; a pod that failed to provision never gets one").
		WithNext("cozy rent ls")
}

// Resolver is what the service entrypoint hands the coordinator as `Options.Rentals`. It
// is the DIAL-TIME resolution — the one place the owner token is read, at the moment it
// becomes Claim.proof. It reads the store on EVERY call rather than closing over a
// snapshot: `cozy rent` is a records-plane act that runs against a service already up, so
// a resolver that cached would refuse the rental the user just made until a restart.
func Resolver(l home.Layout, st *records.Store) func(string) (*coord.RemoteSpec, *exit.Error) {
	return func(id string) (*coord.RemoteSpec, *exit.Error) {
		row, e := st.RentalRow(id)
		if e != nil {
			return nil, e
		}
		if row == nil {
			return nil, unknown(id)
		}
		if row.Address == "" {
			return nil, noAddress(id, row.State)
		}
		token, e := Token(l, id)
		if e != nil {
			return nil, e
		}
		cert := row.CertPath
		if cert == "" {
			cert = l.RentalCert(id)
		}
		if _, err := os.Stat(cert); err != nil {
			return nil, exit.New(exit.NotFound,
				"rental %s pins a certificate this host cannot read: %s", id, err).
				WithRemedy("release this rental and rent again; the pin is written with the token").
				WithNext("cozy rent release " + id + " --yes")
		}
		spec := &coord.RemoteSpec{Addr: row.Address, Token: token, CACert: cert}
		if row.MediaAddress != "" {
			// ONE PROVISIONED IDENTITY, TWO LISTENERS (#506b, tonight's tier). The pod's
			// media server holds its OWN keys — cl-014's rule, and this host pins the same
			// PEM only because the stand-in provisioner mints one certificate covering both
			// names. What this host never does is MINT anything: the bearer the media plane
			// checks is the rental's provisioned owner token, hashed into the pod's
			// token-hash file by whoever provisioned the pod.
			spec.Media = &media.Spec{Addr: row.MediaAddress, Token: token, CACert: cert}
		}
		return spec, nil
	}
}

func write0600(path string, body []byte) *exit.Error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return exit.Internalf("cannot write %s: %s", filepath.Base(path), err)
	}
	defer f.Close()
	// Chmod anyway: O_CREAT's mode is masked by umask, and a credential readable by the
	// group because of an inherited umask is exactly the bug this file prevents.
	if err := f.Chmod(0o600); err != nil {
		return exit.Internalf("cannot restrict %s: %s", filepath.Base(path), err)
	}
	if _, err := f.Write(body); err != nil {
		return exit.Internalf("cannot write %s: %s", filepath.Base(path), err)
	}
	return nil
}
