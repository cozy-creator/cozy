// Package rental is the local side of a rented pod (cl-015): where its facts live, where
// its credential lives, and the ONE function that turns a rental id into the dial triple
// the orchestrator attaches a remote worker with.
//
// The split is deliberate and is the same one cl-006 already made for the CLI's own
// credential: the FACTS are rows in the records authority (they are durable lifecycle
// state and every reader may see them), and the OWNER TOKEN is a 0600 file (every reader
// of the database may NOT). The pinned certificate is public and sits beside it as a file
// only because that is what crypto/x509 wants to be handed.
package rental

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/remotecontrol"
	"github.com/cozy-creator/cozy/internal/rentalid"
	"github.com/cozy-creator/cozy/internal/secret"
)

func validID(id string) *exit.Error {
	if !rentalid.Valid(id) {
		return exit.Named(exit.Validation, "rental.id_invalid",
			"rental id %q is not a portable opaque name", id).
			WithRemedy("use the exact rental id Tensorhub returned; never a path or provider resource id")
	}
	return nil
}

// Attach persists everything a later process needs to dial this rental. Secret/public
// files land before the row can advertise a dialable target. The pre-POST operation row
// already names the paid resource, so a crash at any point resumes rather than orphaning
// a pod or publishing a row whose credential is absent.
func Attach(l home.Layout, st *records.Store, row records.Rental, cert string, token secret.Value,
	creator CreatorIdentity) *exit.Error {
	if e := validID(row.ID); e != nil {
		return e
	}
	// Validate and project the complete exact snapshot BEFORE files or a dialable
	// rental row become visible. A bad snapshot never publishes a WorkerTarget.
	if _, e := controlFacts(row); e != nil {
		return e
	}
	stored, e := st.RentalRow(row.ID)
	if e != nil {
		return e
	}
	if stored != nil {
		if stored.Address != "" && stored.Address != row.Address ||
			stored.MediaAddress != "" && stored.MediaAddress != row.MediaAddress {
			return exit.Named(exit.Conflict, "rental.attach_projection_conflict",
				"rental %s changed its control or media address before attachment", row.ID)
		}
		if stored.CertPath != "" {
			pinned, err := os.ReadFile(stored.CertPath)
			if err != nil || !bytes.Equal(pinned, []byte(cert)) {
				return exit.Named(exit.Conflict, "rental.attach_projection_conflict",
					"rental %s changed its pinned certificate before attachment", row.ID)
			}
		}
	}
	if err := os.MkdirAll(l.Rentals, 0o700); err != nil {
		return exit.Internalf("cannot create the rental credential root %s: %s", l.Rentals, err)
	}
	if err := os.WriteFile(l.RentalCert(row.ID), []byte(cert), 0o644); err != nil {
		return exit.Internalf("cannot pin the rental's certificate: %s", err)
	}
	if e := write0600(l.RentalToken(row.ID), secret.FileBody(token)); e != nil {
		return e
	}
	if len(creator.pem) == 0 {
		return exit.Internalf("rental %s has no pending Creator identity", row.ID)
	}
	if e := write0600(l.RentalCreatorIdentity(row.ID), creator.pem); e != nil {
		return e
	}
	row.CertPath = l.RentalCert(row.ID)
	return st.RecordRental(row)
}

// PendingToken establishes the plaintext credential BEFORE a paid POST. O_EXCL makes the
// file the winner under concurrent retries of one operation key: every contender then
// reads the same token instead of truncating it with fresh entropy.
func PendingToken(l home.Layout, operationKey string) (secret.Value, *exit.Error) {
	if err := os.MkdirAll(l.Rentals, 0o700); err != nil {
		return secret.Value{}, exit.Internalf("cannot create the rental credential root %s: %s", l.Rentals, err)
	}
	path := l.PendingRentalToken(operationKey)
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		return tokenAt(path, "pending rental operation "+operationKey)
	}
	if err != nil {
		return secret.Value{}, exit.Internalf("cannot create pending rental token: %s", err)
	}
	token := secret.Mint()
	ok := false
	defer func() {
		_ = f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	if err := f.Chmod(0o600); err != nil {
		return secret.Value{}, exit.Internalf("cannot restrict pending rental token: %s", err)
	}
	if _, err := f.Write(secret.FileBody(token)); err != nil {
		return secret.Value{}, exit.Internalf("cannot write pending rental token: %s", err)
	}
	if err := f.Sync(); err != nil {
		return secret.Value{}, exit.Internalf("cannot make pending rental token durable: %s", err)
	}
	if runtime.GOOS != "windows" {
		dir, err := os.Open(l.Rentals)
		if err != nil {
			return secret.Value{}, exit.Internalf("cannot open the rental credential root for sync: %s", err)
		}
		err = dir.Sync()
		_ = dir.Close()
		if err != nil {
			return secret.Value{}, exit.Internalf("cannot make the pending rental token name durable: %s", err)
		}
	}
	ok = true
	return token, nil
}

// ForgetPending removes the pre-id token only after the operation is attached or proved
// terminal. An interrupted poll deliberately leaves it for the exact-key retry.
func ForgetPending(l home.Layout, operationKey string) {
	_ = os.Remove(l.PendingRentalToken(operationKey))
	_ = os.Remove(l.PendingRentalCreatorIdentity(operationKey))
}

// Forget removes the local half. The pod is the hub's to destroy; this is what stops
// this host from holding a credential for something that no longer exists.
func Forget(l home.Layout, st *records.Store, id string) (bool, *exit.Error) {
	if e := validID(id); e != nil {
		return false, e
	}
	forgotten, e := st.ForgetRental(id)
	if e != nil {
		return false, e
	}
	_ = os.Remove(l.RentalToken(id))
	_ = os.Remove(l.RentalCert(id))
	_ = os.Remove(l.RentalCreatorIdentity(id))
	return forgotten, nil
}

// Token reads one rental's owner token back, refusing a file whose mode widened. Reading
// a credential that became group- or world-readable would be this client agreeing to a
// leak it created the file to prevent — cl-006's rule for the CLI credential, and the
// same one here because it is the same class of file.
func Token(l home.Layout, id string) (secret.Value, *exit.Error) {
	if e := validID(id); e != nil {
		return secret.Value{}, e
	}
	return tokenAt(l.RentalToken(id), "rental "+id)
}

func tokenAt(path, subject string) (secret.Value, *exit.Error) {
	info, err := os.Stat(path)
	if err != nil {
		return secret.Value{}, exit.New(exit.NotFound,
			"%s has no owner token on this host", subject).
			WithRemedy("`cozy rental new` writes it when the pod comes ready; a rental rented elsewhere is not this host's").
			WithNext("cozy rental list")
	}
	// Windows reports 0666 for every file: the boundary there is the user profile's ACL,
	// which already scopes COZY_HOME to the user, so the bits are not consulted.
	if perm := info.Mode().Perm(); perm&0o077 != 0 && runtime.GOOS != "windows" {
		return secret.Value{}, exit.New(exit.Credential,
			"%s is mode %#o; a rental's owner token is 0600 or it is not used", path, perm).
			WithRemedy("release this rental and rent again: every rental provisions its own token").
			WithNext("cozy rental list")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return secret.Value{}, exit.New(exit.Credential,
			"%s's owner token is unreadable: %s", subject, err)
	}
	v := secret.New(string(data))
	if !v.Present() {
		return secret.Value{}, exit.New(exit.Credential,
			"%s's owner token file is empty", subject).WithNext("cozy rental list")
	}
	return v, nil
}

// Known is the non-secret target question. The HTTP layer asks it so a submission
// naming an absent or not-yet-published pod is refused before a request row exists, and
// so its plan resolves from the exact persisted snapshot. It never reads the owner token.
func Known(st *records.Store) func(string) (*orchestrator.DesiredPlacement, *exit.Error) {
	return func(id string) (*orchestrator.DesiredPlacement, *exit.Error) {
		row, e := st.RentalRow(id)
		if e != nil {
			return nil, e
		}
		if row == nil {
			return nil, unknown(id)
		}
		if row.State != hub.RentalReady {
			refusal, problem := st.RentalRelayRefusal(id)
			if problem != nil {
				return nil, problem
			}
			if refusal != nil {
				return nil, refusal.Error()
			}
			return nil, exit.Named(exit.Unavailable, "rental.convergence_pending",
				"rental %s is %s; Tensorhub has not accepted its relayed worker convergence evidence",
				id, row.State).
				WithRemedy("keep `cozy invoke list` running so this host's RecordOwner can claim and converge the private worker")
		}
		if row.Address == "" {
			return nil, noAddress(id, row.State)
		}
		if row.CertPath == "" {
			return nil, exit.Unavailablef("rental %s has exact control but its WorkerTarget is not attached yet", id).
				WithRemedy("wait for `cozy rental new` to validate and atomically publish the target")
		}
		facts, e := controlFacts(*row)
		if e != nil {
			return nil, e
		}
		placement := facts.Placement
		return &placement, nil
	}
}

// PackageDescriptor returns the exact remote descriptor already frozen into one
// rental. It is the CLI payload surface for --worker; no local install is read.
func PackageDescriptor(st *records.Store, id string) (*launch.PackageDescriptor, *exit.Error) {
	row, e := st.RentalRow(id)
	if e != nil {
		return nil, e
	}
	if row == nil {
		return nil, unknown(id)
	}
	if row.CertPath == "" {
		return nil, exit.Unavailablef("rental %s has not published its WorkerTarget", id)
	}
	facts, e := controlFacts(*row)
	if e != nil {
		return nil, e
	}
	return facts.PackageDescriptor, nil
}

func unknown(id string) *exit.Error {
	return exit.New(exit.NotFound, "no rental %s on this host", id).
		WithRemedy("`cozy rental list` names the pods this host holds").
		WithNext("cozy rental list")
}

func noAddress(id, state string) *exit.Error {
	return exit.Unavailablef("rental %s is %s and carries no address yet", id, state).
		WithRemedy("`cozy rental list` shows its state; a pod that failed to provision never gets one").
		WithNext("cozy rental list")
}

// Resolver is what the daemon entrypoint hands the orchestrator as `Options.Rentals`. It
// is the DIAL-TIME resolution of Creator mTLS identity and media bearer. It reads the
// store on EVERY call rather than closing over a
// snapshot: `cozy rental new` is a records-plane act that runs against a daemon already up, so
// a resolver that cached would refuse the rental the user just made until a restart.
func Resolver(l home.Layout, st *records.Store) func(string) (*orchestrator.RemoteTarget, *exit.Error) {
	return func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
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
		if row.CertPath == "" {
			return nil, exit.Unavailablef("rental %s is not attached on this host", id).
				WithRemedy("resume the original rental operation so control, certificate, and token publish together")
		}
		facts, e := controlFacts(*row)
		if e != nil {
			return nil, e
		}
		facts.Placement.PlacementRevision = row.PlacementRevision
		token, e := Token(l, id)
		if e != nil {
			return nil, e
		}
		if _, e := loadCreatorIdentity(l.RentalCreatorIdentity(id)); e != nil {
			return nil, e.WithRemedy("end and re-rent; a lost per-rental Creator key cannot be rotated into the live pod")
		}
		cert := row.CertPath
		if cert == "" {
			cert = l.RentalCert(id)
		}
		if _, err := os.Stat(cert); err != nil {
			return nil, exit.New(exit.NotFound,
				"rental %s pins a certificate this host cannot read: %s", id, err).
				WithRemedy("release this rental and rent again; the pin is written with the token").
				WithNext("cozy rental end " + id)
		}
		spec := &orchestrator.WorkerConnection{
			RentalID: row.ID, Addr: row.Address, CACert: cert,
			WorkerID: row.ExpectedWorkerID, WorkerBootID: row.ExpectedWorkerBootID,
		}
		if row.MediaAddress != "" {
			// ONE PROVISIONED IDENTITY, TWO LISTENERS (#506b, tonight's tier). The pod's
			// media server holds its OWN keys — cl-014's rule, and this host pins the same
			// PEM only because the stand-in provisioner mints one certificate covering both
			// names. What this host never does is MINT anything: the bearer the media plane
			// checks is the rental's provisioned owner token, whose digest reached the pod
			// as a launch grant from whoever provisioned it.
			spec.Media = &media.Spec{Addr: row.MediaAddress, Token: token, CACert: cert}
		}
		return &orchestrator.RemoteTarget{Connection: spec, Placement: facts.Placement}, nil
	}
}

// RelayWorkerSession is the private-rental observation seam. The ordinary authenticated
// Hub client carries account authority; the media bearer opens only the pod media plane.
func RelayWorkerSession(st *records.Store, client *hub.Client) orchestrator.RentalSessionRelay {
	return func(ctx context.Context, connection *orchestrator.WorkerConnection,
		evidence orchestrator.RentalSessionEvidence) *exit.Error {
		if connection == nil || connection.RentalID == "" {
			return exit.Named(exit.Credential, "rental.worker_observation_authority_missing",
				"the connected worker has no rental identity")
		}
		row, problem := st.RentalRow(connection.RentalID)
		if problem != nil {
			return problem
		}
		if row == nil {
			return unknown(connection.RentalID)
		}
		if row.Hub != client.Base() {
			return exit.Named(exit.Conflict, "rental.hub_mismatch",
				"rental %s belongs to %s, configured hub is %s",
				row.ID, row.Hub, client.Base())
		}
		answer, problem := client.ObserveWorkerSession(
			ctx, connection.RentalID, hub.WorkerSessionObservation{
				ClaimAck: evidence.ClaimAck, Snapshot: evidence.Snapshot,
				ObservedState: evidence.ObservedState, BootFailure: evidence.BootFailure,
				DesiredRevision: evidence.DesiredRevision,
			})
		if problem != nil {
			if problem.Code != exit.Unavailable && problem.Code != exit.Deadline {
				if recordProblem := st.RecordRentalRelayRefusal(row.ID, problem); recordProblem != nil {
					return recordProblem
				}
			}
			return problem
		}
		row.State = answer.State
		if problem := st.RecordRental(*row); problem != nil {
			return problem
		}
		return st.ClearRentalRelayRefusal(row.ID)
	}
}

// RecordControlRefusal makes a private-worker Claim or owner-side identity refusal durable.
// The convergence poll already reads this authority; no separate claim-refusal table exists.
func RecordControlRefusal(st *records.Store) func(string, *exit.Error) *exit.Error {
	return func(rentalID string, problem *exit.Error) *exit.Error {
		return st.RecordRentalRelayRefusal(rentalID, problem)
	}
}

// ObserveWorker turns a remote ClaimAck into the rental's durable actual-hardware
// readback. It is wired into the orchestrator so no remote session can become
// dispatchable without crossing this records boundary.
func ObserveWorker(st *records.Store) func(orchestrator.RentalObservation) *exit.Error {
	return func(observed orchestrator.RentalObservation) *exit.Error {
		return st.ObserveRentalWorker(observed.RentalID, observed.Accelerator,
			observed.Backend, observed.DriverVersion, observed.BackendVersion,
			observed.DeviceMemoryTotalBytes, observed.WorkerInstance, observed.WorkerID,
			observed.WorkerBootID, observed.DeviceCount)
	}
}

func controlFacts(row records.Rental) (remotecontrol.Facts, *exit.Error) {
	if row.ControlSnapshotDigest == "" || len(row.ControlSnapshotBytes) == 0 {
		return remotecontrol.Facts{}, exit.Named(exit.Conflict, "rental.control_snapshot_missing",
			"rental %s has no persisted acquisition-attempt control snapshot", row.ID).
			WithRemedy("release it and rent again; Cozy will not resolve a remote pod from the local install")
	}
	return remotecontrol.Decode(hub.ExactControlDocument{
		CanonicalBytes: row.ControlSnapshotBytes, Digest: row.ControlSnapshotDigest,
		Length: int64(len(row.ControlSnapshotBytes)),
	}, row.PackageRef)
}

// ValidateControl refuses a hub observation before its bytes become the rental's durable,
// write-once control snapshot. Attach repeats the same check at the publication boundary.
func ValidateControl(row records.Rental) *exit.Error {
	_, e := controlFacts(row)
	return e
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
