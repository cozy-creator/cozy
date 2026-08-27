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
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sort"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/hub"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/media"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/remotecontrol"
	"github.com/cozy-creator/cozy-creator-v2/internal/rentalid"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
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
func Attach(l home.Layout, st *records.Store, row records.Rental, cert string, token secret.Value) *exit.Error {
	if e := validID(row.ID); e != nil {
		return e
	}
	// Validate and project the complete exact snapshot BEFORE files or a dialable
	// rental row become visible. A bad snapshot never publishes a WorkerTarget.
	if _, e := controlFacts(row); e != nil {
		return e
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
			WithRemedy("`cozy rent` writes it when the pod comes ready; a rental rented elsewhere is not this host's").
			WithNext("cozy rent ls")
	}
	// Windows reports 0666 for every file: the boundary there is the user profile's ACL,
	// which already scopes COZY_HOME to the user, so the bits are not consulted.
	if perm := info.Mode().Perm(); perm&0o077 != 0 && runtime.GOOS != "windows" {
		return secret.Value{}, exit.New(exit.Credential,
			"%s is mode %#o; a rental's owner token is 0600 or it is not used", path, perm).
			WithRemedy("release this rental and rent again: every rental provisions its own token").
			WithNext("cozy rent ls")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return secret.Value{}, exit.New(exit.Credential,
			"%s's owner token is unreadable: %s", subject, err)
	}
	v := secret.New(string(data))
	if !v.Present() {
		return secret.Value{}, exit.New(exit.Credential,
			"%s's owner token file is empty", subject).WithNext("cozy rent ls")
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
		if row.Address == "" {
			return nil, noAddress(id, row.State)
		}
		if row.CertPath == "" {
			return nil, exit.Unavailablef("rental %s has exact control but its WorkerTarget is not attached yet", id).
				WithRemedy("wait for `cozy rent` to validate and atomically publish the target")
		}
		facts, e := controlFacts(*row)
		if e != nil {
			return nil, e
		}
		placement := facts.Placement
		return &placement, nil
	}
}

// Descriptor returns the exact remote descriptor already frozen into one
// rental. It is the CLI payload surface for --worker; no local install is read.
func Descriptor(st *records.Store, id string) (*launch.Descriptor, *exit.Error) {
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
	return facts.Descriptor, nil
}

// ControlSummary is the exact, non-secret execution closure Tensorhub selected for one
// attached rental. It is an observation surface for acceptance and diagnosis, never an
// input to selection: callers still ask only for endpoint + provider-neutral accelerator.
type ControlSummary struct {
	ControlSnapshotDigest             string   `json:"control_snapshot_digest"`
	EndpointExecutionDigest           string   `json:"endpoint_execution_digest"`
	ArtifactObjectSetDigest           string   `json:"artifact_object_set_digest"`
	ModelRootDigests                  []string `json:"model_root_digests"`
	EndpointReleaseID                 string   `json:"endpoint_release_id"`
	DescriptorDigest                  string   `json:"descriptor_digest"`
	EnvironmentSpecDigest             string   `json:"environment_spec_digest"`
	InstalledEnvironmentReceiptDigest string   `json:"installed_environment_receipt_digest"`
	PlacementSetDigest                string   `json:"placement_set_digest"`
	BindingPlanDigests                []string `json:"binding_plan_digests"`
}

// Inspect returns the persisted rental row plus the verified projection of its exact
// acquisition-attempt control snapshot. No provider resource id, lane selector, or secret
// is exposed or reconstructed.
func Inspect(st *records.Store, id string) (*records.Rental, ControlSummary, *exit.Error) {
	row, e := st.RentalRow(id)
	if e != nil || row == nil {
		if e == nil {
			e = unknown(id)
		}
		return row, ControlSummary{}, e
	}
	facts, e := controlFacts(*row)
	if e != nil {
		return row, ControlSummary{}, e
	}
	plans := make([]string, 0, len(facts.Placement.Bindings))
	for _, binding := range facts.Placement.Bindings {
		planID, problem := binding.PlanID()
		if problem != nil {
			return row, ControlSummary{}, problem
		}
		plans = append(plans, planID)
	}
	sort.Strings(plans)
	return row, ControlSummary{
		ControlSnapshotDigest:             row.ControlSnapshotDigest,
		EndpointExecutionDigest:           facts.EndpointExecutionDigest,
		ArtifactObjectSetDigest:           facts.ArtifactObjectSetDigest,
		ModelRootDigests:                  append([]string(nil), facts.ModelRootDigests...),
		EndpointReleaseID:                 facts.Placement.ReleaseID,
		DescriptorDigest:                  facts.Placement.DescriptorDigest,
		EnvironmentSpecDigest:             facts.Placement.EnvironmentSpecDigest,
		InstalledEnvironmentReceiptDigest: facts.Placement.InstalledEnvironmentReceiptDigest,
		PlacementSetDigest:                facts.Placement.ExactPlacementSetDigest,
		BindingPlanDigests:                plans,
	}, nil
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

// Resolver is what the service entrypoint hands the orchestrator as `Options.Rentals`. It
// is the DIAL-TIME resolution — the one place the owner token is read, at the moment it
// becomes Claim.proof. It reads the store on EVERY call rather than closing over a
// snapshot: `cozy rent` is a records-plane act that runs against a service already up, so
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
		spec := &orchestrator.WorkerConnection{Addr: row.Address, Token: token, CACert: cert}
		if row.MediaAddress != "" {
			// ONE PROVISIONED IDENTITY, TWO LISTENERS (#506b, tonight's tier). The pod's
			// media server holds its OWN keys — cl-014's rule, and this host pins the same
			// PEM only because the stand-in provisioner mints one certificate covering both
			// names. What this host never does is MINT anything: the bearer the media plane
			// checks is the rental's provisioned owner token, hashed into the pod's
			// token-hash file by whoever provisioned the pod.
			spec.Media = &media.Spec{Addr: row.MediaAddress, Token: token, CACert: cert}
		}
		return &orchestrator.RemoteTarget{Connection: spec, Placement: facts.Placement}, nil
	}
}

func controlFacts(row records.Rental) (remotecontrol.Facts, *exit.Error) {
	if row.ControlSnapshotDigest == "" || row.ControlSnapshotLength <= 0 || len(row.ControlSnapshotBytes) == 0 {
		return remotecontrol.Facts{}, exit.Named(exit.Conflict, "rental.control_snapshot_missing",
			"rental %s has no persisted acquisition-attempt control snapshot", row.ID).
			WithRemedy("release it and rent again; Creator will not resolve a remote pod from the local install")
	}
	return remotecontrol.Decode(hub.ExactControlDocument{
		CanonicalBytes: row.ControlSnapshotBytes, Digest: row.ControlSnapshotDigest,
		Length: row.ControlSnapshotLength,
	}, row.EndpointRef)
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
