package records

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

const maxRecoveredInstalls = 4096

// RecoverPackageInventory merges verified package installs and their active pins
// from one explicit read-only Creator database. It never discovers backups, guesses by
// mtime, traverses install trees, changes model roots, or replaces a current row.
// Every source row is joined to exact bounded files before one transaction makes any
// noncolliding row visible. An identical row is an idempotent no-op; any other install
// id or package-pin collision refuses the whole merge.
func (s *Store) RecoverPackageInventory(sourcePath, installsRoot string) (int, *exit.Error) {
	absolute, err := filepath.Abs(sourcePath)
	if err != nil || absolute == "" {
		return 0, exit.Usagef("package inventory source %q is not an absolute database path", sourcePath)
	}
	info, err := os.Stat(absolute)
	if err != nil || !info.Mode().IsRegular() {
		return 0, exit.New(exit.NotFound, "package inventory source %s is not a regular file", absolute)
	}
	u := &url.URL{Scheme: "file", Path: absolute}
	query := u.Query()
	query.Set("mode", "ro")
	u.RawQuery = query.Encode()
	source, err := sql.Open("sqlite", u.String())
	if err != nil {
		return 0, exit.Internalf("cannot open package inventory source %s: %s", absolute, err)
	}
	defer source.Close()
	source.SetMaxOpenConns(1)
	if err := source.Ping(); err != nil {
		return 0, exit.Named(exit.Validation, "package_inventory_source_unreadable",
			"cannot read package inventory source %s: %s", absolute, err)
	}
	// The source is opened READ-ONLY and is never migrated, so it is read with the current
	// schema's own table and column names or not at all. Reading an older shape would mean
	// a second spelling of the install table living on in this one reader; a newer one is
	// read through the columns this build knows.
	version, err := databaseVersion(source)
	if err != nil || version < schemaVersion {
		return 0, exit.Named(exit.Validation, "package_inventory_source_schema_unsupported",
			"package inventory source has schema %d, not %d", version, schemaVersion).
			WithRemedy("open that backup with this build of cozy first; it migrates in place")
	}
	if problem := requireTables(source, "installs", "pins"); problem != nil {
		return 0, problem
	}
	var installCount, pinCount int
	if err := source.QueryRow(`SELECT COUNT(*) FROM installs`).Scan(&installCount); err != nil ||
		installCount < 1 || installCount > maxRecoveredInstalls {
		return 0, exit.Named(exit.Validation, "package_inventory_source_count_invalid",
			"package inventory source contains %d installs; expected 1..%d",
			installCount, maxRecoveredInstalls)
	}
	if err := source.QueryRow(`SELECT COUNT(*) FROM pins`).Scan(&pinCount); err != nil ||
		pinCount < 1 || pinCount > installCount {
		return 0, exit.Named(exit.Validation, "package_inventory_source_pin_count_invalid",
			"package inventory source contains %d pins for %d installs", pinCount, installCount)
	}
	rows, err := source.Query(`SELECT ` + installCols("") + ` FROM installs ORDER BY id`)
	if err != nil {
		return 0, exit.Named(exit.Validation, "package_inventory_source_unreadable",
			"cannot read source package installs: %s", err)
	}
	var installs []PackageInstall
	seen := make(map[string]PackageInstall, installCount)
	for rows.Next() {
		inst, scanErr := scanInstall(rows)
		if scanErr != nil {
			rows.Close()
			return 0, exit.Named(exit.Validation, "package_inventory_source_unreadable",
				"cannot decode source package install: %s", scanErr)
		}
		if problem := verifyRecoverableInstall(inst, installsRoot); problem != nil {
			rows.Close()
			return 0, problem
		}
		seen[inst.ID] = inst
		installs = append(installs, inst)
	}
	rows.Close()
	if len(installs) != installCount {
		return 0, exit.Named(exit.Conflict, "package_inventory_source_changed",
			"package inventory source changed while it was being read")
	}
	pinRows, err := source.Query(`SELECT package,major,install_id,activated_at FROM pins ORDER BY package`)
	if err != nil {
		return 0, exit.Named(exit.Validation, "package_inventory_source_unreadable",
			"cannot read source package pins: %s", err)
	}
	var pins []Pin
	packages := map[string]bool{}
	for pinRows.Next() {
		var pin Pin
		var inst PackageInstall
		exists := false
		scanErr := pinRows.Scan(&pin.Package, &pin.Major, &pin.InstallID, &pin.ActivatedAt)
		if scanErr == nil {
			inst, exists = seen[pin.InstallID]
		}
		if scanErr != nil || pin.Package == "" || packages[pin.Package] || !exists ||
			inst.Package != pin.Package || inst.Major != pin.Major {
			pinRows.Close()
			return 0, exit.Named(exit.Validation, "package_inventory_source_pin_invalid",
				"source package pin is duplicate, malformed, or names an absent install")
		}
		packages[pin.Package] = true
		pins = append(pins, pin)
	}
	pinRows.Close()
	if len(pins) != pinCount {
		return 0, exit.Named(exit.Conflict, "package_inventory_source_changed",
			"package inventory source pins changed while they were being read")
	}

	tx, err := s.db.Begin()
	if err != nil {
		return 0, exit.Internalf("cannot begin package inventory recovery: %s", err)
	}
	defer tx.Rollback()
	missingInstalls := make([]PackageInstall, 0, len(installs))
	for _, inst := range installs {
		current, scanErr := scanInstall(tx.QueryRow(
			`SELECT `+installCols("")+` FROM installs WHERE id=?`, inst.ID))
		switch {
		case errors.Is(scanErr, sql.ErrNoRows):
			missingInstalls = append(missingInstalls, inst)
		case scanErr != nil:
			return 0, exit.Internalf("cannot inspect current package install %s: %s",
				inst.ID, scanErr)
		case current != inst:
			return 0, exit.Named(exit.Conflict, "package_install_recovery_collision",
				"current package install %s differs from the verified backup row", inst.ID).
				WithRemedy("keep the current install or recover a backup whose immutable row is identical")
		}
	}
	missingPins := make([]Pin, 0, len(pins))
	for _, pin := range pins {
		var current Pin
		scanErr := tx.QueryRow(`SELECT package,major,install_id,activated_at FROM pins WHERE package=?`,
			pin.Package).Scan(&current.Package, &current.Major, &current.InstallID, &current.ActivatedAt)
		switch {
		case errors.Is(scanErr, sql.ErrNoRows):
			missingPins = append(missingPins, pin)
		case scanErr != nil:
			return 0, exit.Internalf("cannot inspect current package pin %s: %s", pin.Package, scanErr)
		case current != pin:
			return 0, exit.Named(exit.Conflict, "package_pin_recovery_collision",
				"current package pin %s differs from the verified backup row", pin.Package).
				WithRemedy("keep the current pin or recover a backup whose active pin is identical")
		}
	}
	for _, inst := range missingInstalls {
		verified := 0
		if inst.Verified {
			verified = 1
		}
		if _, err := tx.Exec(`INSERT INTO installs(`+installCols("")+`) VALUES(`+placeholders()+`)`,
			inst.ID, inst.Package, inst.Major, inst.Version,
			inst.SourceKind, inst.SourceRef, verified,
			inst.Dir, inst.Python, inst.Runtime, inst.ProjectDir,
			inst.UV, inst.Platform, inst.Extra,
			inst.Packages, inst.Closure,
			inst.PlacementSetDigest, inst.BytesExcl, inst.BytesShared,
			inst.CreatedAt); err != nil {
			return 0, exit.Internalf("cannot recover package install %s: %s", inst.ID, err)
		}
	}
	for _, pin := range missingPins {
		if _, err := tx.Exec(`INSERT INTO pins(package,major,install_id,activated_at) VALUES(?,?,?,?)`,
			pin.Package, pin.Major, pin.InstallID, pin.ActivatedAt); err != nil {
			return 0, exit.Internalf("cannot recover package pin %s: %s", pin.Package, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return 0, exit.New(exit.Conflict, "package inventory recovery did not commit: %s", err)
	}
	return len(missingInstalls), nil
}

func verifyRecoverableInstall(inst PackageInstall, installsRoot string) *exit.Error {
	if inst.ID == "" || inst.Package == "" || inst.Version == "" ||
		inst.SourceKind != "tensorhub" && inst.SourceKind != "local" {
		return exit.Named(exit.Validation, "package_install_recovery_invalid",
			"backup install %q has incomplete immutable identity", inst.ID)
	}
	if !inst.Verified {
		return exit.Named(exit.Validation, "package_install_recovery_unverified",
			"backup install %s is a local or otherwise unverified one", inst.ID).
			WithRemedy("reinstall that local package from its source; recovery imports only verified custody")
	}
	wantDir := filepath.Join(filepath.Clean(installsRoot), inst.ID)
	if !filepath.IsAbs(inst.Dir) || filepath.Clean(inst.Dir) != wantDir {
		return exit.Named(exit.Validation, "package_install_recovery_path_invalid",
			"backup install %s does not name its exact install directory", inst.ID)
	}
	info, err := os.Lstat(inst.Dir)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return exit.Named(exit.Validation, "package_install_recovery_path_invalid",
			"backup install %s directory is absent, not a directory, or a symlink", inst.ID)
	}
	if inst.PlacementSetDigest != "" {
		if problem := verifyRecoveryDigest(filepath.Join(inst.Dir, "artifact-cache",
			strings.TrimPrefix(inst.PlacementSetDigest, "sha256:")),
			inst.PlacementSetDigest); problem != nil {
			return problem
		}
	}
	return nil
}

func verifyRecoveryDigest(path, spelled string) *exit.Error {
	want, err := hex.DecodeString(strings.TrimPrefix(spelled, "sha256:"))
	if err != nil || len(want) != sha256.Size {
		return exit.Named(exit.Validation, "package_install_recovery_digest_invalid",
			"backup install declares malformed digest %q", spelled)
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 ||
		info.Size() < 1 || info.Size() > 16<<20 {
		return exit.Named(exit.Validation, "package_install_recovery_file_invalid",
			"install file %s is absent, unsafe, or over the 16 MiB bound", path)
	}
	raw, err := os.ReadFile(path)
	measured := sha256.Sum256(raw)
	if err != nil || !bytes.Equal(want, measured[:]) {
		return exit.Named(exit.Validation, "package_install_recovery_file_invalid",
			"install file %s is absent or does not match %s", path, spelled)
	}
	return nil
}
