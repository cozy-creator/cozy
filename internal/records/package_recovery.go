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

const maxRecoveredPackageGenerations = 4096

// RecoverPackageInventory merges verified package generations and their active pins
// from one explicit read-only Creator database. It never discovers backups, guesses by
// mtime, traverses generation trees, changes model roots, or replaces a current row.
// Every source row is joined to exact bounded files before one transaction makes any
// noncolliding row visible. An identical row is an idempotent no-op; any other generation
// id or package-pin collision refuses the whole merge.
func (s *Store) RecoverPackageInventory(sourcePath, generationsRoot string) (int, *exit.Error) {
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
	version, err := databaseVersion(source)
	if err != nil || version < 1 || version > schemaVersion {
		return 0, exit.Named(exit.Validation, "package_inventory_source_schema_unsupported",
			"package inventory source has schema %d; supported backups are 1 through %d",
			version, schemaVersion)
	}
	var generationCount, pinCount int
	if err := source.QueryRow(`SELECT COUNT(*) FROM install_generations`).Scan(&generationCount); err != nil ||
		generationCount < 1 || generationCount > maxRecoveredPackageGenerations {
		return 0, exit.Named(exit.Validation, "package_inventory_source_count_invalid",
			"package inventory source contains %d generations; expected 1..%d",
			generationCount, maxRecoveredPackageGenerations)
	}
	if err := source.QueryRow(`SELECT COUNT(*) FROM pins`).Scan(&pinCount); err != nil ||
		pinCount < 1 || pinCount > generationCount {
		return 0, exit.Named(exit.Validation, "package_inventory_source_pin_count_invalid",
			"package inventory source contains %d pins for %d generations", pinCount, generationCount)
	}
	rows, err := source.Query(`SELECT ` + genCols("") + ` FROM install_generations ORDER BY id`)
	if err != nil {
		return 0, exit.Named(exit.Validation, "package_inventory_source_unreadable",
			"cannot read source package generations: %s", err)
	}
	var generations []PackageInstall
	seen := make(map[string]PackageInstall, generationCount)
	for rows.Next() {
		generation, scanErr := scanGen(rows)
		if scanErr != nil {
			rows.Close()
			return 0, exit.Named(exit.Validation, "package_inventory_source_unreadable",
				"cannot decode source package generation: %s", scanErr)
		}
		if problem := verifyRecoverableGeneration(generation, generationsRoot); problem != nil {
			rows.Close()
			return 0, problem
		}
		seen[generation.ID] = generation
		generations = append(generations, generation)
	}
	rows.Close()
	if len(generations) != generationCount {
		return 0, exit.Named(exit.Conflict, "package_inventory_source_changed",
			"package inventory source changed while it was being read")
	}
	pinRows, err := source.Query(`SELECT package,major,generation,activated_at FROM pins ORDER BY package`)
	if err != nil {
		return 0, exit.Named(exit.Validation, "package_inventory_source_unreadable",
			"cannot read source package pins: %s", err)
	}
	var pins []Pin
	packages := map[string]bool{}
	for pinRows.Next() {
		var pin Pin
		var generation PackageInstall
		exists := false
		scanErr := pinRows.Scan(&pin.Package, &pin.Major, &pin.InstallID, &pin.ActivatedAt)
		if scanErr == nil {
			generation, exists = seen[pin.InstallID]
		}
		if scanErr != nil || pin.Package == "" || packages[pin.Package] || !exists ||
			generation.Package != pin.Package || generation.Major != pin.Major {
			pinRows.Close()
			return 0, exit.Named(exit.Validation, "package_inventory_source_pin_invalid",
				"source package pin is duplicate, malformed, or names an absent generation")
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
	missingGenerations := make([]PackageInstall, 0, len(generations))
	for _, generation := range generations {
		current, scanErr := scanGen(tx.QueryRow(
			`SELECT `+genCols("")+` FROM install_generations WHERE id=?`, generation.ID))
		switch {
		case errors.Is(scanErr, sql.ErrNoRows):
			missingGenerations = append(missingGenerations, generation)
		case scanErr != nil:
			return 0, exit.Internalf("cannot inspect current package generation %s: %s",
				generation.ID, scanErr)
		case current != generation:
			return 0, exit.Named(exit.Conflict, "package_generation_recovery_collision",
				"current package generation %s differs from the verified backup row", generation.ID).
				WithRemedy("keep the current generation or recover a backup whose immutable row is identical")
		}
	}
	missingPins := make([]Pin, 0, len(pins))
	for _, pin := range pins {
		var current Pin
		scanErr := tx.QueryRow(`SELECT package,major,generation,activated_at FROM pins WHERE package=?`,
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
	for _, generation := range missingGenerations {
		verified := 0
		if generation.Verified {
			verified = 1
		}
		if _, err := tx.Exec(`INSERT INTO install_generations(`+genCols("")+`) VALUES(`+placeholders()+`)`,
			generation.ID, generation.Package, generation.Major, generation.Version,
			generation.SourceKind, generation.SourceRef, generation.SourceDigest, verified,
			generation.Dir, generation.Python, generation.Runtime, generation.ProjectDir,
			generation.UV, generation.LockDigest, generation.Platform, generation.Extra,
			generation.Packages, generation.Closure, generation.PackageDescriptor,
			generation.PlacementSetDigest, generation.BytesExcl, generation.BytesShared,
			generation.CreatedAt); err != nil {
			return 0, exit.Internalf("cannot recover package generation %s: %s", generation.ID, err)
		}
	}
	for _, pin := range missingPins {
		if _, err := tx.Exec(`INSERT INTO pins(package,major,generation,activated_at) VALUES(?,?,?,?)`,
			pin.Package, pin.Major, pin.InstallID, pin.ActivatedAt); err != nil {
			return 0, exit.Internalf("cannot recover package pin %s: %s", pin.Package, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return 0, exit.New(exit.Conflict, "package inventory recovery did not commit: %s", err)
	}
	return len(missingGenerations), nil
}

func verifyRecoverableGeneration(generation PackageInstall, generationsRoot string) *exit.Error {
	if generation.ID == "" || generation.Package == "" || generation.Version == "" ||
		generation.SourceDigest == "" || generation.PackageDescriptor == "" ||
		generation.SourceKind != "tensorhub" && generation.SourceKind != "local" {
		return exit.Named(exit.Validation, "package_generation_recovery_invalid",
			"backup generation %q has incomplete immutable identity", generation.ID)
	}
	if !generation.Verified {
		return exit.Named(exit.Validation, "package_generation_recovery_unverified",
			"backup generation %s is a local or otherwise unverified install", generation.ID).
			WithRemedy("reinstall that local package from its source; recovery imports only verified custody")
	}
	wantDir := filepath.Join(filepath.Clean(generationsRoot), generation.ID)
	if !filepath.IsAbs(generation.Dir) || filepath.Clean(generation.Dir) != wantDir {
		return exit.Named(exit.Validation, "package_generation_recovery_path_invalid",
			"backup generation %s does not name its exact generation directory", generation.ID)
	}
	info, err := os.Lstat(generation.Dir)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return exit.Named(exit.Validation, "package_generation_recovery_path_invalid",
			"backup generation %s directory is absent, not a directory, or a symlink", generation.ID)
	}
	if problem := verifyRecoveryDigest(filepath.Join(generation.Dir, "documents", "descriptor.json"),
		generation.PackageDescriptor); problem != nil {
		return problem
	}
	if generation.PlacementSetDigest != "" {
		if problem := verifyRecoveryDigest(filepath.Join(generation.Dir, "artifact-cache",
			strings.TrimPrefix(generation.PlacementSetDigest, "sha256:")),
			generation.PlacementSetDigest); problem != nil {
			return problem
		}
	}
	return nil
}

func verifyRecoveryDigest(path, spelled string) *exit.Error {
	want, err := hex.DecodeString(strings.TrimPrefix(spelled, "sha256:"))
	if err != nil || len(want) != sha256.Size {
		return exit.Named(exit.Validation, "package_generation_recovery_digest_invalid",
			"backup generation declares malformed digest %q", spelled)
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 ||
		info.Size() < 1 || info.Size() > 16<<20 {
		return exit.Named(exit.Validation, "package_generation_recovery_evidence_invalid",
			"generation evidence %s is absent, unsafe, or over the 16 MiB bound", path)
	}
	raw, err := os.ReadFile(path)
	measured := sha256.Sum256(raw)
	if err != nil || !bytes.Equal(want, measured[:]) {
		return exit.Named(exit.Validation, "package_generation_recovery_evidence_invalid",
			"generation evidence %s is absent or does not match %s", path, spelled)
	}
	return nil
}
