package producttest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/privatepackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestPrivatePackageRevisionLifecycle(t *testing.T) {
	project := weightlessProject(t)
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()

	installForSource := func(id string) records.PackageInstall {
		t.Helper()
		pack, problem := packagepublish.PrepareLocalFrom(project)
		fatal(t, problem)
		defer pack.Close()
		digest, _, _, problem := pack.SourceIdentity()
		fatal(t, problem)
		install := records.PackageInstall{ID: id, Package: "local/" + pack.Name,
			Major: 1, Version: pack.Release, SourceKind: "local", SourceRef: project,
			SourceDigest: digest, Dir: layout.GenerationDir(id)}
		_, problem = store.Activate(install)
		fatal(t, problem)
		return install
	}

	firstInstall := installForSource("private-generation-a")
	unlock := privatepackage.Guard()
	first, problem := privatepackage.Stage(context.Background(), layout, firstInstall)
	unlock()
	fatal(t, problem)
	firstRoot := filepath.Join(layout.PrivatePackages, strings.TrimPrefix(first.Digest, "sha256:"))
	revisionBytes, err := os.ReadFile(filepath.Join(firstRoot, "revision.json"))
	must(t, err)
	revisionDoc, err := canonical.Read(revisionBytes, &pb.PrivatePackageRevision{})
	must(t, err)
	if revisionDoc.Str("source_digest") != first.SourceDigest ||
		revisionDoc.Sub("package_descriptor").Str("digest") != first.DescriptorDigest ||
		len(revisionDoc.List("files")) != len(first.Files) {
		t.Fatalf("private revision document lost exact identity: %s", revisionBytes)
	}

	request := records.Request{ID: "req-private-a", IdemKey: "private-a",
		BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: first.Package,
		Entrypoint: "tile", PlanID: "sha256:" + strings.Repeat("b", 64),
		Release: first.Release, PackageRevisionDigest: first.SourceDigest,
		PrivatePackageDigest: first.Digest, Payload: []byte(`{"size":32}`), Rental: true,
		InstallID: firstInstall.ID}
	_, fresh, problem := store.Submit(request)
	fatal(t, problem)
	if !fresh {
		t.Fatal("private request fixture replayed before its first submission")
	}

	source := filepath.Join(project, "weightless.py")
	body, err := os.ReadFile(source)
	must(t, err)
	body = []byte(strings.Replace(string(body), `REVISION = "first"`, `REVISION = "second"`, 1))
	must(t, os.WriteFile(source, body, 0o644))
	secondInstall := installForSource("private-generation-b")
	unlock = privatepackage.Guard()
	second, problem := privatepackage.Stage(context.Background(), layout, secondInstall)
	if problem == nil {
		problem = privatepackage.Sweep(layout, store)
	}
	unlock()
	fatal(t, problem)
	if _, err := os.Stat(firstRoot); err != nil {
		t.Fatalf("active request lost its old private revision: %v", err)
	}
	if first.Digest == second.Digest || first.SourceDigest == second.SourceDigest {
		t.Fatal("source edit did not create a new exact private revision")
	}

	fatal(t, store.SettleRequest(request.ID, "succeeded"))
	orphan := filepath.Join(layout.PrivatePackages, ".stage-orphan")
	must(t, os.Mkdir(orphan, 0o700))
	unlock = privatepackage.Guard()
	problem = privatepackage.Sweep(layout, store)
	unlock()
	fatal(t, problem)
	if _, err := os.Stat(firstRoot); !os.IsNotExist(err) {
		t.Fatalf("terminal superseded revision was not removed: %v", err)
	}
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Fatalf("restart sweep retained orphan staging: %v", err)
	}
	secondRoot := filepath.Join(layout.PrivatePackages, strings.TrimPrefix(second.Digest, "sha256:"))
	if _, err := os.Stat(secondRoot); err != nil {
		t.Fatalf("current editable declaration lost its revision: %v", err)
	}

	external := t.TempDir()
	marker := filepath.Join(external, "survives")
	must(t, os.WriteFile(marker, []byte("safe"), 0o600))
	malicious := filepath.Join(layout.PrivatePackages, strings.Repeat("c", 64))
	must(t, os.Symlink(external, malicious))
	unlock = privatepackage.Guard()
	problem = privatepackage.Sweep(layout, store)
	unlock()
	if problem == nil || problem.ErrName() != "private_package_revision_invalid" {
		t.Fatalf("malicious revision root was not refused: %v", problem)
	}
	if data, err := os.ReadFile(marker); err != nil || string(data) != "safe" {
		t.Fatalf("malicious revision path escaped cleanup: %q, %v", data, err)
	}
}
