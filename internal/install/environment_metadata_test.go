package install

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEnvironmentRosterReadsOnlyDeclaredPythonMetadata(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "pyvenv.cfg"), []byte("version_info = 3.12\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, row := range []struct{ python, name, version string }{{"3.12", "Example_Library", "1.2.3"}, {"3.13", "other-interpreter", "9"}} {
		dir := filepath.Join(root, "lib", "python"+row.python, "site-packages", row.name+"-"+row.version+".dist-info")
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "METADATA"), []byte("Name: "+row.name+"\nVersion: "+row.version+"\n\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// There is deliberately no Python executable. A roster is a metadata read,
	// and cannot start that environment or discover imports through its .pth files.
	count, pins := closure(root)
	if count != 1 || pins != "example-library==1.2.3" {
		t.Fatalf("roster = %d %q", count, pins)
	}
}

func TestBaseInterpreterRefusesDependencyOwnedExecutable(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "pyvenv.cfg"), []byte("version_info = 3.12\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "bin", "python"), []byte("must not execute"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, problem := BasePython(root); problem == nil {
		t.Fatal("accepted an interpreter owned by the dependency environment")
	}
}
