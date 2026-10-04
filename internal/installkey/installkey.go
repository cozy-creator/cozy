// Package installkey is this CLI install's one Ed25519 control key, `<home>/id_ed25519`, as
// ~/.ssh/id_ed25519 is a user's: each Hub account registers its public half as a device key
// (login), a rental's machine admits the account's device keys, and this computer's machine
// admits the keys in its authorized_keys. Every machine call is signed with it.
package installkey

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"golang.org/x/crypto/ssh"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machinev1"
)

// File is the private key's name in the CLI home; File+".pub" is its authorized_keys line.
const File = "id_ed25519"

// Key is the install key. Its field stays private so formatting cannot expose it.
type Key struct{ private ed25519.PrivateKey }

func (k Key) Public() ed25519.PublicKey { return k.private.Public().(ed25519.PublicKey) }

// PublicKey is the raw public key as unpadded base64url, the Hub's and lease's spelling.
func (k Key) PublicKey() string { return base64.RawURLEncoding.EncodeToString(k.Public()) }

func (k Key) Seed() []byte { return k.private.Seed() }

func (k Key) Sign(message []byte) []byte { return ed25519.Sign(k.private, message) }

func (k Key) Private() ed25519.PrivateKey { return k.private }

// Signer is this key as the signer of a machine's Cozy-Caps.
func (k Key) Signer() machinev1.Signer { return machinev1.Signer{Public: k.Public(), Sign: k.Sign} }

// AuthorizedKey is the key as one authorized_keys line: `ssh-ed25519 <base64> <comment>`.
func (k Key) AuthorizedKey() string {
	public, _ := ssh.NewPublicKey(k.Public())
	return strings.TrimSpace(string(ssh.MarshalAuthorizedKey(public))) + " " + comment()
}

// Names reports whether an authorized_keys line names this key, whatever its comment.
func (k Key) Names(line string) bool {
	have, mine := strings.Fields(line), strings.Fields(k.AuthorizedKey())
	return len(have) >= 2 && have[0] == mine[0] && have[1] == mine[1]
}

// Load reads the install key; it never creates one.
func Load(home string) (Key, *exit.Error) {
	path := filepath.Join(home, File)
	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return Key{}, exit.Named(exit.Credential, "auth.install_key_missing",
			"this install has no control key (%s)", path).WithNext("cozy auth login <email>")
	}
	if err != nil {
		return Key{}, exit.Named(exit.Credential, "auth.install_key_unreadable", "%s: %s", path, err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return Key{}, exit.Named(exit.Credential, "auth.install_key_permissions",
			"%s is mode %#o, not 0600", path, info.Mode().Perm()).WithRemedy("chmod 600 %s", path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Key{}, exit.Named(exit.Credential, "auth.install_key_unreadable", "%s: %s", path, err)
	}
	parsed, err := ssh.ParseRawPrivateKey(raw)
	if err == nil {
		switch key := parsed.(type) {
		case *ed25519.PrivateKey:
			return Key{*key}, nil
		case ed25519.PrivateKey:
			return Key{key}, nil
		}
	}
	return Key{}, exit.Named(exit.Credential, "auth.install_key_invalid", "%s is not one OpenSSH Ed25519 private key", path)
}

// Ensure loads the install key, creating it on first use: from the device key a login made
// before install keys (the default Hub's, else the only login's), so that registration carries
// over, else at random.
func Ensure(home string) (Key, *exit.Error) {
	key, problem := Load(home)
	if problem == nil || problem.ErrName() != "auth.install_key_missing" {
		return key, problem
	}
	seed := loginSeed(home)
	private := ed25519.PrivateKey(nil)
	if len(seed) == ed25519.SeedSize {
		private = ed25519.NewKeyFromSeed(seed)
	} else if _, generated, err := ed25519.GenerateKey(rand.Reader); err == nil {
		private = generated
	} else {
		return Key{}, exit.Internalf("cannot generate the install key: %s", err)
	}
	block, err := ssh.MarshalPrivateKey(private, comment())
	if err != nil {
		return Key{}, exit.Internalf("cannot encode the install key: %s", err)
	}
	if err := os.MkdirAll(home, 0o700); err != nil {
		return Key{}, exit.Internalf("cannot create %s: %s", home, err)
	}
	path := filepath.Join(home, File)
	staged, err := os.CreateTemp(home, "."+File+"-*")
	if err != nil {
		return Key{}, exit.Internalf("cannot write the install key: %s", err)
	}
	_, err = staged.Write(pem.EncodeToMemory(block))
	if closeErr := staged.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		// Two first uses at once keep whichever key landed first.
		err = os.Link(staged.Name(), path)
	}
	_ = os.Remove(staged.Name())
	if err != nil && !errors.Is(err, os.ErrExist) {
		return Key{}, exit.Internalf("cannot install the control key: %s", err)
	}
	if key, problem = Load(home); problem != nil {
		return Key{}, problem
	}
	if err := os.WriteFile(path+".pub", []byte(key.AuthorizedKey()+"\n"), 0o644); err != nil {
		return Key{}, exit.Internalf("cannot write the install public key: %s", err)
	}
	return key, nil
}

// Retire deletes the install key (logout); the next login creates another.
func Retire(home string) *exit.Error {
	for _, name := range []string{File, File + ".pub"} {
		if err := os.Remove(filepath.Join(home, name)); err != nil && !errors.Is(err, os.ErrNotExist) {
			return exit.Internalf("cannot retire the install key: %s", err)
		}
	}
	return nil
}

// loginSeed is the device key of this home's login at the default Hub, else of its only login.
func loginSeed(home string) []byte {
	paths, _ := filepath.Glob(filepath.Join(home, "auth", "*.json"))
	seeds := map[string][]byte{}
	for _, path := range paths {
		var login struct {
			Hub        string `json:"hub"`
			PrivateKey string `json:"private_key"`
		}
		raw, err := os.ReadFile(path)
		if err != nil || json.Unmarshal(raw, &login) != nil {
			continue
		}
		if seed, err := base64.RawURLEncoding.DecodeString(login.PrivateKey); err == nil && len(seed) == ed25519.SeedSize {
			seeds[login.Hub] = seed
		}
	}
	if seed, ok := seeds[config.DefaultHubURL]; ok {
		return seed
	}
	for _, seed := range seeds {
		if len(seeds) == 1 {
			return seed
		}
	}
	return nil
}

func comment() string {
	host, _ := os.Hostname()
	if host == "" {
		return "cozy"
	}
	return "cozy@" + host
}
