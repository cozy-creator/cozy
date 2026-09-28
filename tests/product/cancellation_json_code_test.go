package producttest

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestCancellationPresentationPreservesMachineVocabulary(t *testing.T) {
	root := t.TempDir()
	startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for i, tc := range []struct{ actor, want string }{
		{"cozy run cancel", "cancelled by user"}, {"cozy job cancel", "cancelled by user"},
		{"cozy down --all", "cancelled by user"}, {"cozy rental end fixture", "cancelled by user"},
		{"deadline", "cancelled by system"}, {"provider timeout", "cancelled by system"}, {"", "cancelled"},
	} {
		id := fmt.Sprintf("job-cancellation-presentation-%d", i)
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: childDigest("c"), Package: "proof/presentation", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`)})
		fatal(t, problem)
		changed, problem := store.CancelQueuedRequest(id, map[string]any{"actor": tc.actor, "status": "CANCELED", "outputs": []any{}, "requeuing": false})
		fatal(t, problem)
		if !changed {
			t.Fatal("fixture did not reach terminal cancellation")
		}
		code, stdout, stderr := runCozyStreams(t, root, "run", "watch", id)
		said := tc.want
		if tc.actor != "" {
			said = "was " + tc.want // an attributed cancellation names who did it
		}
		if code != 1 || !strings.Contains(stdout+stderr, said) {
			t.Fatalf("actor %q: %d %s %s", tc.actor, code, stdout, stderr)
		}
		_, document := runCozy(t, root, "run", "watch", id, "--json")
		if !strings.Contains(document, `"code":"canceled"`) {
			t.Fatalf("human spelling changed machine vocabulary: %s", document)
		}
		// Neither projection rewrites the lifecycle facts it renders.
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		actor, _, _, problem := store.CancelAttribution(id)
		fatal(t, problem)
		if row.State != "canceled" || actor != tc.actor {
			t.Fatalf("projection rewrote lifecycle facts: state=%s actor=%q", row.State, actor)
		}
	}
}
