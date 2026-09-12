package cli

import "testing"

func TestHumanCancellationStatus(t *testing.T) {
	for _, tc := range []struct{ actor, want string }{
		{"cozy run cancel", "cancelled by user"},
		{"cozy job cancel", "cancelled by user"},
		{"cozy down --all", "cancelled by user"},
		{"cozy rental end darkrai", "cancelled by user"},
		{"deadline", "cancelled by system"},
		{"provider timeout", "cancelled by system"},
		{"", "cancelled"},
	} {
		if got := humanCancellationStatus(tc.actor); got != tc.want {
			t.Errorf("humanCancellationStatus(%q)=%q, want %q", tc.actor, got, tc.want)
		}
	}
}

func TestHumanRentalState(t *testing.T) {
	if got := humanRentalState("release_requested"); got != "draining" {
		t.Fatalf("release_requested displayed as %q, want draining", got)
	}
	if got := humanRentalState("ready"); got != "ready" {
		t.Fatalf("ready displayed as %q", got)
	}
}
