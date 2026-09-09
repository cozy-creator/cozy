package cli

import "strings"

// humanCancellationActor is the one presentation mapping for cancellation
// attribution. Storage, API and worker protocol retain their exact actor/source;
// human output uses a stable audience-facing vocabulary.
func humanCancellationActor(actor string) string {
	actor = strings.ToLower(strings.TrimSpace(actor))
	if actor == "" {
		return "system"
	}
	for _, marker := range []string{"user", "cozy run cancel", "cozy job cancel", "cozy down --all", "cozy rental end", "sigint", "interrupt"} {
		if strings.Contains(actor, marker) {
			return "user"
		}
	}
	return "system"
}

func humanCancellationStatus(actor string) string {
	if actor == "" {
		return "cancelled"
	}
	return "cancelled by " + humanCancellationActor(actor)
}

// humanRentalState intentionally differs from the machine-readable hub state.
// release_requested means no new work should be admitted while the provider
// teardown is converging; "draining" is the concise state a person can act on.
func humanRentalState(state string) string {
	if state == "release_requested" {
		return "draining"
	}
	return state
}
