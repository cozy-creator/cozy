package cli

import localapi "github.com/cozy-creator/cozy/internal/client"

// Project the current manual stop for presentation only. The durable historical
// event, its id, its reason and its occurrence timestamp are unchanged.
func publicFailureEvent(event localapi.Event) localapi.Event {
	event.Type = "run.failed"
	payload := make(map[string]any, len(event.Payload))
	for key, value := range event.Payload {
		payload[key] = value
	}
	payload["status"] = "FAILED"
	event.Payload = payload
	return event
}

// Older daemons report a current manual stop as blocked without the optional
// event identity. They already report ambiguous acceptance as queued.
func currentManualStop(status string, stoppedEventID, eventID int64) bool {
	return status == "blocked" || status == "failed" && stoppedEventID == eventID
}

func publicObservedStatus(status string) string {
	if status == "blocked" {
		return "failed"
	}
	return status
}
