package cli

import localapi "github.com/cozy-creator/cozy/internal/client"

// Project the current manual stop for presentation only. The durable historical
// event, its id, its reason and its occurrence timestamp are unchanged.
func publicFailureEvent(event localapi.Event) localapi.Event {
	event.Type = "request.failed"
	payload := make(map[string]any, len(event.Payload))
	for key, value := range event.Payload {
		payload[key] = value
	}
	payload["status"] = "FAILED"
	event.Payload = payload
	return event
}
