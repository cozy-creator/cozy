package accountauth

import (
	"net/http"
	"testing"
)

func TestAuthRefusalCarriesServerRetryInterval(t *testing.T) {
	problem := authRefusal(http.StatusTooManyRequests, []byte(`{
		"error": {
			"code": "rate_limited",
			"message": "Too many requests. Please try again later.",
			"metadata": {"retry_after_seconds": 36}
		}
	}`))
	if problem.ErrName() != "rate_limited" || problem.Remedy != "retry after 36 seconds" {
		t.Fatalf("rate-limit refusal = %#v", problem)
	}
}
