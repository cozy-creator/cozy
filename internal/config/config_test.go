package config

import (
	"strings"
	"testing"
)

func TestRentalSpendCeilingIsNestedAndExact(t *testing.T) {
	file, err := strictYAML(strings.NewReader("rentals:\n  max_hourly_spend_usd: 4.125001\n"))
	if err != nil {
		t.Fatal(err)
	}
	input, err := resolve(file, &resolver{values: map[string]any{}})
	if err != nil {
		t.Fatal(err)
	}
	micros, err := usdMicros(input.RentalsMaxHourlySpendUSD)
	if err != nil || micros != 4_125_001 {
		t.Fatalf("rental spend = %d, %v", micros, err)
	}
	if _, err := strictYAML(strings.NewReader("cloud:\n  max_hourly_spend_usd: 4\n")); err == nil {
		t.Fatal("deleted cloud config key was accepted")
	}
	if _, err := usdMicros("4.0000001"); err == nil {
		t.Fatal("over-precise rental ceiling was accepted")
	}
}
