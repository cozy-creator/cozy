package cli

import "testing"

// PRICES ARE READ IN PENNIES. Micro-dollar precision is how the provider quotes and
// how we bill; `$0.463504/hour` asks a reader to parse six decimals to learn "about
// forty-six cents". Rounding is for DISPLAY only -- caps, comparisons and the ledger
// upstream still work in whole micros.
func TestUSDPerHourBareRoundsToPennies(t *testing.T) {
	for _, row := range []struct {
		name   string
		micros int64
		want   string
	}{
		{"rtx-a4000 total", 463504, "$0.46"},
		{"rtx-4090 total", 953504, "$0.95"},
		{"storage adder", 213504, "$0.21"},
		{"b200 total", 7003504, "$7.00"},
		{"cpu total", 72780, "$0.07"},
		{"spend cap", 10_000_000, "$10.00"},
		{"rounds up at half a penny", 465000, "$0.47"},
		{"exact penny", 460000, "$0.46"},
		{"zero is free", 0, "$0.00"},
		// A rate that is real but under a penny must not render as free: the CPU
		// storage adder is $0.00278/hour and "$0.00" would say the wrong thing.
		{"sub-penny is not free", 2780, "<$0.01"},
	} {
		t.Run(row.name, func(t *testing.T) {
			if got := usdPerHourBare(row.micros); got != row.want {
				t.Fatalf("usdPerHourBare(%d) = %q, want %q", row.micros, got, row.want)
			}
		})
	}
}
