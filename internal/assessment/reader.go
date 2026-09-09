package assessment

import (
	"bytes"
	"context"
	"os/exec"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Inspect invokes the host's trusted evaluator, never the submitted package's
// interpreter. Only immutable report bytes cross stdin; diagnostics expose no
// report content, path, URL or user-authored exception.
func Inspect(ctx context.Context, report []byte, env []string) ([]byte, *exit.Error) {
	if len(report) == 0 || len(report) > MaxBytes {
		return nil, exit.New(exit.Validation, "assessment report exceeds its artifact bound")
	}
	bin, err := exec.LookPath("cozy-eval")
	if err != nil {
		return nil, exit.Named(exit.Structural, "assessment.reader_missing", "cozy-eval report inspector is not installed on the client").WithRemedy("install cozy-eval 0.6.0 or newer with uv tool install")
	}
	command := exec.CommandContext(ctx, bin, "report", "inspect")
	command.Env = append([]string{}, env...)
	command.Stdin = bytes.NewReader(report) //cozy:stdin-value immutable native report for the trusted evaluator reader
	command.WaitDelay = 250 * time.Millisecond
	var output boundedReport
	command.Stdout = &output
	if err := command.Run(); err != nil {
		return nil, exit.Named(exit.Validation, "assessment.report_invalid", "the trusted evaluator refused this assessment report")
	}
	return output.Bytes(), nil
}

type boundedReport struct{ buffer bytes.Buffer }

func (b *boundedReport) Write(value []byte) (int, error) {
	if len(value) > MaxBytes-b.buffer.Len() {
		return 0, exit.New(exit.Validation, "assessment inspection exceeds its bound")
	}
	return b.buffer.Write(value)
}

func (b *boundedReport) Bytes() []byte { return b.buffer.Bytes() }
