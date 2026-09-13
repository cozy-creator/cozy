package localpackage

import (
	"bufio"
	"bytes"
	"io"
	"net/textproto"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
)

// RequireRuntimeFloor verifies the controller's additional immutable execution
// constraint. Older captures remain readable history; this function never changes
// their wheels or requests. A fresh capture can independently reuse retained work.
func RequireRuntimeFloor(revision Revision, floor string) *exit.Error {
	for _, file := range revision.Files {
		if file.Kind != "project" {
			continue
		}
		raw, problem := wheel.Metadata(file.Path)
		if problem != nil {
			return problem
		}
		headers, err := textproto.NewReader(bufio.NewReader(bytes.NewReader(raw))).ReadMIMEHeader()
		if err != nil && err != io.EOF {
			return exit.New(exit.Validation, "captured project metadata is malformed")
		}
		for _, requirement := range headers.Values("Requires-Dist") {
			// This exact unmarked constraint is emitted by the capture owner alongside
			// all authored constraints. Conditional/extras bounds cannot prove it.
			if strings.TrimSpace(requirement) == "cozy-runtime>="+floor {
				return nil
			}
		}
	}
	return exit.Named(exit.Conflict, "request.capture_runtime_floor_unproven", "this immutable capture does not prove Runtime %s or newer", floor).
		WithRemedy("run the source script or editable package again to capture its current code and Runtime requirements; the prior history, TensorFS bytes and compatible completed memoized results remain retained")
}
