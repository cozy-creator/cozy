package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
)

// A published root reaches its machine as one message: the release, the callable, the
// payload, the caller's Model choices and its inputs. The machine installs what it lacks from
// its own Hub, resolves every open slot for its own devices, prepares and mints the offer;
// nothing here reads the Hub, a ladder or a package.

// capturedRevision is the unpublished installation a root names. A root carries that one
// installation; one whose package calls other unpublished packages still goes by capture.
func (m *machineRuns) capturedRevision(request records.Request) (localpackage.Installation, *exit.Error) {
	capture, problem := m.resolver.CaptureMachineExecution(request)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	return localpackage.CapturedRoot(capture, request.LocalInstallationID)
}

// runAccount is the account owning the run at its Hub. Unpublished code has no org: the
// org-relative Model defaults of it and of its unpublished callees name this account, which
// the machine needs only for such a default and refuses without.
func (m *machineRuns) runAccount(request records.Request) string {
	caller, problem := m.resolver.namespaceAt(request.Hub)
	if problem != nil {
		return ""
	}
	return caller.Account
}
