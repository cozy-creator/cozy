package cli

import (
	"context"
	"slices"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
)

// A published root reaches its machine as one message: the release, the callable, the
// payload, the caller's Model choices and its inputs. The machine installs what it lacks from
// its own Hub, resolves every open slot for its own devices, prepares and mints the offer;
// nothing here reads the Hub, a ladder or a package.

// localSourceV1 writes the unpublished code of install to machine and answers its manifest's
// digest, and whether any byte moved. The machine keeps one tree per package, so only what
// changed since this daemon last sent it there is offered again.
func (m *machineRuns) localSourceV1(ctx context.Context, machine *machines.V1, install string) (string, bool, *exit.Error) {
	source, problem := m.resolver.LocalInstallation(install)
	if problem != nil {
		return "", false, problem
	}
	key := machine.Name + "\x00" + machine.BootID + "\x00" + source.Package
	held, _ := m.sent.Load(key)
	sent, _ := held.(machinev1.Held)
	if sent == nil {
		// A machine before live packages would accept the manifest and fail installing it: it
		// is asked first, and one without them is updated before the run is sent (dispatch).
		frame, err := machine.Status(ctx)
		if err != nil {
			return "", false, machines.Transport(err)
		}
		if !slices.Contains(frame.GetCapabilities(), "live-source/1") {
			return "", false, exit.Named(exit.Structural, "machine.upgrade_required",
				"%s runs TensorD %s, which installs no live package", machine.Name, frame.GetVersion())
		}
	}
	manifest, moved, now, err := machinev1.LocalSource(ctx, machine.Machine, source, sent)
	if err != nil {
		return "", false, machines.Transport(err)
	}
	m.sent.Store(key, now)
	return manifest, moved, nil
}

// forgetSent drops what machine was last sent of the request's package: the machine said its
// tree lacks a file this daemon did not offer (another computer synced it, or it was reset).
// It answers whether there was anything to forget.
func (m *machineRuns) forgetSent(machine *machines.V1, request records.Request) bool {
	_, held := m.sent.LoadAndDelete(machine.Name + "\x00" + machine.BootID + "\x00" + request.Package)
	return held
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
