package localpackage

import (
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// ExecutionCapture contains only frozen code facts. Runtime chooses call order,
// arguments and machine admission after accepting these bytes. Files retain
// their local transfer paths here; those paths never enter the wire document.
type ExecutionCapture struct {
	Canonical []byte
	Digest    []byte
	Revisions []Revision
}

// CaptureExecution closes the existing intake's immutable child bindings without
// importing code, looking up mutable package pins, or executing future calls.
// Runtime-owned builtins are verified from Runtime's own installed inventory.
func CaptureExecution(rootInstall string, root Revision,
	bindings func(string) ([]records.ChildBinding, *exit.Error),
	resolve func(string, string) (Revision, *exit.Error),
) (ExecutionCapture, *exit.Error) {
	type pending struct {
		install  string
		revision Revision
	}
	queue := []pending{{rootInstall, root}}
	seen := map[string]bool{}
	callables := map[string]*pb.MachineCallableBinding{}
	revisions := map[string]Revision{}
	document := &pb.MachineExecutionCapture{}
	digest, err := canonical.Raw(root.Digest)
	if err != nil || rootInstall == "" || bindings == nil || resolve == nil {
		return ExecutionCapture{}, exit.New(exit.Validation, "machine capture requires an exact root revision")
	}
	document.RootRevisionDigest = digest
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		key := current.install + "/" + current.revision.Digest
		if seen[key] {
			continue
		}
		seen[key] = true
		if len(seen) > 128 {
			return ExecutionCapture{}, exit.New(exit.Validation, "machine capture exceeds 128 package revisions")
		}
		revisions[current.revision.Digest] = current.revision
		rows, problem := bindings(current.install)
		if problem != nil {
			return ExecutionCapture{}, problem
		}
		for _, row := range rows {
			if row.ParentInstallID != current.install {
				return ExecutionCapture{}, exit.New(exit.Conflict, "captured binding belongs to a different parent")
			}
			child := current.revision
			if row.ChildInstallID != current.install {
				child, problem = resolve(row.ChildInstallID, row.LocalRevisionDigest)
				if problem != nil {
					return ExecutionCapture{}, problem
				}
				if child.Digest != row.LocalRevisionDigest {
					return ExecutionCapture{}, exit.New(exit.Conflict, "captured dependency revision changed")
				}
			} else if row.LocalRevisionDigest != "" && row.LocalRevisionDigest != child.Digest {
				return ExecutionCapture{}, exit.New(exit.Conflict, "self binding changed its captured revision")
			}
			caller, callerErr := canonical.Raw(current.revision.Digest)
			callee, calleeErr := canonical.Raw(child.Digest)
			iface, interfaceErr := canonical.Raw(row.InterfaceDigest)
			if callerErr != nil || calleeErr != nil || interfaceErr != nil || row.Module == "" || row.Export == "" || row.Entrypoint == "" {
				return ExecutionCapture{}, exit.New(exit.Validation, "machine callable binding has invalid identity")
			}
			if row.InterfaceDigest != child.PackageInterfaceDigest {
				return ExecutionCapture{}, exit.New(exit.Conflict, "captured callable interface differs from its revision")
			}
			value := &pb.MachineCallableBinding{
				CallerRevisionDigest: caller, InterfaceDigest: iface, Module: row.Module,
				Export: row.Export, CalleeRevisionDigest: callee, Entrypoint: row.Entrypoint,
			}
			callKey := strings.Join([]string{current.revision.Digest, row.InterfaceDigest, row.Module, row.Export}, "\x00")
			if prior := callables[callKey]; prior != nil {
				if !proto.Equal(prior, value) {
					return ExecutionCapture{}, exit.New(exit.Conflict, "machine capture changed a callable binding")
				}
			} else {
				callables[callKey] = value
				document.Bindings = append(document.Bindings, value)
			}
			queue = append(queue, pending{row.ChildInstallID, child})
		}
	}
	keys := make([]string, 0, len(revisions))
	for key := range revisions {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := ExecutionCapture{}
	for _, key := range keys {
		revision := revisions[key]
		verified, raw, problem := identity(revision.Package, revision.Release, revision.SourceDigest,
			revision.PackageInterfaceDigest, revision.PackageInterfaceLength, revision.Files, revision.DependencyRequirements)
		if problem != nil {
			return ExecutionCapture{}, problem
		}
		if verified.Digest != revision.Digest {
			return ExecutionCapture{}, exit.New(exit.Conflict, "machine capture revision inventory changed")
		}
		var value pb.LocalPackageRevision
		if err := canonical.Unmarshal(raw, &value); err != nil {
			return ExecutionCapture{}, exit.Internalf("cannot encode captured revision: %s", err)
		}
		document.Revisions = append(document.Revisions, &value)
		result.Revisions = append(result.Revisions, revision)
	}
	bindingKey := func(v *pb.MachineCallableBinding) string {
		return strings.Join([]string{string(v.CallerRevisionDigest), string(v.InterfaceDigest), v.Module, v.Export}, "\x00")
	}
	sort.Slice(document.Bindings, func(i, j int) bool { return bindingKey(document.Bindings[i]) < bindingKey(document.Bindings[j]) })
	for i := 1; i < len(document.Bindings); i++ {
		if bindingKey(document.Bindings[i-1]) == bindingKey(document.Bindings[i]) {
			return ExecutionCapture{}, exit.New(exit.Conflict, "machine capture repeats a callable binding")
		}
	}
	result.Canonical, result.Digest, err = canonical.Identity(document)
	if err != nil {
		return ExecutionCapture{}, exit.Internalf("cannot encode machine capture: %s", err)
	}
	if len(result.Canonical) > 1<<20 {
		return ExecutionCapture{}, exit.New(exit.Validation, "machine capture exceeds 1 MiB")
	}
	return result, nil
}
