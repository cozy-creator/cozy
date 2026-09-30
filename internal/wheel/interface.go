package wheel

import (
	"archive/zip"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

// EmbedInterface adds a source-derived interface to a newly built project wheel.
// A backend-provided interface or signature keeps the original wheel authoritative.
// Callers must never use this operation for prebuilt artifacts supplied by the user.
func EmbedInterface(source, target string, document []byte) (string, *exit.Error) {
	if len(document) == 0 || len(document) > 1<<20 || !json.Valid(document) {
		return "", wheelStructure("source package interface is not bounded JSON")
	}
	if _, problem := InspectIdentity(source); problem != nil {
		return "", problem
	}
	reader, err := zip.OpenReader(source)
	if err != nil {
		return "", wheelStructure("cannot read source project wheel")
	}
	defer reader.Close()
	for _, member := range reader.File {
		if distInfoMember(member.Name, "package-interface.json") ||
			distInfoMember(member.Name, "RECORD.jws") || distInfoMember(member.Name, "RECORD.p7s") {
			return source, nil
		}
	}
	if problem := rewriteMetadata(source, target, map[string][]byte{"package-interface.json": document}); problem != nil {
		return "", problem
	}
	return target, nil
}
