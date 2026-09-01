package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelproduction"
)

func TestModelTransferInstructionIsSourceFirstAndPlacementIsFrozen(t *testing.T) {
	upload := modelproduction.Instruction{Kind: "model-upload", Source: "hf://acme/model@deadbeef",
		Destination: "acme/model", Producer: "acme/tools@v2/quantize", Placement: "rental-only"}
	raw, err := upload.Bytes()
	must(t, err)
	replayed, err := modelproduction.ParseInstruction(raw)
	must(t, err)
	if replayed != upload || !strings.HasPrefix(upload.ID(), "modelupload-") {
		t.Fatalf("upload instruction changed on replay: %+v", replayed)
	}
	download := upload
	download.Kind, download.Destination, download.Placement = "model-download", "local/model", ""
	if !strings.HasPrefix(download.ID(), "modeldownload-") || download.ID() == upload.ID() {
		t.Fatal("transfer kind and destination are not part of canonical run identity")
	}
}

func TestOrdinaryProducerCarriesMultipleSourceProfilesAndNamedContracts(t *testing.T) {
	raw := []byte(`{"application":"h3:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"artifact_outputs":[{"max_bytes":4096,"mime_type":"application/vnd.cozy.model-manifest","output_id":"bf16-full","required_contract":{"encodings":["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],"topology_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}},{"max_bytes":4096,"mime_type":"application/vnd.cozy.model-manifest","output_id":"fp8-adaln-pruned","required_contract":{"encodings":["sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"],"topology_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}}],"models":[{"class":"H3Dits","component_use":{},"path":"four-lane.models.dits","source_profile":"hf/minimax-h3/native-dual-bf16/1","stamps":{}},{"class":"H3Shared","component_use":{},"path":"four-lane.models.shared","source_profile":"hf/minimax-h3/shared-bf16/1","stamps":{}}],"name":"four-lane","publishes":false,"request":{"fields":[]},"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"},"result":{"fields":[]}}]}`)
	descriptor, problem := launch.DecodeDescriptor(raw)
	fatal(t, problem)
	job, problem := descriptor.Function("four-lane")
	fatal(t, problem)
	if len(job.Models) != 2 || len(job.ArtifactOutputs) != 2 ||
		job.Models[0].SourceProfile == job.Models[1].SourceProfile ||
		job.ArtifactOutputs[1].RequiredContract == nil {
		t.Fatalf("ordinary producer hook lost its typed inputs or outputs: %+v", job)
	}
}
