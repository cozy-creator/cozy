package launch

import (
	"encoding/json"
	"testing"
)

func TestAssetKindAndBoundFollowExactNestedField(t *testing.T) {
	var ep Entrypoint
	if err := json.Unmarshal([]byte(`{
      "name":"run",
      "request":{"struct":"Request","fields":[
        {"name":"image","type":{"asset":"image"},"asset_bound":{"max_bytes":100},"wire":"required"},
        {"name":"references","type":{"list":{"struct":"Reference","fields":[
          {"name":"video","type":{"asset":"video"},"asset_bound":{"max_bytes":200},"wire":"required"}
        ]}},"wire":"required"}
      ]}
    }`), &ep); err != nil {
		t.Fatal(err)
	}
	if kind, limit, ok := AssetKind(&ep, "image"); !ok || kind != "image" || limit != 100 {
		t.Fatalf("image = %s %d %v", kind, limit, ok)
	}
	if kind, limit, ok := AssetKind(&ep, "references.3.video"); !ok || kind != "video" || limit != 200 {
		t.Fatalf("video = %s %d %v", kind, limit, ok)
	}
	if _, _, ok := AssetKind(&ep, "references.x.video"); ok {
		t.Fatal("nonnumeric list path accepted")
	}
}

func TestValidatePayloadRejectsUnknownAndWrongTypes(t *testing.T) {
	ep := &Entrypoint{Name: "run", Request: Struct{Fields: []Field{
		{Name: "prompt", Type: json.RawMessage(`"str"`), Wire: "required"},
		{Name: "seed", Type: json.RawMessage(`"int"`), Wire: "optional"},
	}}}
	for _, payload := range []json.RawMessage{
		[]byte(`{}`), []byte(`{"prompt":3}`), []byte(`{"prompt":"ok","unknown":true}`),
	} {
		if problem := ValidatePayload(ep, payload); problem == nil {
			t.Fatalf("payload accepted: %s", payload)
		}
	}
	if problem := ValidatePayload(ep, []byte(`{"prompt":"ok","seed":4}`)); problem != nil {
		t.Fatal(problem)
	}
}

func TestValidatePayloadEnforcesRecordedConstraints(t *testing.T) {
	var ep Entrypoint
	if err := json.Unmarshal([]byte(`{
      "name":"run",
      "request":{"struct":"Request","fields":[
        {"name":"prompt","type":"str","constraints":{"min_length":2,"max_length":4},"wire":"required"},
        {"name":"videos","type":{"list":{"asset":"video"}},"constraints":{"min_length":2,"max_length":3},"wire":"required"},
        {"name":"steps","type":"int","constraints":{"ge":1,"le":8},"wire":"required"}
      ]}
    }`), &ep); err != nil {
		t.Fatal(err)
	}
	good := []byte(`{"prompt":"shot","videos":["sha256:a","sha256:b"],"steps":8}`)
	if problem := ValidatePayload(&ep, good); problem != nil {
		t.Fatal(problem)
	}
	for _, payload := range []json.RawMessage{
		[]byte(`{"prompt":"x","videos":["a","b"],"steps":1}`),
		[]byte(`{"prompt":"shot","videos":["a"],"steps":1}`),
		[]byte(`{"prompt":"shot","videos":["a","b"],"steps":9}`),
	} {
		if problem := ValidatePayload(&ep, payload); problem == nil {
			t.Fatalf("constrained payload accepted: %s", payload)
		}
	}
}
