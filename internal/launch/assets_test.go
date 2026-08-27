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
        {"name":"image","type":{"asset":"image"},"asset_bound":{"max_bytes":100,"media_types":["image/png"]},"wire":"required"},
        {"name":"references","type":{"list":{"struct":"Reference","fields":[
          {"name":"video","type":{"asset":"video"},"asset_bound":{"max_bytes":200},"wire":"required"}
        ]}},"wire":"required"}
      ]}
    }`), &ep); err != nil {
		t.Fatal(err)
	}
	if spec, ok := AssetSpec(&ep, "image"); !ok || spec.Kind != "image" || spec.MaxBytes != 100 ||
		!spec.AcceptsMediaType("image/png") || spec.AcceptsMediaType("image/jpeg") {
		t.Fatalf("image = %#v %v", spec, ok)
	}
	if spec, ok := AssetSpec(&ep, "references.3.video"); !ok || spec.Kind != "video" || spec.MaxBytes != 200 {
		t.Fatalf("video = %#v %v", spec, ok)
	}
	if _, ok := AssetSpec(&ep, "references.x.video"); ok {
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

func TestValidatePayloadFailsClosedOnlyWhenUnknownConstraintIsUsed(t *testing.T) {
	var ep Entrypoint
	if err := json.Unmarshal([]byte(`{"name":"run","request":{"fields":[
      {"name":"future","type":"str","constraints":{"pattern":"x+"},"wire":"optional"}
    ]}}`), &ep); err != nil {
		t.Fatal(err)
	}
	if problem := ValidatePayload(&ep, []byte(`{}`)); problem != nil {
		t.Fatalf("unused future field rejected descriptor: %v", problem)
	}
	if problem := ValidatePayload(&ep, []byte(`{"future":"x"}`)); problem == nil || problem.ErrName() != "descriptor_constraint_unknown" {
		t.Fatalf("unknown constraint = %#v", problem)
	}
}

func TestPopulatedAssetPathsUsesTheValidationWalk(t *testing.T) {
	var ep Entrypoint
	if err := json.Unmarshal([]byte(`{"name":"run","request":{"fields":[
      {"name":"first","type":{"union":["null",{"asset":"image"}]},"wire":"optional"},
      {"name":"references","type":{"list":{"fields":[
        {"name":"video","type":{"asset":"video"},"wire":"required"}
      ]}},"wire":"required"}
    ]}}`), &ep); err != nil {
		t.Fatal(err)
	}
	payload := []byte(`{"first":null,"references":[{"video":"sha256:a"},{"video":"sha256:b"}]}`)
	if problem := ValidatePayload(&ep, payload); problem != nil {
		t.Fatal(problem)
	}
	paths, problem := PopulatedAssetPaths(&ep, payload)
	if problem != nil || len(paths) != 2 || paths[0] != "references.0.video" ||
		paths[1] != "references.1.video" {
		t.Fatalf("paths=%#v problem=%v", paths, problem)
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
	if problem := ValidatePayload(&ep,
		[]byte(`{"prompt":"🙂🙂🙂🙂","videos":["a","b"],"steps":1}`)); problem != nil {
		t.Fatalf("four Unicode code points: %v", problem)
	}
	if problem := ValidatePayload(&ep,
		[]byte(`{"prompt":"🙂🙂🙂🙂🙂","videos":["a","b"],"steps":1}`)); problem == nil {
		t.Fatal("five Unicode code points passed max_length=4")
	}
}
