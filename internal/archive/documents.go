// Package archive reads this controller's retained records, never a worker RPC.
package archive

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/custody"
)

type Doc = canonical.Doc
type Value = canonical.Value

// These names belong to existing stored document bytes, not current transport identity.
const (
	TerminalBody   = "cozy.worker.v1.AttemptOutcomeBody/1"
	Invocation     = "cozy.worker.v1.InvocationSpec/1"
	Placement      = "cozy.worker.v1.PlacementSet/1"
	WeightsReceipt = "cozy.worker.v1.WeightsReceipt/1"
)

func Read(data []byte, format string) (Doc, error) {
	doc, err := canonical.ReadObject(data)
	if err != nil {
		return nil, err
	}
	if doc.Str("format") != format {
		return nil, refuse("unknown_format", "%v is not %q", doc["format"], format)
	}
	switch format {
	case TerminalBody:
		err = weightsReceiptList(doc)
	case WeightsReceipt:
		err = weightsReceipt(doc)
	}
	return doc, err
}
func refuse(code, format string, args ...any) *canonical.Error {
	return &canonical.Error{Code: code, Detail: fmt.Sprintf(format, args...)}
}
func weightsReceiptList(d Doc) error {
	raw, present := d["weights_receipts"]
	if !present {
		return nil
	}
	items, ok := raw.([]Value)
	if !ok {
		return refuse("weights_receipt_shape", "weights_receipts is a list")
	}
	if len(items) > custody.MaxWeightsReceipts {
		return refuse("weights_receipt_count_cap", "%d receipts exceeds the %d-item cap",
			len(items), custody.MaxWeightsReceipts)
	}
	aggregate, slots := 0, map[string]bool{}
	for _, item := range items {
		fields, ok := item.(map[string]Value)
		if !ok {
			return refuse("weights_receipt_shape", "an weights_receipts item is not an object")
		}
		receipt, size, err := readWeightsReceiptRef(Doc(fields))
		if err != nil {
			return err
		}
		aggregate += size
		slot := receipt.Str("output_slot")
		if slots[slot] {
			return refuse("weights_receipt_duplicate", "output slot %q has two receipts", slot)
		}
		slots[slot] = true
	}
	if aggregate > custody.MaxWeightsReceiptAggregateBytes {
		return refuse("weights_receipt_aggregate_cap", "%d receipt bytes exceeds the %d-byte cap",
			aggregate, custody.MaxWeightsReceiptAggregateBytes)
	}
	return nil
}

func weightsReceipt(d Doc) error {
	for _, field := range []string{"owner_authority_scope", "request_id", "invocation_spec_digest",
		"output_slot", "weights_transaction_id", "tensorfs_receipt_digest",
		"tensorfs_receipt_canonical_bytes"} {
		if d.Str(field) == "" {
			return refuse("weights_receipt_incomplete", "%s is empty or absent", field)
		}
	}
	nested, err := decodeCanonicalBytes(d.Str("tensorfs_receipt_canonical_bytes"))
	if err != nil {
		return refuse("weights_receipt_nested_malformed", "%s", err)
	}
	if !sameDigest(nested, d.Str("tensorfs_receipt_digest")) {
		return refuse("weights_receipt_nested_digest_mismatch",
			"tensorfs_receipt_digest does not hash the exact carried bytes")
	}
	if _, err := canonical.Raw(d.Str("invocation_spec_digest")); err != nil {
		return refuse("weights_receipt_identity_malformed", "invocation_spec_digest: %s", err)
	}
	return nil
}

func readWeightsReceiptRef(ref Doc) (Doc, int, error) {
	if ref.Str("weights_receipt_digest") == "" || ref.Str("weights_receipt_canonical_bytes") == "" {
		return nil, 0, refuse("weights_receipt_ref_shape", "the reference carries a digest and bytes")
	}
	data, err := decodeCanonicalBytes(ref.Str("weights_receipt_canonical_bytes"))
	if err != nil {
		return nil, 0, refuse("weights_receipt_malformed", "%s", err)
	}
	if len(data) == 0 || len(data) > custody.MaxWeightsReceiptBytes {
		return nil, 0, refuse("weights_receipt_item_cap", "%d B is outside the 1..%d B range",
			len(data), custody.MaxWeightsReceiptBytes)
	}
	if !sameDigest(data, ref.Str("weights_receipt_digest")) {
		return nil, 0, refuse("weights_receipt_digest_mismatch",
			"weights_receipt_digest does not hash the exact carried bytes")
	}
	receipt, err := Read(data, WeightsReceipt)
	return receipt, len(data), err
}

func decodeCanonicalBytes(value string) ([]byte, error) {
	return base64.StdEncoding.Strict().DecodeString(value)
}

func sameDigest(data []byte, spelled string) bool {
	raw, err := canonical.Raw(spelled)
	return err == nil && bytes.Equal(canonical.Digest(data), raw)
}
