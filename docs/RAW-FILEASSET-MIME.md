# Preserve the standard type of unrecognized binary inputs

An ordinary CLI request attaching an exact 131072-byte raw .bin FileAsset was rejected
before submission despite an authored application/octet-stream policy. The byte sniffer
already returns that standard fallback, but inputasset.normalizeMediaType replaces it
with an empty string. Launch then correctly refuses the empty type against the authored list.

Retain the sniffed application/octet-stream value for unknown binary content. Keep known
image/text/audio sniffing and aliases, original bytes, exact hashes, file identity checks,
max_bytes and the authored MIME list. This is a classification fix, not permission to remove
or broaden the author's bound. Existing accepted records remain their recorded identities.

Proof covers Probe/Fingerprint/Bind on opaque bytes, explicit octet-stream admission through
the ordinary typed FileAsset CLI/machine/API path, unchanged file identity after reconnect,
and rejection of an over-bound input or a known incompatible media type. CPU files are
diagnostic fixtures; model/GPU inference and VAE fidelity remain independent gates.

Tracker cozy-creator/tracker#322. Design is recorded before implementation and proof.

CONFIRMED CPU: inputasset/records component packages pass. The real typed FileAsset CLI
test `TestRawFileAssetRetainsOctetStreamAndAuthoredBoundsThroughOrdinaryCLI` passes in 7.93 s
with the exact named machine3e7 binary, Runtime H370 SHA0d50f6b1 and TensorFSf089 wheels.
It checks original bytes/digest/type on the managed machine and after controller restart,
then rejects131073 bytes and a real PNG renamed .bin against the same unchanged policy.
CUDA_VISIBLE_DEVICES is empty and no GPU/model operation is performed. Logs are retained
under outputs/codex-machine-audit-20261004/api/raw-fileasset-{components,ordinary}.log.
