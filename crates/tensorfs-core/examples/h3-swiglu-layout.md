# H3 native fused MLP layout

`h3.native/2` converts the source `[gate; value]` rows of `mlp.fc1` into the
`[value; gate]` order consumed by Diffusers SwiGLU. This applies to both main
transformer blocks and token-refiner blocks in FL2VA and Ref2VA. The source
`fc2.weight` proves the gate width; odd rows, inconsistent dimensions and missing
companions refuse before payload reads. A supplied fc1 bias follows the same
row permutation, although the pinned real H3 source has no such bias.

The native converter uses `fill::Xform::SwapHalves`, which contributes two source
runs to the existing `RunRegion` streaming writer. It performs no floating-point
arithmetic and does not buffer a whole weight. Native-source encoded companion
roles remain refused until their source transformation has its own reviewed
mapping. Existing cozytensors checkpoint repair is a separate derived job.

The incorrect `h3.native/1` recipe is removed. Reviewed source-profile names are
unchanged because they identify the source selection; their native fingerprint
entries now require `/2`. Source preparation re-derives the tensor plan before
replaying a completed result. Both the recipe and the actual transform geometry
participate in the existing journal digest, so a pre-fix journal/result cannot
silently satisfy the corrected work. Reusing the same operation for changed work
refuses; use a fresh operation or the explicit repair job for existing checkpoints.

Run `tfs ingest golden` and the `h3_swiglu` carrier-to-transaction regressions.
The existing `resident_conversion_profile` example can convert a bounded actual
carrier and then resume it with zero payload rereads. The 2026-09-07 trained proof
used Ref2VA block0's complete BF16 fc1/fc2 pair:462,422,016 source bytes. Native
conversion took16.96s in a debug build with7.53MB peakRSS; resume took16.1ms,
converted zero roles, and hashed/wrote zero payload bytes. The full fc1 result
matches the independent source-range oracle
`e7cf97ae56a095d37b98a17fc4da01df7aaee6ef022fa045d02530038cae3b19`;
fc2 remains unchanged. This is source-conversion proof, not a new quantization or
an end-to-end generation benchmark.
