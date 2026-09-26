# TensorFS runtime-only closure disagrees with retained repository GC

At source 4146fe9, a Hub closure with scope=runtime correctly selects CozyTensors objects while excluding an ordinary README sibling from the same manifest. FetchPlan.complete_cozytensors proves that selected closure. Repository recording then revalidates the entire snapshot and refuses OBJECT_ABSENT for the sibling.

A native Store/local-HTTP reproduction demonstrates the refusal. Simply skipping the redundant cache validation is unsafe: the pull succeeds, but the next real gc::collect refuses OBJECT_ABSENT while projecting the retained repository, because Census::validate_checkpoint_inputs requires every ordinary sibling in its inventory.

Do not remove checkpoint integrity or mark missing bytes resident. Authored snapshot roots and public staged-publication validation still require the full closure. Explore whether an exact cache-owned repository can retain all declared object IDs while GC marks only the resident subset, preserving the exact manifest identity and all header/length/digest checks. Unknown or authored roots must retain strict validation, and missing header/tensor records must not silently gain completeness claims.

Falsifiable qualification: runtime-only pull with a real ordinary sibling succeeds; runtime reads succeed; GC remains safe and can collect unrelated bytes; a full-snapshot operation still refuses the missing sibling; a corrupt present sibling/header/tensor refuses; authored and unowned roots stay strict; cache ownership mismatch cannot authorize reduced validation. Prove actual Creator/Host/Runtime consumers and present component/snapshot contracts before choosing a representation.

Related audit/implementation: TensorFS PR198 and tracker #230. Evidence: outputs/tfs-cache-pull-audit-20260926/baseline-tests.log, runtime-sibling-gc.log, intermediate-runtime-scope-probe.patch. The PR198 safety test preserves the current pre-retention refusal; it does not claim the functional mismatch is solved.
