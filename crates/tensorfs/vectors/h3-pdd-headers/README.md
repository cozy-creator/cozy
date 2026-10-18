These are only the safetensors length prefix and header bytes from
`alibaba-pai/MiniMax-H3-Acc-LoRAs@335001fb9e5455d68a0caa18ec2e319072150328`.
`MANIFEST.json` records the complete carrier sizes reported by the pinned HTTP
range responses. No model tensor payload is stored here.

Both adapters contain 728 BF16 tensors: rank64 LoRA factors and 32-head banks,
with alpha64 and PDD block size4. Each source profile selects exactly one carrier
and preserves its keys under one adapter component. The producer owns collapsing
heads, forming modulation tables and constructing the serving overlay.

No converter table claims these adapters: an import stores them as they are.
