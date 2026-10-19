"""Checks `tensorfs/src/import/gguf_names.json` against llama.cpp's own forward map.

Every upstream name maps, through gguf-py's `TensorNameMap` (`tensor_mapping.py`), to the GGUF name
it is listed under. An expert row is checked in the merged name llama.cpp's converter stacks the
experts under (`conversion/llama.py`, `conversion/qwen.py`). Every arch's GGUF names are tensors
llama.cpp declares for it (`constants.py` `MODEL_TENSORS`).

  uv run --python 3.12 --with numpy \
    --with 'gguf @ git+https://github.com/ggml-org/llama.cpp@ff5a23376725e5a889fa0bfb987b7541dc304da0#subdirectory=gguf-py' \
    python -I scripts/gguf_names.py crates/tensorfs/src/import/gguf_names.json
"""

import json
import sys

import gguf

# The names the converters stack experts under, before mapping.
MERGED = {
    "llama": "layers.{bid}.feed_forward.experts.{w}",
    "qwen3moe": "model.layers.{bid}.mlp.experts.{w}",
}


def main(path):
    arches = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}
    failures = 0
    for arch, m in json.load(open(path)).items():
        enum = arches[arch]
        declared = {gguf.TENSOR_NAMES[t] for t in gguf.MODEL_TENSORS[enum]}
        names = gguf.get_tensor_name_map(enum, 4)
        for g, up in m["tensors"].items():
            if g not in declared:
                print(f"{arch}: {g} is not a tensor llama.cpp declares")
                failures += 1
            for bid in ("0", "3"):
                want = g.replace("{bid}", bid)
                hf = up.replace("{bid}", bid)
                if "{xid}" in hf:
                    hf = MERGED[arch].format(bid=bid, w=hf.rsplit(".", 1)[1])
                for suffix in (".weight", ".bias"):
                    got = names.get_name(hf + suffix, try_suffixes=(".weight", ".bias"))
                    if got != want + suffix:
                        print(f"{arch}: {hf}{suffix} -> {got}, listed under {want}")
                        failures += 1
        for g in list(m.get("permute", {})) + m.get("llama_cpp_only", []):
            if g not in declared:
                print(f"{arch}: {g} is not a tensor llama.cpp declares")
                failures += 1
    print(f"{failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(sys.argv[1])
