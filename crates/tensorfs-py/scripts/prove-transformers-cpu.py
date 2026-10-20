#!/usr/bin/env python3
"""A6: actual tiny Transformers models and a path-loaded generic local tokenizer.

Pinned to the Runtime lock: torch 2.14.0, transformers 5.17.0, tokenizers 0.23.1.
No pretrained model/tokenizer is downloaded. This proves a generic WordLevel
tokenizer, not pretrained Llama/CLIP tokenizer algorithms or assets.

    python -I scripts/prove-transformers-cpu.py --wheel <final TensorFS wheel>

Upstream APIs:
https://huggingface.co/docs/transformers/v5.17.0/main_classes/tokenizer
https://huggingface.co/docs/transformers/v5.17.0/model_doc/llama
https://huggingface.co/docs/transformers/v5.17.0/model_doc/clip
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import weakref
import zipfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tokenizers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
import torch
import transformers
from transformers import (
    AutoTokenizer, CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig,
    LlamaConfig, LlamaForCausalLM, TokenizersBackend,
)

import tensorfs

NATIVE_SOURCE = "0800dbcfee93b887891624a303124de2c017a034"
WHEEL_SHA256 = "8ee675f083511563e1573e13b035706241b2fd9270327c4e16f03cc2b4b6a432"
TEXTS = ["A red cube", "a blue sphere", "a mystery"]
TOKEN_IDS = [[1, 3, 4, 5, 15], [1, 3, 6, 7, 15], [1, 3, 2, 15, 0]]
ATTENTION = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def make_checkpoint(root):
    """Return only inputs, independent oracles and weakrefs; retain no producer."""
    vocab = {"<pad>": 0, "<s>": 1, "<unk>": 2, "a": 3, "red": 4,
             "cube": 5, "blue": 6, "sphere": 7,
             **{f"unused{i}": i for i in range(8, 15)}, "</s>": 15}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 15)],
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>", model_max_length=16,
    )
    encoded = tokenizer(TEXTS, padding=True, return_tensors="pt")
    assert encoded.input_ids.tolist() == TOKEN_IDS
    assert encoded.attention_mask.tolist() == ATTENTION
    models_by_name = {
        "llama": LlamaForCausalLM(LlamaConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=16, tie_word_embeddings=False,
            pad_token_id=0, bos_token_id=1, eos_token_id=15, use_cache=False,
            attn_implementation="eager",
        )).eval(),
        "clip": CLIPModel(CLIPConfig(
            text_config=CLIPTextConfig(
                vocab_size=16, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2,
                max_position_embeddings=16, pad_token_id=0,
                bos_token_id=1, eos_token_id=15, attention_dropout=0.0,
            ),
            vision_config=CLIPVisionConfig(
                hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, image_size=8, patch_size=4,
                num_channels=3, attention_dropout=0.0,
            ),
            projection_dim=8, attn_implementation="eager",
        )).eval(),
    }
    for model in models_by_name.values():
        model.set_attn_implementation("eager")
    pixels = torch.randn(3, 3, 8, 8)
    with torch.inference_mode():
        llama_logits = models_by_name["llama"](
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
        ).logits.clone()
        clip_output = models_by_name["clip"](
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
            pixel_values=pixels,
        )
        expected = {"llama": llama_logits,
                    "clip_text": clip_output.logits_per_text.clone(),
                    "clip_image": clip_output.logits_per_image.clone()}
    source = root / "tokenizer-source"
    tokenizer.save_pretrained(source)
    tokenizer_files = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    tokenizer_config = json.loads(tokenizer_files["tokenizer_config.json"])
    refs = [weakref.ref(tokenizer), *(weakref.ref(m) for m in models_by_name.values()),
            *(weakref.ref(p) for m in models_by_name.values() for p in m.parameters())]
    manifests = {}
    writer = tensorfs.Writer(root / "checkpoint", cache_dir=root / "cache")
    for name, model in models_by_name.items():
        component = writer.component(
            name, cls=f"transformers.{type(model).__name__}",
            config=json.loads(model.config.to_json_string(use_diff=False)),
        )
        manifests[name] = {}
        for key, value in model.state_dict().items():
            assert value.dtype == torch.float32 and value.device.type == "cpu"
            component.put(key, value.contiguous())
            manifests[name][key] = digest(value)
    component = writer.component(
        "tokenizer", cls="transformers.TokenizersBackend", config=tokenizer_config,
    )
    for name, data in tokenizer_files.items():
        component.asset(name, data)
    checkpoint = writer.commit()
    header = checkpoint.id
    checkpoint.close()
    shutil.rmtree(source)
    return expected, pixels, manifests, tokenizer_files, refs, header


def prove(root):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261019)
    expected, pixels, manifests, assets, refs, header = make_checkpoint(root)
    gc.collect()
    assert all(ref() is None for ref in refs), "producer/tokenizer retained"
    assert not (root / "tokenizer-source").exists()
    checkpoint = tensorfs.open(root / "checkpoint", cache_dir=root / "cache")
    assert checkpoint.id == header
    assert set(checkpoint.components) == {"llama", "clip", "tokenizer"}
    constructors = {"llama": (LlamaConfig, LlamaForCausalLM),
                    "clip": (CLIPConfig, CLIPModel)}
    restored = {}
    tensor_count = nbytes = 0
    for name, component in checkpoint.components.items():
        if name == "tokenizer":
            assert component.cls == "transformers.TokenizersBackend"
            assert len(component.tensors) == 0
            path = Path(component.export_files(root / "compatibility-tokenizer"))
            assert path != root / "tokenizer-source"
            assert json.loads((path / "tokenizer_config.json").read_text()) == component.config
            assert set(component.assets) == set(assets)
            for asset, data in assets.items():
                assert component.asset(asset) == (path / asset).read_bytes() == data
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            continue
        config_cls, model_cls = constructors[name]
        assert component.cls == f"transformers.{model_cls.__name__}"
        arrays = component.tensors.read(device="cpu", decode=True)
        assert set(arrays) == set(manifests[name])
        loaded = {key: torch.from_dlpack(array) for key, array in arrays.items()}
        for key, value in loaded.items():
            assert digest(value) == manifests[name][key], f"{name}/{key} changed"
            tensor_count += 1
            nbytes += value.numel() * value.element_size()
        # Normal construction initializes nonpersistent rotary/position buffers
        # from config; the strict assign below replaces every persistent tensor.
        model = model_cls(config_cls.from_dict(component.config))
        model.set_attn_implementation("eager")
        model.load_state_dict(loaded, strict=True, assign=True)
        model.eval()
        for key, value in model.state_dict().items():
            assert value.data_ptr() == loaded[key].data_ptr(), f"copied {name}/{key}"
        restored[name] = model
        del arrays, loaded, model, value
    checkpoint.close()
    del checkpoint, component
    gc.collect()
    assert all(ref() is None for ref in refs)
    # The tokenizer was reconstructed with explicit export_files, and now both it and the models
    # must run after every TensorFS handle/Array and original producer is gone.
    encoded = tokenizer(TEXTS, padding=True, return_tensors="pt")
    assert encoded.input_ids.tolist() == TOKEN_IDS
    assert encoded.attention_mask.tolist() == ATTENTION
    with torch.inference_mode():
        actual_llama = restored["llama"](
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
        ).logits
        actual_clip = restored["clip"](
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
            pixel_values=pixels,
        )
    actual = {"llama": actual_llama, "clip_text": actual_clip.logits_per_text,
              "clip_image": actual_clip.logits_per_image}
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
    assert torch.equal(actual_llama.argmax(-1), expected["llama"].argmax(-1))
    return {"proof": "public-tensorfs-transformers-cpu", "header": header,
            "models": ["LlamaForCausalLM", "CLIPModel"],
            "tokenizer": "generic locally-created WordLevel; not pretrained family-specific",
            "tokenizer_class": type(tokenizer).__name__, "token_ids": TOKEN_IDS,
            "tokenizer_assets": sorted(assets), "tensors": tensor_count, "bytes": nbytes,
            "outputs": {name: {"shape": list(value.shape), "sha256": digest(value),
                               "max_abs_error": (value - expected[name]).abs().max().item()}
                        for name, value in actual.items()},
            "producer_released_before_reload": True, "original_tokenizer_dir_removed": True,
            "tokenizer_loaded_only_from_component_path": True,
            "forward_after_all_tensorfs_handles_released": True,
            "assigned_without_copy": True, "reference_computed_before_write": True,
            "tolerance": {"rtol": 0, "atol": 0}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args()
    assert torch.__version__.split("+")[0] == "2.14.0", torch.__version__
    assert transformers.__version__ == "5.17.0", transformers.__version__
    assert tokenizers.__version__ == "0.23.1", tokenizers.__version__
    wheel_hash = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    assert wheel_hash == WHEEL_SHA256, (wheel_hash, WHEEL_SHA256)
    package = Path(tensorfs.__file__).resolve().parent
    installed = {}
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith("tensorfs/") and not name.endswith("/"):
                data = (package / Path(name).relative_to("tensorfs")).read_bytes()
                assert data == archive.read(name), name
                installed[name] = hashlib.sha256(data).hexdigest()
    with TemporaryDirectory(prefix="tensorfs-transformers-") as temp:
        receipt = prove(Path(temp))
    receipt.update(native_source=NATIVE_SOURCE, wheel_sha256=wheel_hash,
                   tensorfs=tensorfs.__version__, torch=torch.__version__,
                   transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                   installed_tensorfs_package=str(package), installed_file_sha256=installed,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   boundary="actual CPU F32; no pretrained download, GPU or Runtime proof")
    print(json.dumps(receipt, sort_keys=True))
