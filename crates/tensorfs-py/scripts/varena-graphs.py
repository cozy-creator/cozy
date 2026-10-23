#!/usr/bin/env python3
"""Real-CUDA proof of cached graph addresses across physical weight eviction.

Runs through varena-rental.py. A seeded linear network is larger than the imposed VRAM cap;
its segments each use an explicit complete weight binding and capture into the device's one
Torch graph pool, mapped before each capture (`Memory.prepare_pool`). The same CUDAGraph
objects replay after backing eviction/refill. No compile, tuning,
or kernel selection occurs here. Every assertion/error propagates to the parent Cozy run.
"""
from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path

import tensorfs
import torch
from torch import nn
from tensorfs.varena import Fill, Level, VarenaError
from tensorfs.varena.torch import Declared, Memory, Weights

WIDTH, DEPTH = 4096, 20
CAP = 1 << 30


class Network(nn.Module):
    _no_split_modules = ["Linear"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(DEPTH))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    mem = Memory.open(0, limit=CAP)
    print(json.dumps({"torch": torch.__version__, "tensorfs": tensorfs.__version__, "gpu": torch.cuda.get_device_name(), "cap": CAP, "native_sha256": hashlib.sha256(Path(tensorfs._ext.__file__).read_bytes()).hexdigest()}), flush=True)
    torch.manual_seed(19)
    plain = Network().eval()
    source = torch.randn(2, WIDTH)
    with torch.inference_mode():
        for parameter in plain.parameters():
            parameter.mul_(3 ** 0.5)
        expected = plain(source)
    writer = tensorfs.Writer(cache_dir=Path(tempfile.mkdtemp(prefix="varena-graphs-")) / "store")
    component = writer.component("network", cls=("torch", torch.__version__.split("+")[0], "Module"), config={"width": WIDTH, "depth": DEPTH})
    for key, value in plain.state_dict().items():
        component.put(key, value.contiguous().view(torch.uint8).numpy().tobytes(), dtype="F32", shape=list(value.shape))
    checkpoint = writer.commit()
    del plain, writer, component, value
    gc.collect()
    with torch.device("meta"):
        model = Network().eval()
    vars(model)["_varena"] = Declared(Weights(checkpoint, "network"))
    placed = mem.load(model, "ram")
    bindings = [placed.retain([f"layers.{i}"]) for i in range(DEPTH)]
    assert len({b.identity for b in bindings}) == DEPTH, "range identities include their dependency subsets"
    weight_bytes = WIDTH * WIDTH * 4 * DEPTH
    assert weight_bytes > CAP
    measurable = mem.native.context_measurable()
    print(json.dumps({"context_measurable": measurable}), flush=True)
    assert measurable, "a capped graph proof requires measured per-process driver/context bytes"
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()  # the device's one capture pool
    records = []
    capture_ms = 0.0
    with torch.inference_mode(), mem.request("graph-capture", plan=[model] * DEPTH):
        for i, (layer, binding) in enumerate(zip(model.layers, bindings, strict=True)):
            graph = torch.cuda.CUDAGraph()
            with binding.prepare(stream), torch.cuda.stream(stream):
                static = source.to(mem.device)
                # Warm on the capture's stream (its cuBLAS workspace), in the request's arena.
                warm = torch.nn.functional.linear(static, layer.weight)
                torch.cuda.synchronize()
                mem.prepare_pool(pool, warm.nbytes)
                del warm
                before = mem.torch_stats()["mapped"]
                start = time.perf_counter()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    output = torch.nn.functional.linear(static, layer.weight)
                capture_ms += (time.perf_counter() - start) * 1000
                assert mem.torch_stats()["mapped"] == before, "a capture maps nothing: its pool was prepared"
                graph.replay()
                wanted = torch.nn.functional.linear(static, layer.weight)
                torch.testing.assert_close(output, wanted)
                address = layer.weight.data_ptr()
            records.append((binding, graph, static, output, address))
            torch.cuda.synchronize()
            mem.native.load(placed.set, Level.RAM)  # release backing; graph+VA stay live
            print(f"CAPTURE {i + 1}/{DEPTH} ptr={address:#x}", flush=True)

    def run(label, value, force_evict):
        moved = None
        start = time.perf_counter()
        with torch.inference_mode(), mem.request(label, plan=[model]) as request, placed.passing():
            x = value.to(mem.device)
            for i, (binding, graph, static, output, address) in enumerate(records):
                with binding.prepare(stream):
                    assert model.layers[i].weight.data_ptr() == address
                    static.copy_(x)
                    graph.replay()  # the exact instance captured before backing eviction
                    x = output.clone()
                if force_evict:
                    torch.cuda.synchronize()
                    mem.native.load(placed.set, Level.RAM)
            actual = x.cpu()
            moved = request.stats().moved
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        print(json.dumps({"run": label, "ms": (time.perf_counter() - start) * 1000, "store_bytes": moved.store.bytes}), flush=True)
        return actual

    run("graph-refill-1", source, True)
    run("graph-streaming-automatic", source, False)

    # One graph instance's lock covers its input copies and output materialization too.
    binding, graph, static, output, address = records[0]
    with binding.prepare(stream):
        reference = torch.nn.functional.linear(source.to(mem.device), model.layers[0].weight).cpu()
    def caller(scale):
        own = torch.cuda.Stream()
        with torch.inference_mode(), mem.request(f"concurrent-{scale}", plan=[model]):
            for _ in range(4):
                with binding.prepare(own):
                    static.copy_(source.to(mem.device) * scale)
                    graph.replay()
                    actual = output.cpu()
                torch.testing.assert_close(actual, reference * scale, atol=1e-5, rtol=1e-4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(caller, [1.0, 2.0]))
    print("PASS two concurrent callers serialize input, replay and output use", flush=True)

    # A capture short of its prepared pool is refused, making no room, and CUDA stays usable.
    before = mem.torch_stats()["mapped"]
    failed = False
    with mem.request("capture-growth-refused"):
        try:
            bad = torch.cuda.CUDAGraph()
            with torch.cuda.graph(bad, pool=torch.cuda.graph_pool_handle(), stream=stream):
                torch.empty(8 << 20, dtype=torch.uint8, device=mem.device)
        except (torch.OutOfMemoryError, RuntimeError):
            failed = True
    assert failed and mem.torch_stats()["mapped"] == before, "a capture's shortfall is refused, never made room for"
    torch.cuda.synchronize()
    assert torch.ones(1, device=mem.device).item() == 1
    print("PASS capture growth refused without recovery; CUDA remains usable", flush=True)

    mem.native.fill(placed.id, Fill.Declared("changed-content"))
    try:
        binding.native.prepare(stream.cuda_stream)
        raise AssertionError("stale graph binding entered")
    except VarenaError as error:
        assert error.code == "STALE_BINDING", f"stale identity raised the wrong refusal: {error}"
    for binding in bindings:
        binding.close()
    print(json.dumps({"PASS": True, "graphs": len(records), "capture_ms": capture_ms, "weights": weight_bytes, "cap": CAP, "torch": mem.torch_stats()}), flush=True)


if __name__ == "__main__":
    main()
