"""Small admitted mapped-host feasibility; no large-output or performance claim."""

import gc
import json

import tensorfs
import torch
from tensorfs.varena.torch import Memory


assert torch.cuda.device_count() == 1
device = torch.device("cuda", 0)
with torch.cuda.device(device):
    torch.cuda.current_blas_handle()
memory = Memory.open(device)
assert memory.native is not None


def allocate(shape, dtype):
    array = memory.native.mapped_output(shape, dtype, torch.cuda.current_stream(device).cuda_stream)
    assert array is not None, "assigned device/admission must support this small mapped output"
    return array


class ObservedProducer:
    """Forward the real Torch capsule; observe whether a rejected write exported it."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.exports = 0

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        self.exports += 1
        return self.tensor.__dlpack__(**kwargs)


closed = False
try:
    with memory.request("mapped-host-feasibility") as request:
        for spelling, dtype in (("F16", torch.float16), ("BF16", torch.bfloat16),
                                ("F32", torch.float32), ("F64", torch.float64)):
            array = allocate((2, 3, 4), spelling)
            assert array.__dlpack_device__() == (2, 0)
            result = torch.from_dlpack(array)
            assert result.device == device and result.dtype == dtype
            try:
                array.__dlpack__(copy=True)
            except BufferError:
                pass
            else:
                raise AssertionError("mapped copy=True bypassed the framework allocator")
            # A noncontiguous source, and a band that is noncontiguous in the full output.
            source = torch.arange(48, dtype=dtype, device=device).reshape(2, 3, 8)[:, :, ::4]
            assert not source.is_contiguous()
            array._write_band(source, axis=2, offset=1, stream=torch.cuda.current_stream(device).cuda_stream)
            expected = torch.zeros((2, 3, 4), dtype=dtype)
            expected[:, :, 1:3] = source.cpu()
            assert torch.equal(result.detach().cpu(), expected)
            assert torch.equal((result + 1).cpu(), expected + 1)
            assert len(array.tobytes()) == 24 * result.element_size()
            del source, result, array

        array = allocate((1, 2, 8, 8), "F32")
        tensor = torch.from_dlpack(array)
        array._write_band(torch.ones_like(tensor), axis=2, offset=0, stream=torch.cuda.current_stream(device).cuda_stream)
        assert torch.backends.cudnn.is_available()
        weights = torch.ones((3, 2, 3, 3), device=device)
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                convolution = torch.nn.functional.conv2d(tensor, weights)
        assert any("cudnn_convolution" in event.key for event in profile.key_averages()), "cuDNN was not executed"
        assert torch.equal(convolution.cpu(), torch.full((1, 3, 6, 6), 18.0))
        # The ordinary consumer boundary must work on the CUDA-typed host-backed result.
        assert (tensor.detach().cpu().numpy() == 1).all()
        alias = tensor.view(-1)[8:32]
        del tensor, array
        gc.collect()
        side = torch.cuda.Stream(device=device)
        with torch.cuda.stream(side):
            torch.cuda._sleep(10_000_000)
            observed = (alias + 3).clone()
        del alias
        gc.collect()
        side.synchronize()
        assert torch.equal(observed.cpu(), torch.full((24,), 4.0))

        refused_array = allocate((4, 4), "F32")
        refused_tensor = torch.from_dlpack(refused_array)
        producer = ObservedProducer(torch.ones((4, 4), device=device))
        graph = torch.cuda.CUDAGraph()
        rejected = False
        with torch.cuda.graph(graph):
            declined = memory.native.mapped_output((4, 4), "F32", torch.cuda.current_stream(device).cuda_stream)
            try:
                refused_array._write_band(producer, axis=0, offset=0,
                                          stream=torch.cuda.current_stream(device).cuda_stream)
            except tensorfs.UnsupportedError:
                rejected = True
        assert declined is None, "mapped allocation must refuse the actual capturing stream"
        assert rejected and producer.exports == 0, "a refused write exported a CUDA producer"
        assert torch.equal(refused_tensor.cpu(), torch.zeros((4, 4)))
        refused_array._write_band(producer, axis=0, offset=0,
                                  stream=torch.cuda.current_stream(device).cuda_stream)
        assert producer.exports == 1 and torch.equal(refused_tensor.cpu(), torch.ones((4, 4)))
        del refused_tensor, refused_array, producer
        gc.collect()
        survivor_array = allocate((4, 4), "F32")
        survivor = torch.from_dlpack(survivor_array)
        survivor.fill_(9)
        del survivor_array, weights, convolution, observed, graph, profile
    memory.close()
    closed = True
    assert torch.equal(survivor.detach().cpu(), torch.full((4, 4), 9.0))
    del survivor
    gc.collect()
    torch.cuda.synchronize(device)
    assert request.stats is not None and not request.cleanup_errors
    print("PASS mapped host CUDA arithmetic, cuDNN, strided bands, D2H, capture refusal and aliases")
    print(json.dumps({"gpu": str(torch.cuda.get_device_properties(device).uuid), "largest_mapped_payload_bytes": 512, "leftover": request.stats.leftover}))
finally:
    if not closed:
        memory.close()
