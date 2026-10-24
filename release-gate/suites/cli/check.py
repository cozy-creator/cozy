"""Suite `cli` (./cells.sh): each GPU-only `run` flag did what it says, in its run's record.
    check(gate) -> (fails, notes)"""
import json
import os

GIB = 1 << 30


def ints(o, key):
    """Every integer value under `key`, anywhere in the record."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key and isinstance(v, int) and not isinstance(v, bool):
                yield v
            yield from ints(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from ints(v, key)


def check(gate):
    fails, notes, recs, summary = [], [], gate['records'], gate['summary']
    if not any(n.startswith('cli-') for n in summary) and not os.path.exists(os.path.join(gate['out'], 'cli-fails.txt')):
        return fails, notes
    if 'cli-gpus' in recs:
        gpus = set(ints(recs['cli-gpus'], 'gpu')) - {-1}
        (fails if len(gpus) > 1 else notes).append(f'cli-gpus: --gpus 1 ran on GPU(s) {sorted(gpus) or [0]}')
    if 'cli-lane16' in recs:
        precisions = {m.get('precision') for m in (recs['cli-lane16'].get('result') or {}).get('models') or []}
        int8 = [g for g in summary['cli-lane16']['gemm'] if g[0].startswith(('int8', 'fp8', 'nvfp4'))]
        if not precisions & {'fp16', 'bf16'} or int8:
            fails.append(f'cli-lane16: --lane 16 served {sorted(p for p in precisions if p)} and GEMM {int8}')
    if 'cli-attn' in summary:
        attn = summary['cli-attn']['attention']
        (notes if attn == ['sdpa'] else fails).append(f'cli-attn: --attention-kernel sdpa served {attn}')
    if 'cli-lora' in recs:
        text = json.dumps(recs['cli-lora']).lower()
        if 'lcm-lora-sdxl' not in text and 'adapter' not in text:
            fails.append('cli-lora: the record names no LoRA (lcm-lora-sdxl) applied')
    if 'cli-cap' in summary:
        peak = summary['cli-cap'].get('peak_vram_mb')
        if peak and peak * (1 << 20) > 12 * GIB:
            fails.append(f'cli-cap: peak {peak} MiB over the 12 GiB --gpu-memory-limit')
    p = os.path.join(gate['out'], 'cli-fails.txt')
    if os.path.exists(p):
        fails += [f'cli: {l.strip()}' for l in open(p) if l.strip()]
    return fails, notes
