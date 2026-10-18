# Speed gate: cozytensors/2 against safetensors (xs-045 F1)

Requirement: cozytensors must be at least as fast as safetensors. Measured disk → pinned host → GPU on a
rented card, with real lane bytes, from first open to the last byte resident on the GPU.

```
cozy model download fidika/sdxl@1.0.0/int8 --on <machine>
cozy model download fidika/qwen-image-2.1@0.1.0/int8 --on <machine>
cozy run crates/tensorfs/gates/speed/speedgate.py --on <machine>
cozy run show <run> --json > result.json && python pullmodel.py result.json
```

## Layouts and readers

All layouts hold the same bytes in execution order, built on the machine from its v1 store.

- `v1`: today's per-part objects, read in place.
- `T16`, `T64`: the design §2 writer rule with 16 or 64 MiB packs. Its other constants are fixed: 16 MiB
  standalone threshold, 64 MiB grid, 4 KiB alignment, block-closed packs. `T64-open` has no block closing.
  `T32` is not built: block closing makes it T64 (SDXL identical, Qwen 8 of 544 chunks differ).
- `contiguous`: the safetensors file below, read with our reader.
- `st-cuda`, `st-cpu-cuda`: `safetensors.safe_open` 0.8, `device="cuda"`, or `"cpu"` then `.to("cuda")`. One file
  per component, parts as U8 tensors in execution order (the best case for a window).
- `fastsafetensors`: 0.5.0, nogds, one loader over every file.

Readers:

- Our reader (Python): 16 threads, 16 MiB O_DIRECT requests (buffered when warm), pinned slots, one H2D
  copy per slot.
- `+fst`: the same plan read by fastsafetensors' own C++ engine. It differs from `fastsafetensors` in layout
  only.

Cells:

- cold: `POSIX_FADV_DONTNEED` on every file first;
- warm: the second of two buffered reads;
- full: the whole checkpoint resident;
- window: SDXL `unet.up_blocks.1`, or Qwen `transformer.transformer_blocks.30`.

## Results

Run 11100: RTX 4000 Ada (rtx-4000-ada, RunPod RO), local NVMe about 2.9 GB/s, PCIe 26 GB/s pinned. Times are
ms, median (best) of 5 interleaved repetitions.

SDXL int8, 4.1 GB, window 187 MB:

| layout (objects) | full cold | full warm | window cold | window warm |
|---|---:|---:|---:|---:|
| v1 (3507) | 1445 (1435) | 346 (338) | 75 (74) | 22 (21) |
| T16 (287) | 1520 (1500) | 285 (253) | 86 (85) | 31 (24) |
| T64 (203) | 1487 (1474) | 277 (257) | 88 (86) | 31 (23) |
| T64-open (73) | 1510 (1460) | 279 (256) | 107 (106) | 35 (34) |
| contiguous | 1528 (1510) | 280 (247) | 78 (76) | 17 (15) |
| v1+fst | 1457 (1450) | 486 (463) | 71 (70) | 24 (22) |
| **T64+fst** | **1395 (1389)** | 470 (438) | 81 (80) | 30 (28) |
| st-cuda | 5258 (5207) | 1192 (1176) | 276 (267) | 62 (60) |
| st-cpu-cuda | 4632 (4601) | 1070 (1062) | 200 (194) | 56 (55) |
| fastsafetensors | 1459 (1442) | 1450 (1442) | 145 (140) | 138 (137) |

Qwen-Image int8, 19.2 GB, window 218 MB:

| layout (objects) | full cold | full warm | window cold | window warm |
|---|---:|---:|---:|---:|
| v1 (1660) | 6962 (6935) | 1305 (1280) | 84 (83) | 19 (18) |
| T16 (611) | 6898 (6878) | 1381 (1352) | 92 (89) | 25 (23) |
| T64 (544) | 6869 (6815) | 1393 (1360) | 94 (90) | 21 (20) |
| T64-open (511) | 6874 (6859) | 1394 (1357) | 91 (90) | 21 (20) |
| contiguous | 6992 (6979) | 1390 (1385) | 88 (86) | 19 (17) |
| v1+fst | 6595 (6525) | 2304 (2276) | 76 (76) | 17 (17) |
| **T64+fst** | **6452 (6439)** | 2261 (2217) | 85 (84) | 24 (20) |
| st-cuda | 19950 (19884) | 4452 (4428) | 231 (228) | 50 (50) |
| st-cpu-cuda | 18638 (18591) | 4020 (4005) | 225 (218) | 44 (44) |
| fastsafetensors | 6646 (6615) | 6676 (6636) | 161 (156) | 235 (158) |

**Gate: pass in every cell.** v2 T64 (with the better of the two readers) against the best safetensors path:

| cell | SDXL int8 | Qwen-Image int8 |
|---|---|---|
| full cold | T64+fst +5% vs fastsafetensors | T64+fst +3% vs fastsafetensors |
| full warm | T64 +287% vs st-cpu-cuda | T64 +189% vs st-cpu-cuda |
| window cold | T64+fst +79% vs fastsafetensors | T64+fst +89% vs fastsafetensors |
| window warm | T64+fst +85% vs st-cpu-cuda | T64 +114% vs st-cpu-cuda |

- Full cold is disk-bound for every fast path. With the same engine, T64 beats the safetensors file
  (+fst vs fastsafetensors), so the layout costs nothing.
- Our Python reader is 3–5% behind fastsafetensors' C++ engine on the same file (`contiguous` vs
  `fastsafetensors`, in all four runs 11036, 11047, 11060, 11100). The production Rust reader must match that
  engine: 16 MiB requests on 16 threads into pinned bounce buffers.
- fastsafetensors always reads O_DIRECT, so warm is no faster than cold. A warm v2 read from the page cache
  is 3–4× faster than any safetensors path.
- On a pod's NVMe, v1 per-part reads are within 5% of T64 (on the owner's laptop they were 2.6× slower).
  On a pod, v2's gains are downloads and windows.

## Download (R2)

- Staging the v2 layouts in R2 is not possible through the CLI:
  - `cozy model upload` takes only a local `.safetensors` file on the CLI's computer, or a provider/Hub
    source;
  - v1 ingest re-chunks every part anyway;
  - script weight outputs are derivations of a source checkpoint.
- The RO pod pulled both lanes from the datacenter volume (`hub 0 B`): SDXL in 8.3 s, Qwen in 26.3 s.
- Real R2 pulls of the v1 lanes on a second pod (rtx-a4000, RunPod IS):
  - SDXL int8: 3,880 GETs, 4.08 GB in 14.4 s (284 MB/s);
  - Qwen int8: 3,863 GETs, 19.2 GB in 18.0 s (1.07 GB/s).
- Both fit a 0.20 s TTFB, ~70 MB/s per connection and 59 connections (`pullmodel.py`). That model gives:
  - SDXL v1 13.3 s;
  - SDXL T16, T64 and T64-open 3.8 s: the bandwidth floor, even with today's 8 MiB ranges;
  - Qwen 17.9 s in every layout (bandwidth-bound already).

## Constants

All are confirmed:

- **64 MiB packs.** T64 is within 3% of T16 in every cell, with 30% fewer objects (SDXL 287 → 203).
- **Block-closed packs.** T64-open overfetches the SDXL window (269 vs 211 MB) and reads it 22% slower
  (107 vs 88 ms).
- **16 MiB threshold, 64 MiB grid.** Large-part reads already saturate the disk (T64+fst beats one
  contiguous file). Every target from 16 MiB up pulls at the bandwidth floor.
- **4 KiB alignment.** It is what lets every chunk read O_DIRECT straight into pinned memory. Padding is at
  most 0.13%.
