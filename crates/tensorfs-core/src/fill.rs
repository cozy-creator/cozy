//! The ONE fill-transform implementation (tensorfs.md §4) — the code that applies a
//! permutation-rule encoding during the host staging pass.
//!
//! It lives here, once, and has exactly two invocation sites: the ingest border's staging
//! pass (`ingest::convert`) and the serve-side fill path (`read`). Two producers of layout
//! semantics already disagreed in v1; the fence keeps the variants and `apply` defined in
//! this file alone.
//!
//! **The decomposition is arithmetic, never a walk.** `runs()` computes the contiguous
//! source→destination runs from the declared geometry in O(groups), and `apply` is DERIVED
//! from `runs` rather than written beside it — so the copy can no longer disagree with the
//! plan (v1's 250x lesson: 765 ms → 3.03 ms per 16 MiB once the decomposition stopped being
//! discovered per element).
//!
//! **Gather-class work is planned, never assumed free.** A decomposition whose mean run is
//! short is labelled `Gather` and says so: per-element gathers ran at 0.12 GiB/s in v1, so a
//! caller that sees `Gather` is being told it bought real work, not a free ride on the
//! staging copy.

use crate::err::{refuse, Code, Result};

/// Below this mean run length a decomposition is gather-class: the copy stops riding the
/// staging pass at ~zero marginal cost and becomes real work with its own budget.
pub const GATHER_FLOOR_BYTES: u64 = 4096;

/// One contiguous copy. Both offsets are byte offsets into whole logical runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub src_off: u64,
    pub dest_off: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Long runs — rides the staging copy at ~zero marginal cost.
    LongRun,
    /// Short runs — real work, budgeted and reported, never claimed free.
    Gather,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::LongRun => "long-run",
            Class::Gather => "gather",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Decomposition {
    pub runs: Vec<Run>,
    pub class: Class,
    pub bytes: u64,
}

impl Decomposition {
    fn of(runs: Vec<Run>) -> Decomposition {
        let bytes: u64 = runs.iter().map(|r| r.len).sum();
        let mean = if runs.is_empty() {
            0
        } else {
            bytes / runs.len() as u64
        };
        Decomposition {
            class: if mean >= GATHER_FLOOR_BYTES {
                Class::LongRun
            } else {
                Class::Gather
            },
            runs,
            bytes,
        }
    }
    pub fn mean_run(&self) -> u64 {
        if self.runs.is_empty() {
            0
        } else {
            self.bytes / self.runs.len() as u64
        }
    }
}

/// The ONE fused-projection transform, in the vocabulary the v1 quarry ratified for its
/// seam registry: `groups` x `shares`, cut on whole units.
///
/// **The group count is the load-bearing fact.** A fused QKV is not three stacked blocks —
/// upstream reads it as `qkv.view(total, heads, 3, head_dim)`, so q, k and v are adjacent
/// INSIDE each head. The obvious flat three-way split is right for head 0 and wrong for
/// every other head, and it does not crash: the v1 quarry pinned it numerically at
/// **max|d| 6.09e+01 against |ref|max 6.78e+01 — roughly 90% error, with every name, dtype
/// and shape correct** (jobs/conversion/.../h3_native_layout.py:106). An adapter trained on
/// the wrong fusion converges beautifully and serves garbage.
///
/// `groups = 1` is the contiguous case (OpenCLIP `in_proj_weight`, the video-VAE 1x1-conv
/// `to_qkv`), so one transform covers both and the difference is DATA, not a second code path.
/// `shares` is per-member so an unequal split is representable: the quarry's Qwen3.6 seam
/// is 1:1:2 (q 2048, k 2048, v 4096), which equal thirds does not even divide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Xform {
    /// Fused [gate; value] rows become [value; gate]. Both runs stream unchanged.
    SwapHalves { half_bytes: u64 },
    QkvSplit {
        groups: u64,
        shares: [u64; 3],
        /// 0 = q, 1 = k, 2 = v.
        take: u8,
        /// Bytes in one share unit — derived from the source run, never assumed.
        unit_bytes: u64,
    },
    /// A row-major `[rows, cols]` matrix becomes `[cols, rows]`. Every element moves on its
    /// own, so this is gather-class by construction: it is applied buffered, never as runs.
    Transpose {
        rows: u64,
        cols: u64,
        elem_bytes: u64,
    },
}

impl Xform {
    /// Total source bytes this transform consumes.
    pub fn in_bytes(&self) -> u64 {
        match self {
            Xform::SwapHalves { half_bytes } => 2 * half_bytes,
            Xform::QkvSplit {
                groups,
                shares,
                unit_bytes,
                ..
            } => groups * (shares[0] + shares[1] + shares[2]) * unit_bytes,
            Xform::Transpose {
                rows,
                cols,
                elem_bytes,
            } => rows * cols * elem_bytes,
        }
    }

    /// Output byte length, derived from the transform alone.
    pub fn out_bytes(&self) -> u64 {
        match self {
            Xform::SwapHalves { half_bytes } => 2 * half_bytes,
            Xform::QkvSplit {
                groups,
                shares,
                take,
                unit_bytes,
            } => groups * shares[*take as usize] * unit_bytes,
            Xform::Transpose { .. } => self.in_bytes(),
        }
    }

    /// Whether the transform streams its source runs straight out of the carrier. A
    /// transpose reads its bounded source once and reorders it in memory instead.
    pub fn streams(&self) -> bool {
        !matches!(self, Xform::Transpose { .. })
    }

    /// `(bytes, runs, class)` of the decomposition, without materializing element runs.
    pub fn census(&self) -> (u64, usize, Class) {
        match self {
            Xform::Transpose {
                rows,
                cols,
                elem_bytes,
            } => (
                rows * cols * elem_bytes,
                (rows * cols) as usize,
                if *elem_bytes >= GATHER_FLOOR_BYTES {
                    Class::LongRun
                } else {
                    Class::Gather
                },
            ),
            _ => {
                let d = self.runs();
                (d.bytes, d.runs.len(), d.class)
            }
        }
    }

    /// The decomposition — computed from the declared geometry in O(groups), never
    /// discovered by inspecting bytes. This is the single source of transform semantics:
    /// `apply` executes exactly these runs and nothing else.
    pub fn runs(&self) -> Decomposition {
        match self {
            Xform::SwapHalves { half_bytes } => Decomposition::of(vec![
                Run {
                    src_off: *half_bytes,
                    dest_off: 0,
                    len: *half_bytes,
                },
                Run {
                    src_off: 0,
                    dest_off: *half_bytes,
                    len: *half_bytes,
                },
            ]),
            Xform::QkvSplit {
                groups,
                shares,
                take,
                unit_bytes,
            } => {
                let i = *take as usize;
                let group = (shares[0] + shares[1] + shares[2]) * unit_bytes;
                let skip: u64 = shares[..i].iter().sum::<u64>() * unit_bytes;
                let len = shares[i] * unit_bytes;
                Decomposition::of(
                    (0..*groups)
                        .map(|g| Run {
                            src_off: g * group + skip,
                            dest_off: g * len,
                            len,
                        })
                        .collect(),
                )
            }
            Xform::Transpose {
                rows,
                cols,
                elem_bytes,
            } => Decomposition::of(
                (0..*cols)
                    .flat_map(|j| {
                        (0..*rows).map(move |i| Run {
                            src_off: (i * cols + j) * elem_bytes,
                            dest_off: (j * rows + i) * elem_bytes,
                            len: *elem_bytes,
                        })
                    })
                    .collect(),
            ),
        }
    }

    fn check_in(&self, n: usize) -> Result<()> {
        if n as u64 != self.in_bytes() {
            return refuse(
                Code::BYTE_LENGTH_MISMATCH,
                format!(
                    "{n} source bytes, the transform's geometry wants {}",
                    self.in_bytes()
                ),
            );
        }
        Ok(())
    }

    /// Apply into a CALLER-OWNED buffer — the fill path's form. Pure reordering of whole
    /// units; no element ever changes value, which is what makes this class safe to run at
    /// either border site.
    pub fn apply_into(&self, src: &[u8], dest: &mut [u8]) -> Result<Decomposition> {
        self.check_in(src.len())?;
        if dest.len() as u64 != self.out_bytes() {
            return refuse(
                Code::BYTE_LENGTH_MISMATCH,
                format!(
                    "destination is {} B, the transform emits {} B",
                    dest.len(),
                    self.out_bytes()
                ),
            );
        }
        if let Xform::Transpose {
            rows,
            cols,
            elem_bytes,
        } = *self
        {
            let (rows, cols, e) = (rows as usize, cols as usize, elem_bytes as usize);
            for (i, row) in src.chunks_exact(cols * e).enumerate() {
                for (j, value) in row.chunks_exact(e).enumerate() {
                    dest[(j * rows + i) * e..][..e].copy_from_slice(value);
                }
            }
            return Ok(Decomposition {
                runs: Vec::new(),
                class: Class::Gather,
                bytes: self.out_bytes(),
            });
        }
        let d = self.runs();
        for r in &d.runs {
            let (s, o, n) = (r.src_off as usize, r.dest_off as usize, r.len as usize);
            dest[o..o + n].copy_from_slice(&src[s..s + n]);
        }
        Ok(d)
    }

    /// The allocating form, for callers that do not own a destination yet (the ingest
    /// staging pass). Same runs, same semantics — it delegates.
    pub fn apply(&self, src: &[u8], out: &mut Vec<u8>) -> Result<()> {
        self.check_in(src.len())?;
        let base = out.len();
        out.resize(base + self.out_bytes() as usize, 0);
        self.apply_into(src, &mut out[base..])?;
        Ok(())
    }

    /// The inverse read-back proves that the transform loses or duplicates no bytes.
    /// It uses the transform's own indexing, so it cannot establish the producer's
    /// QKV layout: that needs an independent projection-output regression.
    pub fn reassembles(&self, src: &[u8]) -> Result<bool> {
        match self {
            Xform::Transpose {
                rows,
                cols,
                elem_bytes,
            } => {
                let mut once = Vec::new();
                self.apply(src, &mut once)?;
                let mut back = Vec::new();
                Xform::Transpose {
                    rows: *cols,
                    cols: *rows,
                    elem_bytes: *elem_bytes,
                }
                .apply(&once, &mut back)?;
                Ok(back == src
                    && self.runs().runs.iter().all(|r| {
                        once[r.dest_off as usize..][..r.len as usize]
                            == src[r.src_off as usize..][..r.len as usize]
                    }))
            }
            Xform::SwapHalves { .. } => {
                let mut swapped = Vec::new();
                self.apply(src, &mut swapped)?;
                let mut restored = Vec::new();
                self.apply(&swapped, &mut restored)?;
                Ok(restored == src)
            }
            Xform::QkvSplit {
                groups,
                shares,
                unit_bytes,
                ..
            } => {
                let mut members = Vec::new();
                for take in 0..3u8 {
                    let mut out = Vec::new();
                    Xform::QkvSplit {
                        groups: *groups,
                        shares: *shares,
                        take,
                        unit_bytes: *unit_bytes,
                    }
                    .apply(src, &mut out)?;
                    members.push(out);
                }
                let mut back = Vec::with_capacity(src.len());
                for g in 0..*groups as usize {
                    for (i, m) in members.iter().enumerate() {
                        let unit = (shares[i] * unit_bytes) as usize;
                        back.extend_from_slice(&m[g * unit..(g + 1) * unit]);
                    }
                }
                Ok(back == src)
            }
        }
    }
}

/// The boundary rule for permutation encodings (tensorfs.md §4): a permute over a
/// BLOCK-QUANTIZED carrier is fill-applicable only when it moves whole blocks. An
/// element-level permute would need dequant→requant, which is math — an AOT job's work,
/// never the fill path's.
///
/// `block_elements` is the encoding's scale-block size (0 = unquantized carrier, which this
/// rule does not govern); `unit_elements` is the transform's smallest moved unit.
pub fn guard_block_granular(what: &str, unit_elements: u64, block_elements: u64) -> Result<()> {
    if block_elements <= 1 {
        return Ok(());
    }
    if unit_elements == 0 || !unit_elements.is_multiple_of(block_elements) {
        return refuse(
            Code::PERMUTE_NOT_BLOCK_GRANULAR,
            format!(
                "{what}: a permute unit of {unit_elements} elements is not a whole multiple \
                 of the carrier's {block_elements}-element quantization block — an \
                 element-level permute needs dequant→requant, which is an AOT job, never fill"
            ),
        );
    }
    Ok(())
}
