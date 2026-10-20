//! Checked format plans executed by the prebuilt catalog through the read engine's driver.

use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::sync::{Arc, Mutex, OnceLock};

use super::cuda::{self, CtxGuard};
use super::{Copy, DeviceBuffer, Dst, ReadOpts, Src};
use crate::failure::{Failure, Kind, Result};
use crate::format::decode::plan::{Core, Plan};
use crate::format::{decode, Dtype, Encoding, PartBytes};
use crate::kernels::{self, Binary, Factors, Launch, Prepared};

/// Declared format metadata and an existing contiguous byte span. No store is consulted.
/// `src` supports Host or Device; packed carriers retain their format dtype/shape separately.
pub struct Part<'a> {
    pub role: &'a str,
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub src: Src<'a>,
    pub len: usize,
}

/// Actual output metadata must match the explicit logical descriptor exactly.
pub struct Output<'a> {
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub dst: Dst,
}

/// Validate declared roles, dtypes, shapes, lengths and params without reading data or
/// initializing CUDA. The ordinary format registry remains the only scheme schema.
pub fn check(
    encoding: &Encoding,
    dtype: &Dtype,
    shape: &[u64],
    parts: &[decode::PartMetadata<'_>],
) -> Result<()> {
    Ok(decode::check_metadata(dtype, shape, encoding, parts)?)
}

fn invalid<T>(detail: impl Into<String>) -> Result<T> {
    crate::failure::fail(Kind::Invalid, detail)
}

fn span(ptr: u64, len: usize) -> Result<std::ops::Range<u64>> {
    if len > isize::MAX as usize || (ptr == 0 && len != 0) {
        return invalid("decode buffer has an invalid address or length");
    }
    let end = ptr.checked_add(len as u64).ok_or_else(|| Failure {
        kind: Kind::Invalid,
        detail: "decode buffer address overflow".into(),
    })?;
    Ok(ptr..end)
}

fn alignment(dtype: &Dtype) -> usize {
    match dtype {
        Dtype::F32 | Dtype::I32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        _ => 1,
    }
}

/// Decode existing parts into an existing destination. Host parts/output use the reference
/// CPU decoder; all-Device U7 parts/output use the SAME typed plan and prebuilt kernels as
/// checkpoint reads. Mixed placement, files and implicit transfers are unsupported.
///
/// Metadata, placement, address, alignment and overlap refusals precede output writes. After
/// a runtime/value failure output contents are unspecified. Success proves completion.
///
/// # Safety
/// Every span must be valid and retained, output exclusively writable, for this call. Device
/// streams must belong to that device's primary context and be valid on the calling thread
/// (including PTDS). On error callers must retain ALL parts/output owners until io::drain for
/// that device AND their producer streams report checked completion; failure is not proof of
/// quiescence. This includes owners shared through DLPack and views of larger allocations.
pub unsafe fn into(
    encoding: &Encoding,
    dtype: &Dtype,
    shape: &[u64],
    parts: &[Part<'_>],
    out: Output<'_>,
) -> Result<()> {
    let metadata: Vec<_> = parts
        .iter()
        .map(|p| decode::PartMetadata {
            role: p.role,
            dtype: p.dtype,
            shape: p.shape,
            len: p.len,
        })
        .collect();
    check(encoding, dtype, shape, &metadata)?;
    if out.dtype != dtype || out.shape != shape {
        return invalid("decode output dtype/shape differs from the logical descriptor");
    }
    if dtype.byte_len(shape) != Some(out.dst.len() as u64) {
        return invalid("decode output byte length differs from the logical descriptor");
    }
    let (output, device) = match out.dst {
        Dst::Host { ptr, len } => (span(ptr as u64, len)?, None),
        Dst::Device {
            device, ptr, len, ..
        } => (span(ptr, len)?, Some(device)),
    };
    if device.is_some() && !(output.start as usize).is_multiple_of(alignment(dtype)) {
        return invalid("decode output address is not aligned for its dtype");
    }
    if device.is_some() && !decode::supports_sources(encoding) {
        return crate::failure::fail(
            Kind::Unsupported,
            "this scheme has no resident device-input plan",
        );
    }
    for part in parts {
        let (pointer, placement) = match part.src {
            Src::Host(ptr) => (ptr as u64, None),
            Src::Device { ptr, device, .. } => (ptr, Some(device)),
            _ => {
                return crate::failure::fail(
                    Kind::Unsupported,
                    "decode inputs must be Host or Device byte spans",
                );
            }
        };
        if placement != device {
            return crate::failure::fail(
                Kind::Unsupported,
                "resident decode requires all inputs and output on the same device or all on Host",
            );
        }
        let input = span(pointer, part.len)?;
        if device.is_some() && !(pointer as usize).is_multiple_of(alignment(part.dtype)) {
            return invalid(format!(
                "decode part {:?} address is not aligned for its format dtype",
                part.role
            ));
        }
        if !input.is_empty()
            && !output.is_empty()
            && input.start < output.end
            && output.start < input.end
        {
            return invalid("decode output overlaps an input part");
        }
    }
    match device {
        None => {
            let host: BTreeMap<_, _> = parts
                .iter()
                .map(|p| {
                    let Src::Host(ptr) = p.src else {
                        unreachable!("placement checked")
                    };
                    // SAFETY: caller owns each checked Host span. Empty null spans are permitted.
                    let bytes = if p.len == 0 {
                        &[]
                    } else {
                        unsafe { std::slice::from_raw_parts(ptr, p.len) }
                    };
                    (
                        p.role,
                        PartBytes {
                            dtype: p.dtype,
                            shape: p.shape,
                            bytes,
                        },
                    )
                })
                .collect();
            let decoded = decode::decode(dtype, shape, encoding, &host)?;
            let Dst::Host { ptr, .. } = out.dst else {
                unreachable!("placement checked")
            };
            if !decoded.is_empty() {
                unsafe { std::ptr::copy_nonoverlapping(decoded.as_ptr(), ptr, decoded.len()) };
            }
            Ok(())
        }
        Some(device) => {
            decode::planned_sources(dtype, shape, encoding, &metadata, |plan, geometry| {
                let count = geometry
                    .rows
                    .checked_mul(geometry.cols)
                    .ok_or_else(|| crate::format::Error::Decode("decode size overflow".into()))?;
                Ok(
                    device_plan(plan, geometry, dtype, 0, count as u64).and_then(|launch| {
                        execute_into(device, &launch, parts, Some(out.dst)).map(|_| ())
                    }),
                )
            })?
        }
    }
}

/// Decode a logical element window through the prebuilt CUDA catalog. Host-only plans are
/// unsupported; logical CPU expansion and upload are never a GPU decode fallback.
pub fn read(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &BTreeMap<&str, PartBytes>,
    device: i32,
    begin: u64,
    count: u64,
) -> Result<DeviceBuffer> {
    decode::planned(dtype, shape, encoding, parts, true, |plan, geometry| {
        Ok(device_plan(plan, geometry, dtype, begin, count)
            .and_then(|launch| execute(device, &launch)))
    })?
}

fn device_plan<'a>(
    plan: Plan<'a>,
    geometry: decode::Geometry,
    dtype: &Dtype,
    begin: u64,
    count: u64,
) -> Result<Box<Launch<'a>>> {
    // Refuse before prepare can apply host factors, transpose or round a Values payload.
    if matches!(&plan.core, Core::Values(_)) {
        return crate::failure::fail(
            Kind::Unsupported,
            "GPU decode requires a device core; host value plans are unsupported",
        );
    }
    match kernels::prepare(plan, geometry, dtype, begin, count)? {
        Prepared::Kernel(launch) => Ok(launch),
        Prepared::Host(_) => crate::failure::fail(
            Kind::Unsupported,
            "GPU decode has no prebuilt device core for this plan",
        ),
    }
}

fn io_error(error: super::Error) -> Failure {
    error.source.into()
}

struct UploadHold(Option<DeviceBuffer>);
impl Drop for UploadHold {
    fn drop(&mut self) {
        if let Some(buffer) = self.0.take() {
            super::device::retain_decode(None, vec![buffer]);
        }
    }
}

fn upload(device: i32, bytes: &[u8]) -> Result<DeviceBuffer> {
    let mut hold = UploadHold(Some(DeviceBuffer::new(device, bytes.len())?));
    write(hold.0.as_ref().expect("upload buffer live"), bytes, 0)?;
    Ok(hold.0.take().expect("completed upload buffer"))
}

struct DecodeHold {
    output: Option<DeviceBuffer>,
    inputs: Vec<DeviceBuffer>,
    scratch: Vec<DeviceBuffer>,
    stream: Option<cuda::Stream>,
    complete: bool,
}
impl Drop for DecodeHold {
    fn drop(&mut self) {
        if !self.complete {
            let mut buffers = std::mem::take(&mut self.inputs);
            buffers.append(&mut self.scratch);
            buffers.extend(self.output.take());
            // An external output has no owned allocation until its first argument upload.
            // No private work can precede that registration: casts register their output
            // scratch before launch, and every decode kernel needs the owned slots buffer.
            // Early catalog/context/producer errors therefore have nothing to quarantine.
            // The caller still retains external owners when a producer drain failed.
            if !buffers.is_empty() {
                super::device::retain_decode(self.stream.take(), buffers);
            }
        }
    }
}

fn write(buffer: &DeviceBuffer, bytes: &[u8], stream: usize) -> Result<()> {
    let copy = Copy {
        src: Src::Host(bytes.as_ptr()),
        dst: Dst::Device {
            device: buffer.device(),
            ptr: buffer.ptr(),
            len: bytes.len(),
            stream,
        },
    };
    // SAFETY: source and exclusive destination remain live until the engine has drained.
    unsafe { super::read_staged(&[copy], &ReadOpts::default()) }.map_err(io_error)?;
    Ok(())
}

fn bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: used only for initialized fixed-width integer/float arrays in the kernel ABI.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

struct Module {
    context: usize,
    raw: usize,
    function: usize,
}

impl Drop for Module {
    fn drop(&mut self) {
        if let (Ok(driver), Ok(_context)) = (cuda::driver(), CtxGuard::enter(self.context as _)) {
            unsafe { (driver.module_unload)(self.raw as _) };
        }
    }
}

fn module(context: usize, binary: &'static Binary) -> Result<Arc<Module>> {
    type Key = (usize, u32, &'static str, &'static str);
    static MODULES: OnceLock<Mutex<BTreeMap<Key, Arc<Module>>>> = OnceLock::new();
    let mut modules = MODULES
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let key = (context, binary.family, binary.kind, binary.dtype);
    if let Some(module) = modules.get(&key) {
        return Ok(module.clone());
    }
    let driver = cuda::driver()?;
    let _context = CtxGuard::enter(context as _)?;
    let mut raw = std::ptr::null_mut();
    cuda::check("cuModuleLoadData", unsafe {
        (driver.module_load_data)(&mut raw, binary.image.as_ptr().cast())
    })?;
    let mut module = Module {
        context,
        raw: raw as usize,
        function: 0,
    };
    let name = CString::new(binary.symbol).map_err(|_| Failure {
        kind: Kind::Invalid,
        detail: "kernel symbol contains NUL".into(),
    })?;
    let mut function = std::ptr::null_mut();
    cuda::check("cuModuleGetFunction", unsafe {
        (driver.module_get_function)(&mut function, raw, name.as_ptr())
    })?;
    module.function = function as usize;
    let module = Arc::new(module);
    modules.insert(key, module.clone());
    Ok(module)
}

/// Triton3.8 appends global/profile scratch parameters to the eleven source arguments.
/// AOT refuses nonzero scratch requirements, but CUDA still reads BOTH parameter slots.
fn launch_values(values: [u64; 11]) -> [u64; 13] {
    let mut abi = [0; 13];
    abi[..11].copy_from_slice(&values);
    abi
}

fn argument_pointers(values: &mut [u64; 13]) -> [*mut c_void; 13] {
    std::array::from_fn(|i| (&mut values[i] as *mut u64).cast())
}

fn launch_kernel(
    stream: &cuda::Stream,
    binary: &'static Binary,
    n: u64,
    values: [u64; 11],
) -> Result<()> {
    let module = module(stream.context() as usize, binary)?;
    let blocks = u32::try_from(n.div_ceil(256)).map_err(|_| Failure {
        kind: Kind::Invalid,
        detail: "CUDA decode grid exceeds u32".into(),
    })?;
    if blocks == 0 {
        return Ok(());
    }
    let mut values = launch_values(values);
    // These are thirteen LIVE host addresses, not null entries. The final two
    // addresses point at zero device-pointer values for the unused scratch inputs.
    let mut args = argument_pointers(&mut values);
    let driver = cuda::driver()?;
    cuda::check("cuLaunchKernel", unsafe {
        (driver.launch_kernel)(
            module.function as _,
            blocks,
            1,
            1,
            binary.warps * 32,
            1,
            1,
            binary.shared,
            stream.raw() as _,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    Ok(())
}

fn owned_input(hold: &mut DecodeHold, device: i32, data: &[u8]) -> Result<u64> {
    let input = upload(device, data)?;
    let ptr = input.ptr();
    hold.inputs.push(input);
    Ok(ptr)
}

fn source_pointer(parts: &[Part<'_>], index: usize) -> Result<u64> {
    match parts.get(index).map(|p| p.src) {
        Some(Src::Device { ptr, .. }) => Ok(ptr),
        _ => crate::failure::fail(
            Kind::Invalid,
            "decode input slot is not a resident device part",
        ),
    }
}

fn factor_input(
    hold: &mut DecodeHold,
    device: i32,
    launch: &Launch<'_>,
    family: u32,
    factors: &Factors,
    parts: &[Part<'_>],
) -> Result<u64> {
    let (index, dtype, count) = match factors {
        Factors::Host(values) => return owned_input(hold, device, bytes(values)),
        Factors::Slot {
            index,
            dtype,
            count,
        } => (*index, dtype, *count),
    };
    let src = source_pointer(parts, index)?;
    if *dtype == Dtype::F32 {
        return Ok(src);
    }
    let mode = match dtype {
        Dtype::BF16 => 0,
        Dtype::F16 => 1,
        _ if decode::elem::table(dtype).is_some() => 2,
        _ => {
            return crate::failure::fail(
                Kind::Unsupported,
                format!("resident factor dtype {dtype} is unsupported"),
            );
        }
    };
    let len = count.checked_mul(4).ok_or_else(|| Failure {
        kind: Kind::Invalid,
        detail: "factor allocation overflow".into(),
    })?;
    let buffer = DeviceBuffer::new(device, len)?;
    let out = buffer.ptr();
    hold.scratch.push(buffer);
    let table = decode::elem::table(dtype).unwrap_or_else(|| vec![0.0]);
    let table = owned_input(hold, device, bytes(&table))?;
    let params = owned_input(hold, device, bytes(&[count as u64, mode]))?;
    let binary = launch
        .stage_binary(family, "cast_metadata")
        .ok_or_else(|| Failure {
            kind: Kind::Unsupported,
            detail: format!(
                "no prebuilt CUDA sm{family} cast_metadata/{} kernel in this build",
                launch.dtype
            ),
        })?;
    launch_kernel(
        hold.stream.as_ref().expect("decode stream live"),
        binary,
        count as u64,
        [out, src, 0, 0, 0, table, 0, 0, 0, 0, params],
    )?;
    Ok(out)
}

fn execute(device: i32, launch: &Launch<'_>) -> Result<DeviceBuffer> {
    execute_into(device, launch, &[], None)?.ok_or_else(|| Failure {
        kind: Kind::Invalid,
        detail: "decode output missing".into(),
    })
}

/// External parts/output remain owned by the unsafe caller. Unknown completion retains this
/// private stream and ALL engine allocations; io::drain is the checked external-owner boundary.
fn execute_into(
    device: i32,
    launch: &Launch<'_>,
    parts: &[Part<'_>],
    destination: Option<Dst>,
) -> Result<Option<DeviceBuffer>> {
    let width = if launch.dtype == Dtype::F32 { 4 } else { 2 };
    let size = launch.params[5]
        .checked_mul(width)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Failure {
            kind: Kind::Invalid,
            detail: "CUDA decode size overflow".into(),
        })?;
    let output = if destination.is_none() {
        Some(DeviceBuffer::new(device, size)?)
    } else {
        None
    };
    if size == 0 && destination.is_none() {
        return Ok(output);
    }
    let mut hold = DecodeHold {
        output,
        inputs: Vec::new(),
        scratch: Vec::new(),
        stream: Some(cuda::Stream::new(device)?),
        complete: false,
    };
    let _context = CtxGuard::enter(hold.stream.as_ref().expect("decode stream live").context())?;
    // Decode is synchronous. Synchronize each supplied producer/output stream on the calling
    // OS thread (important for PTDS), before kernels access borrowed allocations.
    let mut streams = std::collections::BTreeSet::new();
    for part in parts {
        if let Src::Device { stream, .. } = part.src {
            streams.insert(stream);
        }
    }
    if let Some(Dst::Device { stream, .. }) = destination {
        streams.insert(stream);
    }
    for stream in streams {
        cuda::check("cuStreamSynchronize(decode producer)", unsafe {
            (cuda::driver()?.stream_synchronize)(stream as _)
        })?;
    }
    // Zero logical output can still have a nonempty optional codebook. Success also
    // proves those supplied producers are complete before the caller releases owners.
    if size == 0 {
        hold.complete = true;
        return Ok(hold.output.take());
    }
    let family = (cuda::attribute(device, 75)? * 10 + cuda::attribute(device, 76)?) as u32;
    let binary = |stage: &str| {
        launch.stage_binary(family, stage).ok_or_else(|| Failure {
            kind: Kind::Unsupported,
            detail: format!(
                "no prebuilt CUDA sm{family} {stage}/{} kernel in this build",
                launch.dtype
            ),
        })
    };
    binary(launch.kind)?;
    if matches!(launch.kind, "rotated" | "grouped_rotated") {
        binary("hadamard")?;
        binary(if launch.kind == "grouped_rotated" {
            "requantize"
        } else {
            "unpack"
        })?;
    }
    let words = match launch.words.binding() {
        Some((index, _)) => source_pointer(parts, index)?,
        None => owned_input(&mut hold, device, launch.words.host()?)?,
    };
    let slots = owned_input(&mut hold, device, bytes(&launch.slots))?;
    let scale = factor_input(&mut hold, device, launch, family, &launch.scale, parts)?;
    let term = factor_input(&mut hold, device, launch, family, &launch.term, parts)?;
    let table = factor_input(&mut hold, device, launch, family, &launch.table, parts)?;
    let global = factor_input(&mut hold, device, launch, family, &launch.global, parts)?;
    let post = factor_input(&mut hold, device, launch, family, &launch.post, parts)?;
    let index = owned_input(&mut hold, device, bytes(&launch.index))?;
    let error = owned_input(&mut hold, device, bytes(&[0i32]))?;
    let params = owned_input(&mut hold, device, bytes(&launch.params))?;
    let output = match destination {
        Some(Dst::Device { ptr, .. }) => ptr,
        None => hold.output.as_ref().expect("decode output live").ptr(),
        _ => unreachable!("checked device destination"),
    };
    let stream = hold.stream.as_ref().expect("decode stream live");
    let call = |stage: &str, out: u64, input: u64, n: u64| -> Result<()> {
        launch_kernel(
            stream,
            binary(stage)?,
            n,
            [
                out, input, slots, scale, term, table, global, post, index, error, params,
            ],
        )
    };
    let scratch = &mut hold.scratch;
    let result = (|| {
        if matches!(launch.kind, "rotated" | "grouped_rotated") {
            let total = launch.params[0];
            let len = total
                .checked_mul(4)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Failure {
                    kind: Kind::Invalid,
                    detail: "rotation scratch overflow".into(),
                })?;
            let start = scratch.len();
            scratch.push(DeviceBuffer::new(device, len)?);
            scratch.push(DeviceBuffer::new(device, len)?);
            call(
                if launch.kind == "grouped_rotated" {
                    "requantize"
                } else {
                    "unpack"
                },
                scratch[start].ptr(),
                words,
                total,
            )?;
            let mut current = 0;
            let mut stride = 1;
            while stride < launch.params[28] {
                let mut next = launch.params;
                next[31] = stride;
                let parameter_buffer = hold
                    .inputs
                    .iter()
                    .find(|b| b.ptr() == params)
                    .expect("parameter allocation live");
                write(parameter_buffer, bytes(&next), stream.raw())?;
                call(
                    "hadamard",
                    scratch[start + 1 - current].ptr(),
                    scratch[start + current].ptr(),
                    total,
                )?;
                current = 1 - current;
                stride *= 4;
            }
            call(
                launch.kind,
                output,
                scratch[start + current].ptr(),
                launch.params[5],
            )?;
        } else {
            call(launch.kind, output, words, launch.params[5])?;
        }
        let mut value = [0u8; 4];
        let copy = Copy {
            src: Src::Device {
                device,
                ptr: error,
                stream: stream.raw(),
            },
            dst: Dst::host(&mut value),
        };
        unsafe { super::read(&[copy], &ReadOpts::default()) }.map_err(io_error)?;
        if i32::from_le_bytes(value) != 0 {
            return crate::failure::fail(
                Kind::Decode,
                "CUDA decode encountered an invalid index or a NaN requantized value",
            );
        }
        Ok(())
    })();
    let drained = stream.drain().and_then(|()| {
        if result.is_err() {
            super::drain(device)
        } else {
            Ok(())
        }
    });
    if let Err(error) = drained {
        return match result {
            Err(error) => Err(error),
            Ok(()) => Err(error.into()),
        };
    }
    hold.complete = true;
    result?;
    Ok(hold.output.take())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::decode::plan::{Elems, Post};

    #[test]
    fn launch_frame_includes_live_zero_scratch_slots_after_source_arguments() {
        let source = std::array::from_fn(|i| 0x1000 + i as u64 * 8);
        let mut values = launch_values(source);
        let pointers = argument_pointers(&mut values);
        assert_eq!(&values[..11], &source);
        for (i, pointer) in pointers.iter().enumerate() {
            assert!(
                !pointer.is_null(),
                "CUDA reads a host address for every argument"
            );
            assert_eq!(*pointer as usize, values.as_ptr().wrapping_add(i) as usize);
            // SAFETY: each pointer above addresses its live u64 slot, as CUDA's driver does.
            let value = unsafe { *pointer.cast::<u64>() };
            assert_eq!(value, if i < 11 { source[i] } else { 0 });
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn large_resident_factor_metadata_is_not_truncated_to_i32_offsets() {
        let (rows, cols, group) = (16385u64, 262144u64, 4u64);
        let shape = [rows, cols];
        let weight = [rows, cols / 2];
        let relative = [rows, cols / group];
        let channel = [rows];
        let correction = [cols / group, rows];
        let typed = [
            ("weight", Dtype::I8, weight.as_slice()),
            ("weight_s_rel", Dtype::F32, relative.as_slice()),
            ("weight_s_channel", Dtype::F32, channel.as_slice()),
            ("weight_correction", Dtype::BF16, correction.as_slice()),
        ];
        let metadata: Vec<_> = typed
            .iter()
            .map(|(role, dtype, shape)| decode::PartMetadata {
                role,
                dtype,
                shape,
                len: dtype.byte_len(shape).unwrap() as usize,
            })
            .collect();
        let encoding = Encoding {
            scheme: "comfy.asym_w4a8_int8".into(),
            params: BTreeMap::from([
                ("group_size".into(), group.into()),
                ("rotation_size".into(), 256u64.into()),
            ]),
        };
        check(&encoding, &Dtype::BF16, &shape, &metadata).unwrap();
        decode::planned_sources(
            &Dtype::BF16,
            &shape,
            &encoding,
            &metadata,
            |plan, geometry| {
                let Prepared::Kernel(launch) =
                    kernels::prepare(plan, geometry, &Dtype::BF16, 0, rows * cols)?
                else {
                    panic!("resident plan fell back to Host");
                };
                let Factors::Slot { dtype, count, .. } = &launch.term else {
                    panic!("correction was expanded on Host");
                };
                assert_eq!(*dtype, Dtype::BF16);
                assert_eq!(*count as u64, rows * (cols / group));
                assert!(*count > (i32::MAX as usize / 2));
                // These are metadata-only slots: no multi-GiB input/scratch allocation or CUDA.
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn external_output_refusal_before_owned_inputs_does_not_panic_in_cleanup() {
        // An early missing-catalog/producer error used to call retain_decode(empty),
        // replacing the typed refusal with its "decode retains its output" panic.
        drop(DecodeHold {
            output: None,
            inputs: Vec::new(),
            scratch: Vec::new(),
            stream: None,
            complete: false,
        });
    }

    #[test]
    fn host_values_and_metadata_only_sentinel_never_expand_or_upload_for_gpu_decode() {
        let geometry = decode::Geometry {
            rows: 2,
            cols: 3,
            height: 2,
            swap: true,
        };
        for values in [Vec::new(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] {
            let plan = Plan {
                core: Core::Values(values),
                post: Post::One(Elems::f32(2.0)),
            };
            let error = device_plan(plan, geometry, &Dtype::F32, 1, 3)
                .err()
                .expect("host values became a GPU decode");
            assert_eq!(error.kind, Kind::Unsupported);
        }
    }
}
