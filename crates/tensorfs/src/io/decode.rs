//! Checked format plans executed by the prebuilt catalog through the read engine's driver.

use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::sync::{Arc, Mutex, OnceLock};

use super::cuda::{self, CtxGuard};
use super::{Copy, DeviceBuffer, Dst, ReadOpts, Src};
use crate::failure::{Failure, Kind, Result};
use crate::format::decode::plan::{Core, Plan};
use crate::format::{decode, Dtype, Encoding, PartBytes};
use crate::kernels::{self, Binary, Launch, Prepared};

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

fn upload(device: i32, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new(device, bytes.len())?;
    write(&buffer, bytes, 0)?;
    Ok(buffer)
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
    unsafe { super::read(&[copy], &ReadOpts::default()) }.map_err(io_error)?;
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

fn execute(device: i32, launch: &Launch<'_>) -> Result<DeviceBuffer> {
    let width = if launch.dtype == Dtype::F32 { 4 } else { 2 };
    let size = launch.params[5]
        .checked_mul(width)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Failure {
            kind: Kind::Invalid,
            detail: "CUDA decode size overflow".into(),
        })?;
    let output = DeviceBuffer::new(device, size)?;
    if output.is_empty() {
        return Ok(output);
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
    // Refuse before allocating/uploading the plan if this wheel has no family catalog.
    binary(launch.kind)?;
    let inputs = [
        upload(device, launch.words)?,
        upload(device, bytes(&launch.slots))?,
        upload(device, bytes(&launch.scale))?,
        upload(device, bytes(&launch.term))?,
        upload(device, bytes(&launch.table))?,
        upload(device, bytes(&launch.global))?,
        upload(device, bytes(&launch.post))?,
        upload(device, bytes(&launch.index))?,
        upload(device, bytes(&[0i32]))?,
        upload(device, bytes(&launch.params))?,
    ];
    let stream = cuda::Stream::new(device)?;
    let _context = CtxGuard::enter(output.context())?;
    let driver = cuda::driver()?;
    let call = |stage: &str, out: u64, words: u64, n: u64| -> Result<()> {
        let binary = binary(stage)?;
        let module = module(output.context() as usize, binary)?;
        let blocks = u32::try_from(n.div_ceil(256)).map_err(|_| Failure {
            kind: Kind::Invalid,
            detail: "CUDA decode grid exceeds u32".into(),
        })?;
        let mut values = [
            out,
            words,
            inputs[1].ptr(),
            inputs[2].ptr(),
            inputs[3].ptr(),
            inputs[4].ptr(),
            inputs[5].ptr(),
            inputs[6].ptr(),
            inputs[7].ptr(),
            inputs[8].ptr(),
            inputs[9].ptr(),
        ];
        let mut args = values
            .iter_mut()
            .map(|v| (v as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
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
    };
    // Keep every allocation alive until the stream is drained, also on launch/read errors.
    let mut scratch = Vec::new();
    let result = (|| {
        if launch.kind == "rotated" {
            let total = launch.params[0];
            let len = total
                .checked_mul(4)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Failure {
                    kind: Kind::Invalid,
                    detail: "rotation scratch overflow".into(),
                })?;
            scratch.push(DeviceBuffer::new(device, len)?);
            scratch.push(DeviceBuffer::new(device, len)?);
            call("unpack", scratch[0].ptr(), inputs[0].ptr(), total)?;
            let mut current = 0;
            let mut stride = 1;
            while stride < launch.params[28] {
                let mut params = launch.params;
                params[31] = stride;
                write(&inputs[9], bytes(&params), stream.raw())?;
                call(
                    "hadamard",
                    scratch[1 - current].ptr(),
                    scratch[current].ptr(),
                    total,
                )?;
                current = 1 - current;
                stride *= 4;
            }
            call(
                "rotated",
                output.ptr(),
                scratch[current].ptr(),
                launch.params[5],
            )?;
        } else {
            call(launch.kind, output.ptr(), inputs[0].ptr(), launch.params[5])?;
        }
        let mut error = [0u8; 4];
        let copy = Copy {
            src: Src::Device {
                device,
                ptr: inputs[8].ptr(),
                stream: stream.raw(),
            },
            dst: Dst::host(&mut error),
        };
        unsafe { super::read(&[copy], &ReadOpts::default()) }.map_err(io_error)?;
        if i32::from_le_bytes(error) != 0 {
            return Err(Failure {
                kind: Kind::Decode,
                detail: "CUDA decode encountered an invalid scale index".into(),
            });
        }
        Ok(())
    })();
    let drained = cuda::check("cuStreamSynchronize", unsafe {
        (driver.stream_synchronize)(stream.raw() as _)
    });
    result?;
    drained?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::decode::plan::{Elems, Post};

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
