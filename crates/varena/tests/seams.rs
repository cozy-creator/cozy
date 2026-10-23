//! varena's OS seams on every OS (Linux, macOS, Windows), with no stored set (TensorFS's
//! sources are Unix's so far): host pages shared through an exported handle.

use varena::{Access, Backend, HostBackend};

const MB: u64 = 1 << 20;

/// A page made and filled here, exported, and imported again: a second mapping of the same bytes,
/// read-only (a memfd, an `shm_open` object, a pagefile section).
#[test]
fn host_pages_are_shared_through_an_exported_handle() {
    let backend = HostBackend::new(64 * MB);
    let len = backend.granularity().max(2 * MB);
    let (va, page) = (backend.reserve(len).unwrap(), backend.create(len).unwrap());
    backend.map(va, 0, &page, Access::ReadWrite).unwrap();
    let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    // SAFETY: `va` maps `len` writable bytes of `page`, nothing else aliases them here.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), va.base as *mut u8, len as usize) };
    backend.protect(va, 0, len, Access::Read).unwrap();
    let imported = backend.import(backend.export(&page).unwrap(), len).unwrap();
    let again = backend.reserve(len).unwrap();
    backend.map(again, 0, &imported, Access::Read).unwrap();
    // SAFETY: `again` maps `len` readable bytes of the same memory.
    let seen = unsafe { std::slice::from_raw_parts(again.base as *const u8, len as usize) };
    assert!(seen == bytes.as_slice(), "the imported page holds the same bytes");
    for (at, p) in [(va, page), (again, imported)] {
        backend.unmap(at, 0, len).unwrap();
        backend.destroy(p).unwrap();
        backend.release(at);
    }
}

