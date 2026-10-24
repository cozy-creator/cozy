use std::sync::Arc;
use varena::arena::{Direct, Torch};
use varena::backend::Allowance;
use varena::{Error, GpuId, HostBackend, Memory, Result};

#[test]
fn poisoned_memory_does_not_reuse_an_already_free_arena_block() -> Result<()> {
    let memory = Memory::open(GpuId { uuid: "quarantine-reuse".into(), model: "host".into(), memory: 64 << 20 }, Box::new(HostBackend::new(64 << 20)))?;
    memory.serve_standalone_torch()?;
    let ptr = memory.torch_alloc(512, 0, None)?;
    memory.torch_free(ptr, 512, 0);
    memory.quarantine_allocator();
    assert!(matches!(memory.torch_alloc(512, 0, None), Err(Error::Ownership(_))));
    assert!(matches!(memory.request("after-fault", Default::default()), Err(Error::Ownership(_))));
    Ok(())
}

#[test]
fn ownership_failure_keeps_source_reservations_alive_after_last_handle() -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use varena::arena::{Chunk, Kind, Source};
    use varena::Va;

    struct Fails {
        direct: Direct,
        fail: Arc<AtomicBool>,
    }
    impl Source for Fails {
        fn map(&self, va: Va, kind: Kind, into: &mut Vec<Chunk>) -> Result<()> {
            if self.fail.load(Ordering::Acquire) {
                return Err(Error::Ownership("unproven map rollback".into()));
            }
            self.direct.map(va, kind, into)
        }
        fn unmap(&self, va: Va, chunks: Vec<Chunk>) -> Vec<(u64, Chunk)> { self.direct.unmap(va, chunks) }
    }

    let backend = Arc::new(HostBackend::new(128 << 20));
    let fail = Arc::new(AtomicBool::new(false));
    let source = Arc::new(Fails { direct: Direct::new(backend.clone(), Arc::new(Allowance::new(128 << 20))), fail: fail.clone() });
    let weak = Arc::downgrade(&source);
    let torch = Torch::new(backend, source)?;
    let ptr = torch.alloc(512, 0, None)?;
    torch.free(ptr, 512, 0);
    fail.store(true, Ordering::Release);
    assert!(matches!(torch.alloc(8 << 20, 0, None), Err(Error::Ownership(_))));
    assert!(matches!(torch.alloc(512, 0, None), Err(Error::Ownership(_))));
    drop(torch);
    assert!(weak.upgrade().is_some(), "a poisoned source must survive until process exit");
    Ok(())
}
