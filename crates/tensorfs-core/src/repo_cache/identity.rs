//! Scope optional cache writes to the local Store owner on root-squashed NFS.
//!
//! Only Linux filesystem IDs change, only on the current native thread, and only
//! while synchronous cache I/O runs. Real/effective IDs, other Runtime threads,
//! existing cache permissions, and the configured cache path are unchanged.

use super::SoftFailure;
use crate::store::Store;

#[cfg(target_os = "linux")]
mod linux {
    use std::fs;
    use std::marker::PhantomData;
    use std::os::unix::fs::MetadataExt;
    use std::rc::Rc;

    use nix::unistd::{geteuid, setfsgid, setfsuid, Gid, Uid};

    use super::{SoftFailure, Store};

    // Linux has no getfsuid/getfsgid. An invalid ID cannot be installed; each
    // setter returns the previous/current ID even when the attempted set fails.
    fn filesystem_ids() -> (Uid, Gid) {
        (
            setfsuid(Uid::from_raw(u32::MAX)),
            setfsgid(Gid::from_raw(u32::MAX)),
        )
    }

    pub(in crate::repo_cache) struct CacheIdentity {
        previous: Option<(Uid, Gid)>,
        // A credential guard must be destroyed by the exact thread that made it.
        _thread: PhantomData<Rc<()>>,
    }

    impl CacheIdentity {
        pub(in crate::repo_cache) fn for_store(store: &Store) -> Result<Self, SoftFailure> {
            let mut guard = Self {
                previous: None,
                _thread: PhantomData,
            };
            if !geteuid().is_root() {
                return Ok(guard);
            }
            // Authority is the already-open LOCAL Store's owner, never a UID
            // supplied by an artifact, request, or remote cache directory.
            let owner = fs::symlink_metadata(store.root()).map_err(|_| SoftFailure::Unavailable)?;
            if !owner.is_dir() || owner.uid() == u32::MAX || owner.gid() == u32::MAX {
                return Err(SoftFailure::Unavailable);
            }
            let desired = (Uid::from_raw(owner.uid()), Gid::from_raw(owner.gid()));
            let previous = filesystem_ids();
            if previous == desired {
                return Ok(guard);
            }
            guard.previous = Some(previous);
            setfsgid(desired.1);
            setfsuid(desired.0);
            if filesystem_ids() != desired {
                // Drop restores either field that changed before refusing the
                // optional cache write. The origin/local operation remains valid.
                return Err(SoftFailure::Unavailable);
            }
            Ok(guard)
        }
    }

    impl Drop for CacheIdentity {
        fn drop(&mut self) {
            let Some((uid, gid)) = self.previous else {
                return;
            };
            setfsuid(uid);
            setfsgid(gid);
            if filesystem_ids() != (uid, gid) {
                // Continuing a reused Runtime thread under the wrong identity is
                // unsafe. No cache error can reach here: this is a failed kernel
                // restoration of credentials that this thread previously held.
                std::process::abort();
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::panic::{catch_unwind, AssertUnwindSafe};
        use std::sync::{Arc, Barrier};
        use std::time::{SystemTime, UNIX_EPOCH};

        #[test]
        fn cache_identity_restores_and_does_not_change_other_threads() {
            if !geteuid().is_root() {
                eprintln!("root is required for the NFS cache identity qualification");
                return;
            }
            let root = std::env::temp_dir().join(format!(
                "tensorfs-cache-identity-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            ));
            let store = Store::init(&root).unwrap();
            std::os::unix::fs::chown(&root, Some(65532), Some(65532)).unwrap();
            let original = filesystem_ids();
            let desired = (Uid::from_raw(65532), Gid::from_raw(65532));
            let entered = Arc::new(Barrier::new(2));
            let observed = Arc::new(Barrier::new(2));
            std::thread::scope(|scope| {
                let other_entered = Arc::clone(&entered);
                let other_observed = Arc::clone(&observed);
                scope.spawn(move || {
                    other_entered.wait();
                    assert_eq!(filesystem_ids(), original);
                    other_observed.wait();
                });
                {
                    let _guard = CacheIdentity::for_store(&store).unwrap();
                    assert_eq!(filesystem_ids(), desired);
                    assert!(geteuid().is_root());
                    entered.wait();
                    observed.wait();
                    let path = root.join("cache-owner-created");
                    fs::write(&path, b"ordinary scoped cache write").unwrap();
                    let owner = fs::metadata(&path).unwrap();
                    assert_eq!((owner.uid(), owner.gid()), (65532, 65532));
                }
                assert_eq!(filesystem_ids(), original);
            });
            let failure: Result<(), SoftFailure> = (|| {
                let _guard = CacheIdentity::for_store(&store)?;
                Err(SoftFailure::Unavailable)
            })();
            assert!(failure.is_err());
            assert_eq!(filesystem_ids(), original);
            let panic = catch_unwind(AssertUnwindSafe(|| {
                let _guard = CacheIdentity::for_store(&store).unwrap();
                panic!("cache operation unwind");
            }));
            assert!(panic.is_err());
            assert_eq!(filesystem_ids(), original);
            drop(store);
            fs::remove_dir_all(root).unwrap();
        }
    }
}

#[cfg(target_os = "linux")]
pub(super) use linux::CacheIdentity;

#[cfg(not(target_os = "linux"))]
pub(super) struct CacheIdentity;

#[cfg(not(target_os = "linux"))]
impl CacheIdentity {
    pub(super) fn for_store(_: &Store) -> Result<Self, SoftFailure> {
        Ok(Self)
    }
}
