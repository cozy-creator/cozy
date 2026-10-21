//! Kept pages. An executor exports each filled, read-only page into TensorD's custody while it
//! maps it, and withdraws it when it unmaps it; custody holds the handles by (fingerprint,
//! page). When the executor dies, `reap` turns what it held into orphans (class O), which the
//! next process to load that fingerprint imports instead of copying: one load path.
