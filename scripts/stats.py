#!/usr/bin/env python3
"""proto-061: `tensorfs.stats()` counts catalog opens, closure presence passes and repo-cache
verified reads, so a consumer can assert how often a warm path walks or rehashes.

    scripts/stats.py <store> <manifest-hex>
"""

import sys
import tempfile

import tensorfs


def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: after[name] - before[name] for name in before}


def main() -> None:
    root, snap = sys.argv[1], sys.argv[2]
    first = tensorfs.stats()
    assert set(first) == {"catalog_opens", "presence_passes", "repo_cache_verified_reads"}, first
    assert all(isinstance(value, int) and value >= 0 for value in first.values()), first

    store = tensorfs.Store.open(root)
    with store.acquire_cozytensors(snap) as lease:
        objects = list(lease.objects)
    warm = tensorfs.stats()
    for _ in range(3):
        store.acquire_cozytensors(snap).release()
    moved = delta(warm, tensorfs.stats())
    assert moved == {"catalog_opens": 0, "presence_passes": 3, "repo_cache_verified_reads": 0}, moved

    before = tensorfs.stats()
    assert store.complete_cozytensors_manifests() == ["sha256:" + snap]
    assert delta(before, tensorfs.stats())["presence_passes"] == 1

    blob, length = objects[0]
    with tempfile.TemporaryDirectory(prefix="tensorfs-stats-") as directory:
        cache = tensorfs.RepoObjectCache(f"{directory}/cache")
        assert cache.backfill(store, blob, length) == "stored"
        before = tensorfs.stats()
        # A blob already at its cache name with its length is present without a read.
        assert cache.backfill(store, blob, length) == "present"
        assert delta(before, tensorfs.stats())["repo_cache_verified_reads"] == 0
        destination = tensorfs.Store.init(f"{directory}/store")
        assert cache.admit(destination, blob, length) == "hit"
        assert delta(before, tensorfs.stats())["repo_cache_verified_reads"] == 1
    print(f"stats: 3 warm leases = 3 presence passes, 0 catalog opens; cache admit = 1 read; {tensorfs.stats()}")


if __name__ == "__main__":
    main()
