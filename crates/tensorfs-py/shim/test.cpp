// The shim's capture routes against a counting agent (shim/build.sh runs it after the build). A
// capture whose capture_begin threw is abandoned by its thread: its route goes, varena ends and
// releases the pool once, the next allocation is no longer routed to it, and abandoning again, a
// pool never begun, or another thread's capture, does nothing.
#include <cstdint>
#include <cstdio>
#include <functional>
#include <thread>
#include <utility>

using Stream = struct CUstream_st*;
using MempoolId = std::pair<unsigned long long, unsigned long long>;

struct Agent {
    uint64_t (*alloc)(uint64_t, int, uintptr_t, int, uint64_t, uint64_t, char*, size_t);
    void (*free)(uint64_t, uint64_t, int, uintptr_t);
    void (*record_stream)(uint64_t, uintptr_t);
    void (*pool_begin)(int, uint64_t, uint64_t);
    void (*pool_end)(int, uint64_t, uint64_t);
    void (*pool_release)(int, uint64_t, uint64_t);
};

extern "C" {
void varena_torch_bind(const Agent* functions);
void* varena_torch_malloc(size_t bytes, int device, Stream stream);
void varena_torch_begin_pool(int device, MempoolId pool, std::function<bool(Stream)> filter);
void varena_torch_end_pool(int device, MempoolId pool);
void varena_torch_abandon_pool(int device, uint64_t pool_a, uint64_t pool_b);
}

namespace {
int routed, plain, begins, ends, releases, failures;

void expect(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    failures += ok ? 0 : 1;
}
}  // namespace

int main() {
    Agent agent{
        [](uint64_t, int, uintptr_t, int r, uint64_t, uint64_t, char*, size_t) -> uint64_t { (r ? routed : plain)++; return 0x1000; },
        [](uint64_t, uint64_t, int, uintptr_t) {},
        [](uint64_t, uintptr_t) {},
        [](int, uint64_t, uint64_t) { begins++; },
        [](int, uint64_t, uint64_t) { ends++; },
        [](int, uint64_t, uint64_t) { releases++; },
    };
    varena_torch_bind(&agent);
    const Stream stream = reinterpret_cast<Stream>(0x10);
    varena_torch_begin_pool(0, {0, 7}, [](Stream) { return true; });
    varena_torch_malloc(256, 0, stream);
    expect(begins == 1 && routed == 1, "a begun capture routes the stream's allocations to its pool");
    varena_torch_abandon_pool(0, 0, 7);
    expect(ends == 1 && releases == 1, "abandoning it ends and releases the pool once");
    varena_torch_malloc(256, 0, stream);
    expect(routed == 1 && plain == 1, "the next allocation is no longer routed");
    varena_torch_abandon_pool(0, 0, 7);
    varena_torch_abandon_pool(0, 0, 9);
    expect(ends == 1 && releases == 1, "abandoning it again, or a pool never begun, does nothing");
    varena_torch_begin_pool(0, {0, 7}, [](Stream) { return true; });
    varena_torch_end_pool(0, {0, 7});
    varena_torch_abandon_pool(0, 0, 7);
    expect(ends == 2 && releases == 1, "a capture torch ended is not abandoned after it");
    varena_torch_begin_pool(0, {0, 7}, [](Stream) { return true; });
    std::thread([] { varena_torch_abandon_pool(0, 0, 7); }).join();
    expect(ends == 2 && releases == 1, "another thread's abandon leaves this thread's capture");
    varena_torch_abandon_pool(0, 0, 7);
    expect(ends == 3 && releases == 2, "its own thread's abandon ends it");
    return failures == 0 ? 0 : 1;
}
