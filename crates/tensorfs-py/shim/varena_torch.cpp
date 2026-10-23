// varena's agent as torch's CUDA allocator (design/varena-leases.md §11, varena-slim step 6).
//
// torch's pluggable allocator calls these hooks; each forwards to the agent's C ABI in the
// TensorFS extension, whose function table `varena_torch_bind` receives. The one part of TensorFS
// built against torch's C++ ABI: a refusal must be thrown as c10::OutOfMemoryError (torch never
// checks a NULL from the allocator), and graph capture hands over a std::function filter. It
// includes no CUDA header: a stream is only a handle here.
#include <c10/util/Exception.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define VARENA_EXPORT __declspec(dllexport)
#else
#define VARENA_EXPORT __attribute__((visibility("default")))
#endif

using Stream = struct CUstream_st*;
using MempoolId = std::pair<unsigned long long, unsigned long long>;  // c10::cuda::MempoolId_t
using Filter = std::function<bool(Stream)>;

namespace {

// The agent's C ABI (tensorfs-py src/agent.rs). `alloc` returns 0 and writes its reason on a
// refusal; none of them unwinds.
struct Agent {
    uint64_t (*alloc)(uint64_t bytes, int device, uintptr_t stream, int routed, uint64_t pool_a,
                      uint64_t pool_b, char* why, size_t why_len);
    void (*free)(uint64_t ptr, uint64_t bytes, int device, uintptr_t stream);
    void (*record_stream)(uint64_t ptr, uintptr_t stream);
    void (*pool_begin)(int device, uint64_t pool_a, uint64_t pool_b);
    void (*pool_end)(int device, uint64_t pool_a, uint64_t pool_b);
    void (*pool_release)(int device, uint64_t pool_a, uint64_t pool_b);
};

Agent agent;

// A capture routing to a pool: torch's filter, kept until end_allocate_to_pool (the graph it
// captures may be destroyed after that).
struct Route {
    int device;
    MempoolId pool;
    Filter filter;
    std::thread::id thread;  // the capturing thread, which begins and ends it
};

std::mutex routes_mutex;
std::vector<Route> routes;
std::atomic<size_t> routing{0};  // routes.size(), read without the mutex on every allocation

// The capture this thread routes into (device, pool), else any route of it (torch ends a
// capture on its capturing thread; the fallback keeps a stray end from leaking a route).
std::vector<Route>::iterator find_route(int device, const MempoolId& pool, bool this_thread_only) {
    const auto me = std::this_thread::get_id();
    auto any = routes.end();
    for (auto route = routes.begin(); route != routes.end(); ++route) {
        if (route->device == device && route->pool == pool) {
            if (route->thread == me) {
                return route;
            }
            if (any == routes.end()) {
                any = route;
            }
        }
    }
    return this_thread_only ? routes.end() : any;
}

const Stream per_thread = reinterpret_cast<Stream>(0x2);  // cudaStreamPerThread

[[noreturn]] void refuse(const char* why) {
    TORCH_CHECK_WITH(OutOfMemoryError, false, why);
    std::abort();  // not reached: the check throws
}

}  // namespace

extern "C" {

VARENA_EXPORT void varena_torch_bind(const Agent* functions) { agent = *functions; }

VARENA_EXPORT void* varena_torch_malloc(size_t bytes, int device, Stream stream) {
    if (stream == per_thread) {
        refuse("varena: cudaStreamPerThread (0x2) names another stream on every thread; allocate on a stream of its own");
    }
    int routed = 0;
    MempoolId pool{0, 0};
    if (routing.load(std::memory_order_acquire) != 0) {
        std::lock_guard<std::mutex> lock(routes_mutex);
        for (const Route& route : routes) {
            if (route.device == device && route.filter(stream)) {
                routed = 1;
                pool = route.pool;
                break;
            }
        }
    }
    char why[512] = {0};
    uint64_t ptr = agent.alloc(bytes, device, reinterpret_cast<uintptr_t>(stream), routed, pool.first, pool.second, why, sizeof why - 1);
    if (ptr == 0) {
        refuse(why[0] != 0 ? why : "varena: refused");
    }
    return reinterpret_cast<void*>(ptr);
}

VARENA_EXPORT void varena_torch_free(void* ptr, size_t bytes, int device, Stream stream) {
    agent.free(reinterpret_cast<uint64_t>(ptr), bytes, device, reinterpret_cast<uintptr_t>(stream));
}

VARENA_EXPORT void varena_torch_record_stream(void* ptr, Stream stream) {
    if (stream == per_thread) {
        refuse("varena: cudaStreamPerThread (0x2) names another stream on every thread; record a stream of its own");
    }
    agent.record_stream(reinterpret_cast<uint64_t>(ptr), reinterpret_cast<uintptr_t>(stream));
}

// Called by capture_begin on the capturing thread, before the capture starts.
VARENA_EXPORT void varena_torch_begin_pool(int device, MempoolId pool, Filter filter) {
    {
        std::lock_guard<std::mutex> lock(routes_mutex);
        routes.push_back({device, pool, std::move(filter), std::this_thread::get_id()});
        routing.store(routes.size(), std::memory_order_release);
    }
    agent.pool_begin(device, pool.first, pool.second);
}

VARENA_EXPORT void varena_torch_end_pool(int device, MempoolId pool) {
    {
        std::lock_guard<std::mutex> lock(routes_mutex);
        auto route = find_route(device, pool, false);
        if (route != routes.end()) {
            routes.erase(route);
        }
        routing.store(routes.size(), std::memory_order_release);
    }
    agent.pool_end(device, pool.first, pool.second);
}

VARENA_EXPORT void varena_torch_release_pool(int device, MempoolId pool) { agent.pool_release(device, pool.first, pool.second); }

// A capture whose capture_begin (or capture_end, before its end hook) threw: torch never ends or
// releases it. This thread's route into the pool (whose filter may hold the dead graph) goes, and
// varena ends and releases the pool. With no such route (never begun, already ended, or another
// thread's capture) it does nothing.
VARENA_EXPORT void varena_torch_abandon_pool(int device, uint64_t pool_a, uint64_t pool_b) {
    const MempoolId pool{pool_a, pool_b};
    bool begun = false;
    {
        std::lock_guard<std::mutex> lock(routes_mutex);
        auto route = find_route(device, pool, true);
        if (route != routes.end()) {
            routes.erase(route);
            begun = true;
        }
        routing.store(routes.size(), std::memory_order_release);
    }
    if (begun) {
        agent.pool_end(device, pool_a, pool_b);
        agent.pool_release(device, pool_a, pool_b);
    }
}

}  // extern "C"
