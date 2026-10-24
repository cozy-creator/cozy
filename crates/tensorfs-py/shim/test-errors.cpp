// CPU-only regression of the actual allocator shim against Torch's exception types.
// Compile with the same include/ABI/libc10 flags as varena_torch.cpp, then run the executable.
#include "varena_torch.cpp"
#include <cassert>
#include <cstring>
#include <iostream>

namespace {
Failure answer;
uint64_t refuse_all(uint64_t, int, uintptr_t, int, uint64_t, uint64_t, char* why, size_t len, Failure* failure) {
    *failure = answer;
    const char* message = "injected allocator refusal";
    std::strncpy(why, message, len);
    return 0;
}
void free_none(uint64_t, uint64_t, int, uintptr_t) {}
void record_none(uint64_t, uintptr_t) {}
void pool_none(int, uint64_t, uint64_t) {}
}

int main() {
    const Agent functions{refuse_all, free_none, record_none, pool_none, pool_none, pool_none};
    varena_torch_bind_v2(&functions);
    for (int kind : {1, 2, 3, 4, 5, 6}) {
        answer = {kind, kind == 2 ? 700 : -1};
        bool caught = false;
        try {
            varena_torch_malloc(4096, 0, nullptr);
        } catch (const c10::OutOfMemoryError&) {
            assert(kind == 1);
            caught = true;
        } catch (const c10::AcceleratorError& error) {
            assert(kind == 2 || kind == 3 || kind == 4);
            assert(error.get_error_code() == (kind == 2 ? 700 : -1));
            caught = true;
        } catch (const c10::ValueError&) {
            assert(kind == 5);
            caught = true;
        } catch (const c10::Error&) {
            assert(kind == 6);
            caught = true;
        }
        assert(caught);
    }
    bool invalid_stream = false;
    try {
        varena_torch_malloc(4096, 0, reinterpret_cast<Stream>(0x2));
    } catch (const c10::ValueError&) {
        invalid_stream = true;
    }
    assert(invalid_stream);
    std::cout << "PASS: only capacity is OOM; CUDA codes, ownership, internal, invalid and other failures keep their exception class\n";
}
