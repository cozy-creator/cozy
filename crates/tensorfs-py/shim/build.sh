#!/usr/bin/env bash
# varena's torch allocator shim: the one part of TensorFS built against torch's C++ ABI. Runs in
# quay.io/pypa/manylinux_2_28_x86_64 (torch 2.14 needs glibc 2.28 and the CXX11 ABI) from
# crates/tensorfs-py, before the wheel build. Writes python/tensorfs/_varena_torch.so, linked to
# torch's own libc10 (RUNPATH into the installed torch), and its build record, which Memory.open
# checks against the running torch before loading it.
set -euo pipefail
torch=2.14.1
venv=$(mktemp -d)
/opt/python/cp312-cp312/bin/python -m venv "$venv"
"$venv/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu "torch==$torch"
read -r site abi minor < <("$venv/bin/python" -c 'import os, torch; print(os.path.dirname(torch.__file__), int(torch._C._GLIBCXX_USE_CXX11_ABI), ".".join(torch.__version__.split(".")[:2]))')
g++ -O2 -std=c++20 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI="$abi" -I"$site/include" shim/varena_torch.cpp \
    -L"$site/lib" -lc10 -Wl,--no-undefined -Wl,-rpath,'$ORIGIN/../torch/lib' -o python/tensorfs/_varena_torch.so
printf '{"torch": "%s", "cxx11_abi": %s}\n' "$minor" "$abi" > python/tensorfs/_varena_torch.json
