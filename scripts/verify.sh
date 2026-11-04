#!/usr/bin/env bash
# The only definition of "passes" for MENOS (CLAUDE.md). Runs the format
# check, clang-tidy, configure+build+test for the dev and release
# presets, a benchmarks/ configure+build+clang-tidy pass (Phase 3;
# benchmarks are not run here -- see README.md's "Benchmarks (opt-in)"),
# then the Python boundary: editable install, pytest (which also runs the
# examples/ scripts and every Python snippet in docs/), a wheel build, and
# a wheel-install smoke test in a clean venv, then the documentation site
# build when Node is available. Exits non-zero and prints which stage failed.
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() {
    echo "verify.sh: FAILED at stage: $1" >&2
    exit 1
}

echo "== Tool versions =="
cc --version | head -n1 || true
c++ --version | head -n1 || true
cmake --version | head -n1
clang-format --version || fail "clang-format not found"
clang-tidy --version | head -n1 || fail "clang-tidy not found"
python3 --version || fail "python3 not found"
uv --version || fail "uv not found"

echo "== Format check =="
mapfile -t cxx_files < <(git ls-files -- '*.h' '*.cpp' '*.hpp' '*.cc' '*.cu')
if [ "${#cxx_files[@]}" -gt 0 ]; then
    clang-format --dry-run --Werror "${cxx_files[@]}" || fail "clang-format"
else
    echo "no C++ files tracked yet"
fi

# A from-scratch FetchContent git clone of googletest/nlohmann_json has
# been observed to hang indefinitely in this environment (docs/adr/0002).
# When a prior run has already populated this repo-local cache, reuse it;
# otherwise let CMake clone normally (e.g. on a fresh machine or CI where
# the hang doesn't occur), populating the cache for next time.
deps_cache_defines=()
if [ -d "$repo_root/build/_deps-cache/googletest-src" ] && [ -d "$repo_root/build/_deps-cache/nlohmann_json-src" ]; then
    deps_cache_defines=(
        "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$repo_root/build/_deps-cache/googletest-src"
        "-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$repo_root/build/_deps-cache/nlohmann_json-src"
    )
fi

echo "== Configure dev (for compile_commands.json) =="
cmake --preset dev "${deps_cache_defines[@]}" || fail "configure dev"

echo "== clang-tidy (C++ dev/release targets) =="
# benchmarks/ is excluded here the same way src/python/ is: it only builds
# (and only has compile_commands.json entries) under a separate
# MENOS_BUILD_BENCHMARKS=ON configuration, clang-tidy'd in its own stage
# below -- running it against build/dev's compile database (where
# benchmarks/ was never configured) fails to find a compile command for
# every benchmark file and errors out.
#
# Translation units only, never headers. A header has no entry in
# compile_commands.json, so passing one makes clang-tidy guess its flags by
# interpolating from the "nearest" entry it can find; which entry that is
# depends on the checkout, and in a fresh clone it picks a googletest entry
# carrying no project include path, failing with "'menos/circuit.h' file not
# found". Our headers are still fully covered: diagnostics inside them are
# reported through whichever .cpp includes them, via .clang-tidy's
# HeaderFilterRegex.
#
# CUDA-only translation units are excluded for the same reason benchmarks/ is:
# they are compiled only under MENOS_ENABLE_CUDA=ON, so build/dev's compile
# database has no entry for them. They are tidied in the CUDA stage below,
# against build/cuda-dev. (.cu files are never passed to clang-tidy at all --
# it cannot parse CUDA without a matching --cuda-path configuration.)
# docs/benchmarks/ holds the measurement records' own driver sources (Phase
# 23's pass-count driver, built by hand against the commit it measured); they
# are records, not translation units of any build.
mapfile -t cxx_files_no_python < <(git ls-files -- '*.cpp' '*.cc' \
    | grep -v '^src/python/' \
    | grep -v '^benchmarks/' \
    | grep -v '^docs/' \
    | grep -v '^src/kernels/cuda/' \
    | grep -v '^src/engines/dense_cuda/' \
    | grep -v '^src/plan/cuda_calibration\.cpp$' \
    | grep -v 'cuda.*_test\.cpp$')
if [ "${#cxx_files_no_python[@]}" -gt 0 ]; then
    clang-tidy -p build/dev "${cxx_files_no_python[@]}" || fail "clang-tidy"
fi

echo "== dev: build =="
cmake --build --preset dev -j"$(nproc)" || fail "build dev"

echo "== dev: test =="
ctest --preset dev --output-on-failure || fail "test dev"

echo "== release: configure =="
cmake --preset release "${deps_cache_defines[@]}" || fail "configure release"

echo "== release: build =="
cmake --build --preset release -j"$(nproc)" || fail "build release"

echo "== release: test =="
ctest --preset release --output-on-failure || fail "test release"

echo "== benchmarks: configure (MENOS_BUILD_BENCHMARKS=ON) =="
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DMENOS_BUILD_BENCHMARKS=ON \
    "${deps_cache_defines[@]}" || fail "configure benchmarks"

echo "== benchmarks: build =="
cmake --build build/bench -j"$(nproc)" || fail "build benchmarks"

echo "== clang-tidy (benchmarks/) =="
# Translation units only, for the same reason as the stage above.
#
# The CUDA benchmark is excluded here and tidied in the CUDA stage instead:
# build/bench is configured with MENOS_ENABLE_CUDA=OFF, so it has no compile
# command for that file, and clang-tidy then interpolates flags from a
# neighbouring entry and fails to find <benchmark/benchmark.h>.
mapfile -t benchmark_files < <(git ls-files -- 'benchmarks/*.cpp' \
    | grep -v '^benchmarks/cuda_.*\.cpp$')
if [ "${#benchmark_files[@]}" -gt 0 ]; then
    clang-tidy -p build/bench "${benchmark_files[@]}" || fail "clang-tidy (benchmarks/)"
fi

echo "== python: create/use venv =="
if [ ! -x .venv/bin/python ]; then
    uv venv .venv --python 3.14 || fail "python venv creation"
fi

# scikit-build-core configures its own, separate build tree (build/py-*),
# so it needs the same cache-reuse treatment as the dev/release presets
# above, expressed as scikit-build-core's own cmake.define.* option form.
py_cmake_defines=()
if [ -d "$repo_root/build/_deps-cache/googletest-src" ] && [ -d "$repo_root/build/_deps-cache/nlohmann_json-src" ]; then
    py_cmake_defines=(
        -C "cmake.define.FETCHCONTENT_SOURCE_DIR_GOOGLETEST=$repo_root/build/_deps-cache/googletest-src"
        -C "cmake.define.FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$repo_root/build/_deps-cache/nlohmann_json-src"
    )
fi

echo "== python: editable install =="
# --no-build-isolation below means the build backend is NOT fetched into a
# throwaway isolated environment, so it has to already be in .venv. A
# long-lived .venv usually has it from an earlier run, which is exactly why
# this stage could pass locally and still fail on a fresh clone with
# "ModuleNotFoundError: No module named 'scikit_build_core'". Install it
# explicitly instead of relying on that. These pins must match
# pyproject.toml's own [build-system] requires.
uv pip install --python .venv/bin/python "scikit-build-core==1.0.3" "nanobind==3.1.0" \
    || fail "build backend install"
uv pip install --python .venv/bin/python -e ".[test]" --no-build-isolation \
    "${py_cmake_defines[@]}" || fail "editable install"

echo "== clang-tidy (menos._core bindings) =="
py_build_dir=$(find build -maxdepth 1 -type d -name 'py-*' | head -n1)
if [ -z "$py_build_dir" ] || [ ! -f "$py_build_dir/compile_commands.json" ]; then
    fail "python build directory / compile_commands.json not found after editable install"
fi
clang-tidy -p "$py_build_dir" src/python/bindings.cpp || fail "clang-tidy (bindings.cpp)"

echo "== python: pytest (editable install) =="
.venv/bin/python -m pytest python/tests --deselect python/tests/test_wheel_smoke.py -v \
    || fail "pytest"

echo "== python: build wheel =="
rm -rf dist
uv build --wheel --python .venv/bin/python --no-build-isolation "${py_cmake_defines[@]}" \
    || fail "wheel build"

echo "== python: install wheel into a clean venv =="
rm -rf /tmp/menos-wheel-venv
uv venv /tmp/menos-wheel-venv --python 3.14 || fail "clean venv creation"
uv pip install --python /tmp/menos-wheel-venv/bin/python dist/*.whl "qiskit==2.5.2" \
    "pennylane==0.45.1" pytest || fail "wheel install"

echo "== python: wheel smoke test =="
/tmp/menos-wheel-venv/bin/python -m pytest python/tests/test_wheel_smoke.py -v \
    || fail "wheel smoke test"

# --- Documentation site -----------------------------------------------------
#
# The documentation under docs/ is part of the release, so its build is part of
# the gate. Auto-detected like the CUDA stages: a machine without Node skips the
# stage loudly rather than passing silently, and every Python snippet in the docs
# is still executed by the pytest stage above (python/tests/test_docs_snippets.py),
# which does not need Node.
#
# `npm run build` runs the structure check (orphan pages, empty pages,
# placeholder text), VitePress's own dead-link check, and the fragment-link check.
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    echo "== docs: install dependencies =="
    node --version
    (cd docs && npm ci --no-audit --no-fund) || fail "docs: npm ci"

    echo "== docs: build and link check =="
    (cd docs && npm run build) || fail "docs: build"
else
    echo "== docs: SKIPPED (node/npm not found; the site was NOT built) =="
fi

# --- CUDA -------------------------------------------------------------------
#
# Auto-detected: a machine without the toolkit skips these and the gate above
# is the whole gate, which is what keeps MENOS_ENABLE_CUDA=OFF the default and
# a CUDA-less clone unaffected. When the toolkit IS present these stages run,
# so the CUDA backend is never "verified" by being absent.
#
# The toolkit is addressed by absolute path on purpose. This machine has two,
# and the one first on PATH (/usr/bin/nvcc, 12.4) rejects GCC 15 outright --
# see docs/adr/0014.
cuda_nvcc=/usr/local/cuda/bin/nvcc
if [ -x "$cuda_nvcc" ]; then
    echo "== cuda: toolkit =="
    "$cuda_nvcc" --version | tail -n2
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader \
        || echo "no GPU visible; CUDA tests will skip themselves"

    echo "== cuda-dev: configure =="
    cmake --preset cuda-dev "${deps_cache_defines[@]}" || fail "configure cuda-dev"

    echo "== clang-tidy (CUDA C++ targets) =="
    mapfile -t cuda_cxx_files < <(git ls-files -- '*.cpp' \
        | grep -E '^(src/kernels/cuda/|src/engines/dense_cuda/|src/plan/cuda_calibration\.cpp)')
    if [ "${#cuda_cxx_files[@]}" -gt 0 ]; then
        clang-tidy -p build/cuda-dev "${cuda_cxx_files[@]}" || fail "clang-tidy (cuda)"
    fi

    echo "== cuda-dev: build =="
    cmake --build --preset cuda-dev -j"$(nproc)" || fail "build cuda-dev"

    echo "== cuda-dev: test =="
    # The cuda-dev test preset sets ASAN_OPTIONS=protect_shadow_gap=0. Without
    # it AddressSanitizer's shadow gap collides with the CUDA driver's address
    # reservation, cudaGetDeviceCount fails, and every CUDA test SKIPS while
    # reporting success -- a green run that never touched the GPU.
    # It also sets LSAN_OPTIONS to scripts/lsan-wsl-libcuda.supp, which
    # suppresses only the per-process leak inside WSL's CUDA driver shim
    # (docs/adr/0014 section 17); every other leak still fails the stage.
    ctest --preset cuda-dev --output-on-failure || fail "test cuda-dev"

    echo "== cuda-release: configure/build/test =="
    cmake --preset cuda-release "${deps_cache_defines[@]}" || fail "configure cuda-release"
    cmake --build --preset cuda-release -j"$(nproc)" || fail "build cuda-release"
    ctest --preset cuda-release --output-on-failure || fail "test cuda-release"

    echo "== cuda: benchmark builds =="
    # Built, not run: the crossover sweep takes minutes (its n=24 CPU rows
    # alone are ~75 s each) and benchmarks are not tests. Its measured output
    # is recorded in docs/adr/0014.
    cmake --preset cuda-release "${deps_cache_defines[@]}" -DMENOS_BUILD_BENCHMARKS=ON \
        || fail "configure cuda-release benchmarks"
    cmake --build --preset cuda-release --target menos_cuda_crossover_benchmark -j"$(nproc)" \
        || fail "build cuda crossover benchmark"
    cmake --build --preset cuda-release --target menos_cuda_factorized_backend_benchmark -j"$(nproc)" \
        || fail "build factorized backend benchmark"
    cmake --build --preset cuda-release --target menos_cuda_kernel_benchmark -j"$(nproc)" \
        || fail "build cuda kernel benchmark"
    cmake --build --preset cuda-release --target menos_cuda_memory_benchmark -j"$(nproc)" \
        || fail "build cuda memory benchmark"
    cmake --build --preset cuda-release --target menos_cuda_tiered_memory_benchmark -j"$(nproc)" \
        || fail "build cuda tiered memory benchmark"

    echo "== clang-tidy (CUDA benchmarks) =="
    # Now that build/cuda-release has both CUDA and benchmarks configured, the
    # CUDA benchmark has a real compile command and can be held to the same bar
    # as every other translation unit.
    mapfile -t cuda_benchmark_files < <(git ls-files -- 'benchmarks/cuda_*.cpp')
    if [ "${#cuda_benchmark_files[@]}" -gt 0 ]; then
        clang-tidy -p build/cuda-release "${cuda_benchmark_files[@]}" \
            || fail "clang-tidy (cuda benchmarks)"
    fi
else
    echo "== cuda: SKIPPED (no toolkit at $cuda_nvcc) =="
fi

# --- CUDA 12.x --------------------------------------------------------------
#
# The supported floor is CUDA 12.0 (docs/adr/0014 section 17). This stage
# builds and tests the same sources against a 12.x toolkit, so the floor is a
# measured claim rather than an inferred one. It runs the cuda12 presets only
# when the toolkit they pin really is 12.x and GCC 13 exists; any other machine
# skips it loudly. A driver older than the CUDA 13 runtime makes every GPU test
# in the stage above SKIP, so on such a machine this stage is the one that
# actually exercises the device.
cuda12_nvcc=/usr/bin/nvcc
if [ -x "$cuda12_nvcc" ] && "$cuda12_nvcc" --version | grep -q 'release 12\.' \
    && command -v g++-13 >/dev/null; then
    echo "== cuda12: toolkit =="
    "$cuda12_nvcc" --version | tail -n2

    echo "== cuda12-dev: configure/build/test =="
    cmake --preset cuda12-dev "${deps_cache_defines[@]}" || fail "configure cuda12-dev"
    cmake --build --preset cuda12-dev -j"$(nproc)" || fail "build cuda12-dev"
    ctest --preset cuda12-dev --output-on-failure || fail "test cuda12-dev"

    echo "== cuda12-release: configure/build/test =="
    cmake --preset cuda12-release "${deps_cache_defines[@]}" || fail "configure cuda12-release"
    cmake --build --preset cuda12-release -j"$(nproc)" || fail "build cuda12-release"
    ctest --preset cuda12-release --output-on-failure || fail "test cuda12-release"
else
    echo "== cuda12: SKIPPED (no CUDA 12.x toolkit at $cuda12_nvcc, or no g++-13) =="
fi

echo "verify.sh: all stages passed"
