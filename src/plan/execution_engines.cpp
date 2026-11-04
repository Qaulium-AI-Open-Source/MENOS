#include "plan/execution_engines.h"

#include "plan/available_engines.h"

#ifdef MENOS_ENABLE_CUDA
#include "engines/dense_cuda/cuda_dense_engine.h"
#endif

namespace menos::plan {

const std::vector<const SimulationEngine*>& execution_engines() {
    // Function-local static, constructed once and never mutated, exactly as
    // available_engines() is.
    static const std::vector<const SimulationEngine*> kEngines = [] {
        std::vector<const SimulationEngine*> engines = noisy_job_engines();
        engines.push_back(&trajectory_engine());
#ifdef MENOS_ENABLE_CUDA
        static const engines::dense_cuda::CudaDenseEngine kCudaDense;
        // Runtime-gated, not build-gated: a CUDA-enabled binary on a machine
        // with no GPU must behave exactly like a CUDA-less one.
        if (engines::dense_cuda::cuda_device_available()) {
            engines.push_back(&kCudaDense);
        }
#endif
        return engines;
    }();
    return kEngines;
}

const SimulationEngine* find_execution_engine(Engine engine, Backend backend, Tier tier) {
    for (const SimulationEngine* candidate : execution_engines()) {
        if (candidate->engine_id() == engine && candidate->backend() == backend &&
            candidate->tier() == tier) {
            return candidate;
        }
    }
    return nullptr;
}

}  // namespace menos::plan
