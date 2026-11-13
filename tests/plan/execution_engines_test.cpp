#include "plan/execution_engines.h"

#include <gtest/gtest.h>

#include <set>
#include <tuple>

// Phase 23 (docs/adr/0027): an implementation is identified by (engine,
// backend, tier). Every engine before Phase 23 is Tier::InMemory, and the
// lookup's default tier is InMemory, so every pre-Phase-23 caller resolves to
// the implementation it always did.

namespace menos::plan {
namespace {

TEST(ExecutionEnginesTest, EveryTripleResolvesToExactlyOneImplementation) {
    std::set<std::tuple<Engine, Backend, Tier>> seen;
    for (const SimulationEngine* engine : execution_engines()) {
        const auto key = std::make_tuple(engine->engine_id(), engine->backend(), engine->tier());
        EXPECT_TRUE(seen.insert(key).second) << "two implementations share a triple";
        EXPECT_EQ(find_execution_engine(engine->engine_id(), engine->backend(), engine->tier()),
                  engine);
    }
}

TEST(ExecutionEnginesTest, TheDefaultTierIsInMemory) {
    const SimulationEngine* dense = find_execution_engine(Engine::Dense, Backend::Cpu);
    ASSERT_NE(dense, nullptr);
    EXPECT_EQ(dense->tier(), Tier::InMemory);
    EXPECT_EQ(dense, find_execution_engine(Engine::Dense, Backend::Cpu, Tier::InMemory));
}

TEST(ExecutionEnginesTest, TheDiskTierIsADenseCpuImplementationOfItsOwn) {
    const SimulationEngine* disk = find_execution_engine(Engine::Dense, Backend::Cpu, Tier::Disk);
    ASSERT_NE(disk, nullptr);
    EXPECT_EQ(disk->tier(), Tier::Disk);
    EXPECT_EQ(disk->engine_id(), Engine::Dense);
    EXPECT_EQ(disk->backend(), Backend::Cpu);
    EXPECT_NE(disk, find_execution_engine(Engine::Dense, Backend::Cpu));
    EXPECT_FALSE(disk->capabilities().statevector);
}

TEST(ExecutionEnginesTest, ATierThisBuildHasNoEngineForIsNull) {
    // The host-offloaded tier was measured and not built (docs/adr/0027
    // section 10.1); no engine reports it. Only the dense algorithm has a
    // disk tier.
    EXPECT_EQ(find_execution_engine(Engine::Dense, Backend::Cuda, Tier::HostOffloaded), nullptr);
    EXPECT_EQ(find_execution_engine(Engine::Stabilizer, Backend::Cpu, Tier::Disk), nullptr);
}

}  // namespace
}  // namespace menos::plan
