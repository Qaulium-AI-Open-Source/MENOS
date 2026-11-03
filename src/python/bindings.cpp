#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <complex>
#include <cstring>
#include <map>

#include "common/error.h"
#include "common/splitmix64.h"
#include "menos/batch.h"
#include "menos/bitstring.h"
#include "menos/menos.h"
#include "menos/mitigation.h"
#include "menos/observable.h"
#include "menos/qasm3.h"
#include "plan/enum_names.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// A freshly-allocated, owned NumPy array -- a safe copy of Result's
// statevector, not a view into its (temporary) storage. See
// docs/adr/0004-qiskit-backendv2-and-python-boundary.md: "Statevector is a
// copy, not a view."
nb::object statevector_to_numpy(
    const std::optional<std::vector<std::complex<double>>>& statevector) {
    if (!statevector.has_value()) {
        return nb::none();
    }
    const std::size_t n = statevector->size();
    auto* data = new std::complex<double>[n];
    std::memcpy(data, statevector->data(), n * sizeof(std::complex<double>));
    nb::capsule owner(data,
                      [](void* p) noexcept { delete[] static_cast<std::complex<double>*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>(data, {n}, owner));
}

// menos::BitString is not exposed as a Python class (no Python-side use
// for it yet -- CLAUDE.md's "no speculative code"); counts keys cross the
// boundary as Python str, already hex-formatted here, so the Qiskit
// adapter (python/menos/qiskit/result_adapter.py) never needs to know
// BitString exists. BitString::to_hex_string() is byte-identical to
// Python's own hex() on the same represented integer (docs/adr/0005),
// which is exactly what result_adapter.py's counts dict needs.
nb::dict counts_to_hex_dict(const std::map<menos::BitString, std::uint64_t>& counts) {
    nb::dict out;
    for (const auto& [key, value] : counts) {
        out[nb::str(key.to_hex_string().c_str())] = value;
    }
    return out;
}

// A dense mitigation vector as a dict keyed like counts: the hex string of
// each outcome's full classical bit string (docs/adr/0018 section 5).
nb::dict mitigated_to_hex_dict(const menos::MitigatedResult& result,
                               const std::vector<double>& values) {
    nb::dict out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[nb::str(result.outcome(i).to_hex_string().c_str())] = values[i];
    }
    return out;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "MENOS C++ core bindings. Internal; use menos.qiskit.";

    m.attr("kMaxUnitaryQubits") = menos::kMaxUnitaryQubits;

    nb::class_<menos::Circuit>(m, "Circuit")
        .def(nb::init<std::uint32_t, std::uint32_t>(), "num_qubits"_a, "num_clbits"_a)
        .def("append",
             &menos::Circuit::append,
             "opcode"_a,
             "qubits"_a,
             "params"_a = std::vector<double>{})
        .def("append_unitary", &menos::Circuit::append_unitary, "qubits"_a, "matrix"_a)
        // readout_error is P(recorded | true) for this measurement, row-major by
        // true outcome: [P(0|0), P(1|0), P(0|1), P(1|1)] -- Qiskit Aer's
        // ReadoutError.probabilities flattened row by row (Phase 12,
        // docs/adr/0016 section 3.2).
        .def(
            "append_measure",
            [](menos::Circuit& circuit,
               menos::QubitId qubit,
               menos::ClbitId clbit,
               std::optional<std::array<double, 4>> readout_error) {
                std::optional<menos::ReadoutError> error;
                if (readout_error.has_value()) {
                    error = menos::ReadoutError{*readout_error};
                }
                circuit.append_measure(qubit, clbit, error);
            },
            "qubit"_a,
            "clbit"_a,
            "readout_error"_a = nb::none())
        // A noise channel by its Kraus operators, each a row-major 2^k x 2^k
        // matrix flattened to 4^k entries (Phase 12, docs/adr/0016 section 3.1).
        // Validated at run(), like every operation.
        .def("append_channel", &menos::Circuit::append_channel, "qubits"_a, "kraus"_a)
        .def("append_batch",
             &menos::Circuit::append_batch,
             "opcodes"_a,
             "qubits"_a,
             "qubit_offsets"_a,
             "params"_a,
             "param_offsets"_a)
        .def("num_qubits", &menos::Circuit::num_qubits)
        .def("num_clbits", &menos::Circuit::num_clbits)
        .def_rw("global_phase", &menos::Circuit::global_phase);

    nb::enum_<menos::Opcode>(m, "Opcode")
        .value("X", menos::Opcode::X)
        .value("Y", menos::Opcode::Y)
        .value("Z", menos::Opcode::Z)
        .value("H", menos::Opcode::H)
        .value("S", menos::Opcode::S)
        .value("Sdg", menos::Opcode::Sdg)
        .value("T", menos::Opcode::T)
        .value("Tdg", menos::Opcode::Tdg)
        .value("SX", menos::Opcode::SX)
        .value("SXdg", menos::Opcode::SXdg)
        .value("RX", menos::Opcode::RX)
        .value("RY", menos::Opcode::RY)
        .value("RZ", menos::Opcode::RZ)
        .value("P", menos::Opcode::P)
        .value("U", menos::Opcode::U)
        .value("CX", menos::Opcode::CX)
        .value("CZ", menos::Opcode::CZ)
        .value("SWAP", menos::Opcode::SWAP)
        .value("CP", menos::Opcode::CP)
        .value("RZZ", menos::Opcode::RZZ)
        .value("CCX", menos::Opcode::CCX);
    // Unitary and Measure are deliberately not exposed here: Python reaches
    // them only via Circuit.append_unitary/append_measure, matching the
    // C++ API's own append surface (append_channel added in Phase 12).

    // PauliString/Observable: fields only, no convenience constructors --
    // the bit-packing logic for building one from a qubit list already
    // lives in Python for convert.py's gate table, so it belongs in
    // Python here too (menos.qiskit._observable_convert), not as a new
    // C++ API this phase's spec doesn't otherwise need
    // (docs/adr/0008).
    nb::class_<menos::PauliString>(m, "PauliString")
        .def(nb::init<>())
        .def_rw("num_qubits", &menos::PauliString::num_qubits)
        .def_rw("x_words", &menos::PauliString::x_words)
        .def_rw("z_words", &menos::PauliString::z_words);

    nb::class_<menos::Observable>(m, "Observable")
        .def(nb::init<>())
        .def_rw("terms", &menos::Observable::terms);

    // Phase 13 (docs/adr/0017): how a noisy job may be simulated.
    nb::enum_<menos::NoiseMethod>(m, "NoiseMethod")
        .value("Exact", menos::NoiseMethod::Exact)
        .value("Auto", menos::NoiseMethod::Auto);

    // Phase 17 (docs/adr/0021 section 2): the machine a job runs on, for
    // RunOptions.backend. Python spells them Backend.Cpu and Backend.Cuda.
    nb::enum_<menos::Backend>(m, "Backend")
        .value("Cpu", menos::Backend::Cpu)
        .value("Cuda", menos::Backend::Cuda);

    nb::class_<menos::RunOptions>(m, "RunOptions")
        .def(nb::init<>())
        .def_rw("shots", &menos::RunOptions::shots)
        .def_rw("want_statevector", &menos::RunOptions::want_statevector)
        .def_rw("seed", &menos::RunOptions::seed)
        .def_rw("memory_budget_bytes", &menos::RunOptions::memory_budget_bytes)
        .def_rw("expectation", &menos::RunOptions::expectation)
        // Exposed in Phase 11 (docs/adr/0015), reversing the Phase 10 deferral
        // recorded in docs/adr/0014 section 16.1. That deferral rested on there
        // being no Python caller for it; Phase 11 makes the budget the trigger
        // for runtime monitoring and replanning, so keeping it C++-only would
        // have put the whole feature out of reach of the primary entry point.
        //
        // From Phase 17 (docs/adr/0021 section 3) the Qiskit BackendV2 adapter
        // sets it too, from its time_budget_seconds run option; MENOSEstimator
        // and MENOSSampler still do not (owner Q3). A run whose observed time
        // exceeds this raises ResourceExceededError when no replan meets it,
        // which the existing exception translator below already maps.
        //
        // The budget bounds ONE execution attempt, not the total wall time of a
        // job that replans: an abandoned-and-restarted job can take several times
        // this figure in total. See RunOptions::time_budget_seconds in
        // include/menos/menos.h and docs/adr/0015 section 2.4.
        .def_rw("time_budget_seconds", &menos::RunOptions::time_budget_seconds)
        // Phase 13 opt-ins for trajectory sampling (docs/adr/0017 section 9).
        // Both defaults keep a noisy job exact on the density matrix.
        .def_rw("noise_method", &menos::RunOptions::noise_method)
        .def_rw("expectation_precision", &menos::RunOptions::expectation_precision)
        // Phase 15 (docs/adr/0019 section 3.3): a cap on CPU threads for gate
        // application. None = every hardware thread; 0 raises InvalidCircuitError.
        // Results do not depend on it. The Qiskit adapters do not set it.
        .def_rw("max_threads", &menos::RunOptions::max_threads)
        .def_rw("fuse_diagonal_gates", &menos::RunOptions::fuse_diagonal_gates)
        // Phase 17 (docs/adr/0021 section 2): None = the planner chooses (the
        // default); Backend.Cpu or Backend.Cuda forces one, and a request that
        // cannot be honoured raises rather than falling back.
        .def_rw("backend", &menos::RunOptions::backend);

    nb::class_<menos::Result>(m, "Result")
        .def_prop_ro("counts",
                     [](const menos::Result& r) { return counts_to_hex_dict(r.counts()); })
        .def_prop_ro("statevector",
                     [](const menos::Result& r) { return statevector_to_numpy(r.statevector()); })
        .def_prop_ro("expectation_value", &menos::Result::expectation_value)
        // None for every exact result; set only for a statistical estimate.
        .def_prop_ro("expectation_standard_error", &menos::Result::expectation_standard_error)
        .def_prop_ro("exact", &menos::Result::exact)
        .def_prop_ro("actual_state_bytes", &menos::Result::actual_state_bytes)
        .def_prop_ro("execute_seconds", &menos::Result::execute_seconds)
        .def_prop_ro("sample_seconds", &menos::Result::sample_seconds)
        // Phase 17 (docs/adr/0021 section 3): the abandoned attempts of a job
        // that replanned, one dict per ReplanEvent, in order. Empty for a job
        // that never replanned. Engines and backends are spelled as explain()
        // spells them. The superseded plan itself is in explain(), not here.
        .def_prop_ro("replans",
                     [](const menos::Result& r) {
                         nb::list events;
                         for (const menos::ReplanEvent& event : r.replans()) {
                             nb::dict entry;
                             entry["sequence"] = event.sequence;
                             entry["threshold_seconds"] = event.threshold_seconds;
                             entry["observed_seconds"] = event.observed_seconds;
                             entry["predicted_seconds"] = event.predicted_seconds;
                             entry["boundaries_crossed"] = event.boundaries_crossed;
                             entry["engine"] = std::string(menos::plan::engine_name(event.engine));
                             entry["backend"] =
                                 std::string(menos::plan::backend_name(event.backend));
                             entry["reason"] = event.reason;
                             events.append(entry);
                         }
                         return events;
                     })
        .def("explain", &menos::Result::explain);

    m.def("run", &menos::run, "circuit"_a, "options"_a);
    m.def("plan_cache_stats", []() {
        const menos::PlanCacheStats s = menos::plan_cache_stats();
        nb::dict d;
        d["hits"] = s.hits;
        d["misses"] = s.misses;
        d["bypasses"] = s.bypasses;
        return d;
    });
    m.def("plan_cache_clear", &menos::plan_cache_clear);

    // Phase 19 (docs/adr/0023): OpenQASM 3 import. menos.from_qasm3 wraps it so
    // that inputs may be None.
    m.def(
        "from_qasm3",
        [](const std::string& text, const std::map<std::string, double>& inputs) {
            return menos::qasm3::parse(text,
                                       menos::qasm3::InputBindings(inputs.begin(), inputs.end()));
        },
        "text"_a,
        "inputs"_a);

    // Phase 14 (docs/adr/0018): batch execution and readout mitigation.
    m.def(
        "run_batch",
        [](const std::vector<menos::Circuit>& circuits, const menos::RunOptions& options) {
            return menos::run_batch(circuits, options);
        },
        "circuits"_a,
        "options"_a);

    // The first `count` SplitMix64 outputs of `seed`: exactly the per-circuit
    // seeds run_batch derives. MENOSSampler gives pub j the j-th, because its
    // pubs can have different shot counts, which one run_batch call (one
    // RunOptions) cannot express.
    m.def(
        "derive_seeds",
        [](std::uint64_t seed, std::size_t count) {
            menos::common::SplitMix64 mixer(seed);
            std::vector<std::uint64_t> seeds(count);
            for (std::uint64_t& s : seeds) {
                s = mixer.next();
            }
            return seeds;
        },
        "seed"_a,
        "count"_a);

    nb::class_<menos::ReadoutMitigationOptions>(m, "ReadoutMitigationOptions")
        .def(nb::init<>())
        .def_rw("calibration_shots", &menos::ReadoutMitigationOptions::calibration_shots);

    nb::class_<menos::MitigatedResult>(m, "MitigatedResult")
        .def_prop_ro("quasi_probabilities",
                     [](const menos::MitigatedResult& r) {
                         return mitigated_to_hex_dict(r, r.quasi_probabilities());
                     })
        .def_prop_ro("standard_errors",
                     [](const menos::MitigatedResult& r) {
                         return mitigated_to_hex_dict(r, r.standard_errors());
                     })
        .def_prop_ro("target_standard_errors",
                     [](const menos::MitigatedResult& r) {
                         return mitigated_to_hex_dict(r, r.target_standard_errors());
                     })
        .def_prop_ro("measured_clbits", &menos::MitigatedResult::measured_clbits)
        .def_prop_ro("gamma", &menos::MitigatedResult::gamma)
        // (clbit, P(1|0), P(0|1)) per measured clbit.
        .def_prop_ro("calibration",
                     [](const menos::MitigatedResult& r) {
                         nb::list out;
                         for (const menos::ClbitCalibration& c : r.calibration()) {
                             out.append(nb::make_tuple(c.clbit, c.p1_given_0, c.p0_given_1));
                         }
                         return out;
                     })
        .def_prop_ro("calibration_shots", &menos::MitigatedResult::calibration_shots)
        .def_prop_ro("raw", &menos::MitigatedResult::raw, nb::rv_policy::reference_internal)
        .def_prop_ro("calibration_results",
                     &menos::MitigatedResult::calibration_results,
                     nb::rv_policy::reference_internal)
        .def_prop_ro("exact", &menos::MitigatedResult::exact)
        .def("explain", &menos::MitigatedResult::explain);

    m.def(
        "readout_calibration_circuits",
        [](const menos::Circuit& target) {
            auto pair = menos::readout_calibration_circuits(target);
            return nb::make_tuple(std::move(pair[0]), std::move(pair[1]));
        },
        "target"_a);
    // calibration is None or a (zero, one) pair of Circuits. Taken as an object:
    // nanobind's std::array caster needs a default-constructible element type,
    // which Circuit is not.
    m.def(
        "run_readout_mitigated",
        [](const menos::Circuit& target,
           const nb::object& calibration,
           const menos::RunOptions& options,
           const menos::ReadoutMitigationOptions& mitigation) {
            std::optional<std::array<menos::Circuit, 2>> pair;
            if (!calibration.is_none()) {
                if (nb::len(calibration) != 2) {
                    throw nb::value_error("calibration must be None or a (zero, one) pair");
                }
                pair = std::array<menos::Circuit, 2>{nb::cast<menos::Circuit>(calibration[0]),
                                                     nb::cast<menos::Circuit>(calibration[1])};
            }
            return menos::run_readout_mitigated(target, pair, options, mitigation);
        },
        "target"_a,
        "calibration"_a.none(),
        "options"_a,
        "mitigation"_a);

    // Exception translators are tried most-recently-registered-first
    // (nanobind/src/nb_internals.h: "append-to-front-only singly linked
    // list traversed while raising exceptions", verified against installed
    // nanobind 3.1.0 source). Registering the base Error translator FIRST
    // and each subclass AFTER means a subclass instance is matched by its
    // own (more specific) translator before it ever reaches the base
    // catch(Error&) -- reversing this order would collapse every C++
    // exception into the generic MenosError.
    static nb::exception<menos::common::Error> menos_error(m, "MenosError");
    static nb::exception<menos::common::InvalidCircuit> invalid_circuit(
        m, "InvalidCircuitError", menos_error.ptr());
    static nb::exception<menos::common::UnsupportedFeature> unsupported_feature(
        m, "UnsupportedFeatureError", menos_error.ptr());
    static nb::exception<menos::common::ResourceExceeded> resource_exceeded(
        m, "ResourceExceededError", menos_error.ptr());
    static nb::exception<menos::common::InternalError> internal_error(
        m, "InternalError", menos_error.ptr());
}
