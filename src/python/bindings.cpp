#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <complex>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <variant>

#include "common/error.h"
#include "common/splitmix64.h"
#include "menos/batch.h"
#include "menos/builder.h"
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


// Operations cross to Python as tuples (docs/adr/0029 section 3.7):
// ("gate", Opcode, qubits, params), ("unitary", qubits, matrix),
// ("measure", qubit, clbit, readout_error or None), ("channel", qubits, kraus).
// A builder gate's angle is a float when it is a constant and a
// ParameterExpression otherwise.
nb::object readout_to_py(const std::optional<menos::ReadoutError>& error) {
    if (!error.has_value()) {
        return nb::none();
    }
    return nb::cast(error->assignment);
}

nb::tuple operation_to_py(const menos::Operation& op) {
    return std::visit(
        [](const auto& o) -> nb::tuple {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, menos::GateOp>) {
                return nb::make_tuple("gate", o.opcode, o.qubits, o.params);
            } else if constexpr (std::is_same_v<T, menos::UnitaryOp>) {
                return nb::make_tuple("unitary", o.qubits, o.matrix);
            } else if constexpr (std::is_same_v<T, menos::MeasureOp>) {
                return nb::make_tuple("measure", o.qubit, o.clbit, readout_to_py(o.readout_error));
            } else {
                return nb::make_tuple("channel", o.qubits, o.kraus);
            }
        },
        op);
}

nb::tuple builder_operation_to_py(const menos::BuilderOperation& op) {
    return std::visit(
        [](const auto& o) -> nb::tuple {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, menos::BuilderGate>) {
                nb::list params;
                for (const menos::ParameterExpression& e : o.params) {
                    if (e.is_constant()) {
                        params.append(nb::cast(e.evaluate()));
                    } else {
                        params.append(nb::cast(e));
                    }
                }
                return nb::make_tuple("gate", o.opcode, o.qubits, params);
            } else {
                return operation_to_py(menos::Operation{o});
            }
        },
        op);
}

// bind()'s values: a dict keyed by parameter name or by Parameter.
std::map<std::string, double, std::less<>> values_from_py(const nb::dict& values) {
    std::map<std::string, double, std::less<>> out;
    for (const auto& [key, value] : values) {
        std::string name;
        if (nb::isinstance<menos::Parameter>(key)) {
            name = nb::cast<const menos::Parameter&>(key).name();
        } else if (nb::isinstance<nb::str>(key)) {
            name = nb::cast<std::string>(key);
        } else {
            throw nb::type_error("bind: keys must be str or Parameter");
        }
        out[name] = nb::cast<double>(value);
    }
    return out;
}

std::string expression_repr(const menos::ParameterExpression& e) {
    std::string out;
    for (const auto& term : e.terms()) {
        out += std::string(out.empty() ? "" : " + ") +
               nb::repr(nb::float_(term.coefficient)).c_str() + "*" + term.name;
    }
    if (e.constant().has_value()) {
        out += (out.empty() ? "" : " + ") +
               std::string(nb::repr(nb::float_(*e.constant())).c_str());
    }
    return "ParameterExpression(" + out + ")";
}

// Arithmetic on Parameter and ParameterExpression alike; the other operand
// may be a float, a Parameter or a ParameterExpression.
template <class Cls>
void add_arithmetic(Cls& cls) {
    using Self = typename Cls::Type;
    using PE = menos::ParameterExpression;
    cls.def("__add__", [](const Self& a, const PE& b) { return PE(a) + b; }, nb::is_operator())
        .def("__radd__", [](const Self& a, const PE& b) { return b + PE(a); }, nb::is_operator())
        .def("__sub__", [](const Self& a, const PE& b) { return PE(a) - b; }, nb::is_operator())
        .def("__rsub__", [](const Self& a, const PE& b) { return b - PE(a); }, nb::is_operator())
        .def("__mul__", [](const Self& a, const PE& b) { return PE(a) * b; }, nb::is_operator())
        .def("__rmul__", [](const Self& a, const PE& b) { return b * PE(a); }, nb::is_operator())
        .def("__neg__", [](const Self& a) { return -PE(a); });
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
        .def_rw("global_phase", &menos::Circuit::global_phase)
        // Read-only, as tuples (docs/adr/0029 section 3.7).
        .def("operations", [](const menos::Circuit& circuit) {
            nb::list out;
            for (const menos::Operation& op : circuit.operations()) {
                out.append(operation_to_py(op));
            }
            return out;
        });

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

    // Phase 25 (docs/adr/0029): the circuit builder, a binding of
    // include/menos/builder.h. Every gate method returns the builder itself.
    nb::class_<menos::Parameter> parameter_cls(m, "Parameter");
    parameter_cls.def(nb::init<std::string>(), "name"_a)
        .def_prop_ro("name", &menos::Parameter::name)
        .def("__repr__",
             [](const menos::Parameter& p) { return "Parameter('" + p.name() + "')"; });

    nb::class_<menos::ParameterExpression> expression_cls(m, "ParameterExpression");
    expression_cls.def(nb::init<double>(), "constant"_a)
        .def(nb::init<const menos::Parameter&>(), "parameter"_a)
        .def_prop_ro("terms",
                     [](const menos::ParameterExpression& e) {
                         nb::list out;
                         for (const auto& term : e.terms()) {
                             out.append(nb::make_tuple(term.name, term.coefficient));
                         }
                         return out;
                     })
        .def_prop_ro("constant", &menos::ParameterExpression::constant)
        .def("is_constant", &menos::ParameterExpression::is_constant)
        .def(
            "evaluate",
            [](const menos::ParameterExpression& e, const nb::dict& values) {
                return e.evaluate(values_from_py(values));
            },
            "values"_a = nb::dict())
        .def("__repr__", &expression_repr);
    nb::implicitly_convertible<double, menos::ParameterExpression>();
    nb::implicitly_convertible<menos::Parameter, menos::ParameterExpression>();
    add_arithmetic(parameter_cls);
    add_arithmetic(expression_cls);

    nb::class_<menos::QuantumRegister>(m, "QuantumRegister")
        .def_ro("name", &menos::QuantumRegister::name)
        .def_ro("offset", &menos::QuantumRegister::offset)
        .def_ro("size", &menos::QuantumRegister::size)
        .def("__len__", [](const menos::QuantumRegister& r) { return r.size; })
        .def("__getitem__", &menos::QuantumRegister::operator[], "i"_a)
        .def("__repr__", [](const menos::QuantumRegister& r) {
            return "QuantumRegister('" + r.name + "', " + std::to_string(r.size) + ")";
        });
    nb::class_<menos::ClassicalRegister>(m, "ClassicalRegister")
        .def_ro("name", &menos::ClassicalRegister::name)
        .def_ro("offset", &menos::ClassicalRegister::offset)
        .def_ro("size", &menos::ClassicalRegister::size)
        .def("__len__", [](const menos::ClassicalRegister& r) { return r.size; })
        .def("__getitem__", &menos::ClassicalRegister::operator[], "i"_a)
        .def("__repr__", [](const menos::ClassicalRegister& r) {
            return "ClassicalRegister('" + r.name + "', " + std::to_string(r.size) + ")";
        });

    using CB = menos::CircuitBuilder;
    constexpr auto self = nb::rv_policy::reference_internal;
    nb::class_<CB>(m, "CircuitBuilder")
        .def(nb::init<std::uint32_t, std::uint32_t>(), "num_qubits"_a = 0, "num_clbits"_a = 0)
        .def_static("from_circuit", &CB::from_circuit, "circuit"_a)
        .def("add_qreg", &CB::add_qreg, "name"_a, "size"_a)
        .def("add_creg", &CB::add_creg, "name"_a, "size"_a)
        .def("qreg", &CB::qreg, "name"_a)
        .def("creg", &CB::creg, "name"_a)
        .def("qregs", &CB::qregs)
        .def("cregs", &CB::cregs)
        .def("num_qubits", &CB::num_qubits)
        .def("num_clbits", &CB::num_clbits)
        .def("x", &CB::x, "qubit"_a, self)
        .def("y", &CB::y, "qubit"_a, self)
        .def("z", &CB::z, "qubit"_a, self)
        .def("h", &CB::h, "qubit"_a, self)
        .def("s", &CB::s, "qubit"_a, self)
        .def("sdg", &CB::sdg, "qubit"_a, self)
        .def("t", &CB::t, "qubit"_a, self)
        .def("tdg", &CB::tdg, "qubit"_a, self)
        .def("sx", &CB::sx, "qubit"_a, self)
        .def("sxdg", &CB::sxdg, "qubit"_a, self)
        .def("rx", &CB::rx, "theta"_a, "qubit"_a, self)
        .def("ry", &CB::ry, "theta"_a, "qubit"_a, self)
        .def("rz", &CB::rz, "phi"_a, "qubit"_a, self)
        .def("p", &CB::p, "theta"_a, "qubit"_a, self)
        .def("u", &CB::u, "theta"_a, "phi"_a, "lam"_a, "qubit"_a, self)
        .def("cx", &CB::cx, "control_qubit"_a, "target_qubit"_a, self)
        .def("cz", &CB::cz, "control_qubit"_a, "target_qubit"_a, self)
        .def("swap", &CB::swap, "qubit1"_a, "qubit2"_a, self)
        .def("cp", &CB::cp, "theta"_a, "control_qubit"_a, "target_qubit"_a, self)
        .def("rzz", &CB::rzz, "theta"_a, "qubit1"_a, "qubit2"_a, self)
        .def("ccx", &CB::ccx, "control_qubit1"_a, "control_qubit2"_a, "target_qubit"_a, self)
        .def("unitary", &CB::unitary, "matrix"_a, "qubits"_a, self)
        .def(
            "measure",
            [](CB& b,
               menos::QubitId qubit,
               menos::ClbitId clbit,
               std::optional<std::array<double, 4>> readout_error) -> CB& {
                std::optional<menos::ReadoutError> error;
                if (readout_error.has_value()) {
                    error = menos::ReadoutError{*readout_error};
                }
                return b.measure(qubit, clbit, error);
            },
            "qubit"_a,
            "clbit"_a,
            "readout_error"_a = nb::none(),
            self)
        .def("measure_all", &CB::measure_all, self)
        .def("channel", &CB::channel, "kraus"_a, "qubits"_a, self)
        .def("compose",
             &CB::compose,
             "other"_a,
             "qubits"_a = std::vector<menos::QubitId>{},
             "clbits"_a = std::vector<menos::ClbitId>{},
             self)
        .def("inverse", &CB::inverse)
        .def("repeat", &CB::repeat, "reps"_a)
        .def("parameters", &CB::parameters)
        .def("operations",
             [](const CB& b) {
                 nb::list out;
                 for (const menos::BuilderOperation& op : b.operations()) {
                     out.append(builder_operation_to_py(op));
                 }
                 return out;
             })
        .def(
            "bind",
            [](const CB& b, const nb::dict& values) { return b.bind(values_from_py(values)); },
            "values"_a)
        .def("to_circuit", &CB::to_circuit)
        .def("to_qasm3", [](const CB& b) { return menos::qasm3::emit(b); })
        .def_rw("global_phase", &CB::global_phase);

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

    // Phase 23 (docs/adr/0027): where a job's state lives while it runs, for
    // RunOptions.tier. Tier.HostOffloaded was measured and not built, and a
    // request for it raises UnsupportedFeatureError.
    nb::enum_<menos::Tier>(m, "Tier")
        .value("InMemory", menos::Tier::InMemory)
        .value("Disk", menos::Tier::Disk)
        .value("HostOffloaded", menos::Tier::HostOffloaded);

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
        // Phase 24 item 7 (docs/adr/0028 section 17): gate fusion by
        // containment in the dense engines, on by default; False applies
        // every gate on its own as before.
        .def_rw("fuse_gates", &menos::RunOptions::fuse_gates)
        // Phase 17 (docs/adr/0021 section 2): None = the planner chooses (the
        // default); Backend.Cpu or Backend.Cuda forces one, and a request that
        // cannot be honoured raises rather than falling back.
        .def_rw("backend", &menos::RunOptions::backend)
        // Phase 23 (docs/adr/0027): None = the planner chooses (the disk tier
        // only when the in-memory state does not fit); Tier.Disk forces it and
        // never falls back. scratch_directory is where the disk tier's file
        // goes; None = the system's temporary directory.
        .def_rw("tier", &menos::RunOptions::tier)
        .def_rw("scratch_directory", &menos::RunOptions::scratch_directory);

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
        // Phase 21 (docs/adr/0025 section 3.2): the most host and CUDA device
        // memory the execution held, measured by the runtime's memory ledger.
        .def_prop_ro("peak_host_bytes", &menos::Result::peak_host_bytes)
        .def_prop_ro("peak_device_bytes", &menos::Result::peak_device_bytes)
        // Phase 23 (docs/adr/0027): the disk tier's scratch file; 0 for a
        // plan with no disk-tier region.
        .def_prop_ro("peak_disk_bytes", &menos::Result::peak_disk_bytes)
        .def_prop_ro("execute_seconds", &menos::Result::execute_seconds)
        .def_prop_ro("sample_seconds", &menos::Result::sample_seconds)
        // Phase 24 (docs/adr/0028 section 16.7 Q3): the initial plan's
        // planning time, and the replans' planning time summed.
        .def_prop_ro("plan_seconds", &menos::Result::plan_seconds)
        .def_prop_ro("replan_seconds", &menos::Result::replan_seconds)
        // Phase 17 (docs/adr/0021 section 3): the abandoned attempts of a job
        // that replanned, one dict per ReplanEvent, in order. Empty for a job
        // that never replanned. Engines and backends are spelled as explain()
        // spells them. The superseded plan itself is in explain(), not here.
        .def_prop_ro(
            "replans",
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
                    entry["backend"] = std::string(menos::plan::backend_name(event.backend));
                    entry["reason"] = event.reason;
                    entry["plan_seconds"] = event.plan_seconds;
                    nb::list regions;
                    for (const menos::RegionObservation& region : event.regions) {
                        nb::dict observed;
                        observed["qubits"] = region.qubits;
                        observed["engine"] = std::string(menos::plan::engine_name(region.engine));
                        observed["backend"] =
                            std::string(menos::plan::backend_name(region.backend));
                        observed["tier"] = std::string(menos::plan::tier_name(region.tier));
                        observed["estimated_seconds"] = region.estimated_seconds;
                        observed["observed_seconds"] = region.observed_seconds;
                        observed["completed"] = region.completed;
                        regions.append(observed);
                    }
                    entry["regions"] = regions;
                    events.append(entry);
                }
                return events;
            })
        .def("explain", &menos::Result::explain);

    m.def("run",
          nb::overload_cast<const menos::Circuit&, const menos::RunOptions&>(&menos::run),
          "circuit"_a,
          "options"_a);
    // Phase 25: a builder with no parameters (InvalidCircuitError naming them
    // otherwise).
    m.def("run",
          nb::overload_cast<const menos::CircuitBuilder&, const menos::RunOptions&>(&menos::run),
          "circuit"_a,
          "options"_a);
    m.def("plan_cache_stats", []() {
        const menos::PlanCacheStats s = menos::plan_cache_stats();
        nb::dict d;
        d["hits"] = s.hits;
        d["misses"] = s.misses;
        d["bypasses"] = s.bypasses;
        return d;
    });
    m.def("plan_cache_clear", &menos::plan_cache_clear);

    // Phase 22 (docs/adr/0026): give the device memory MENOS retains between
    // jobs back to the driver. A no-op without CUDA.
    m.def("release_device_memory", &menos::release_device_memory);

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
