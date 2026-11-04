"""MENOS: a quantum circuit simulation runtime.

This top-level package exposes the thin C++ core surface
(``menos._core``) directly. See ``menos.qiskit`` for the Qiskit
``BackendV2`` adapter.
"""

from menos import _core
from menos._core import (
    Backend,
    Circuit,
    MitigatedResult,
    NoiseMethod,
    Opcode,
    ReadoutMitigationOptions,
    Result,
    RunOptions,
    Tier,
    readout_calibration_circuits,
    release_device_memory,
    run,
    run_batch,
    run_readout_mitigated,
)
from menos.exceptions import (
    InternalError,
    InvalidCircuitError,
    MenosError,
    ResourceExceededError,
    UnsupportedFeatureError,
)

__all__ = [
    "Backend",
    "Circuit",
    "NoiseMethod",
    "Opcode",
    "Result",
    "RunOptions",
    "Tier",
    "run",
    "run_batch",
    "ReadoutMitigationOptions",
    "MitigatedResult",
    "readout_calibration_circuits",
    "run_readout_mitigated",
    "from_qasm3",
    "release_device_memory",
    "MenosError",
    "InvalidCircuitError",
    "UnsupportedFeatureError",
    "ResourceExceededError",
    "InternalError",
]



def from_qasm3(text: str, inputs: dict[str, float] | None = None) -> Circuit:
    """Parse an OpenQASM 3 program into a Circuit (docs/adr/0023).

    ``inputs`` binds the program's ``input`` declarations by name. Every
    ``input`` needs a binding and every binding must name an ``input``.
    Constructs outside the supported static subset raise
    UnsupportedFeatureError naming the construct, line and column; a
    malformed program raises InvalidCircuitError.
    """
    bindings = {} if inputs is None else {str(k): float(v) for k, v in inputs.items()}
    return _core.from_qasm3(text, bindings)


__version__ = "1.2.1"
