#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "menos/circuit.h"

namespace menos::ir {

// Diagonal-gate fusion (opt-in: RunOptions::fuse_diagonal_gates).
//
// Every diagonal gate commutes with every other, so a run of consecutive
// diagonal operations (a QAOA cost layer: RZ and RZZ on every pair) can be
// applied in any order and in any grouping. This pass groups each such run
// into blocks of at most max_qubits distinct qubits and replaces each block
// of two or more gates by ONE UnitaryOp carrying the block's fused diagonal
// (a 2^k x 2^k matrix whose off-diagonal entries are exactly zero, so the
// dense kernels take their diagonal fast path: one complex multiply per
// amplitude per block instead of one per gate). A block of one gate is left
// as it was; a non-diagonal operation, or a measurement, ends the run.
//
// Diagonal by construction: GateOp with opcode Z, S, Sdg, T, Tdg, RZ, P, CZ,
// CP or RZZ; UnitaryOp whose off-diagonal entries are all exactly zero.
//
// Exact up to rounding: the fused diagonal entry is the ordinary product of
// the gates' own diagonal entries, so the statevector after the block equals
// the statevector after the gates within a few ulp -- it is NOT bitwise
// identical to applying the gates one by one, which is why this is opt-in
// and why the CUDA engine (which applies the original operations) is not
// bitwise identical to a fused CPU run (docs/adr/0014 section 10 holds with
// fusion off).
//
// Grouping is greedy in program order: a block opens at the first unassigned
// gate of the run and takes every later unassigned gate of the run whose
// operands fit within max_qubits qubits alongside it. Blocks are emitted in
// the order they open. Precondition: max_qubits >= 2 and <= kMaxUnitaryQubits.
[[nodiscard]] std::vector<Operation> fuse_diagonal_gates(std::span<const Operation> operations,
                                                         std::uint32_t max_qubits);

// True iff op is diagonal by the rule above.
[[nodiscard]] bool is_diagonal_operation(const Operation& op);

}  // namespace menos::ir
