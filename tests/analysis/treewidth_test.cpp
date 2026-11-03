// docs/BUILD_PLAN.md Phase 8 acceptance criterion 2: "The retained and
// extended treewidth-exactness suite passes." Retained verbatim from
// the "Phase 8 precondition, step 2" analysis tool (docs/adr/0010,
// benchmarks/treewidth_estimate.cpp, now productionized into
// src/analysis/treewidth.{h,cpp}): known-exact-treewidth graphs from
// graph theory. Extended with graphs shaped like real KIR circuits'
// own tensor-network structure (not only abstract graphs), and a
// constructed elimination-heuristic tie case for acceptance criterion 3
// (determinism).

#include "analysis/treewidth.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace menos::analysis {
namespace {

TensorNetworkGraph make_path_graph(std::size_t n) {
    TensorNetworkGraph g{n, {}};
    for (std::size_t i = 0; i + 1 < n; ++i) {
        g.edges.emplace_back(i, i + 1);
    }
    return g;
}

TensorNetworkGraph make_cycle_graph(std::size_t n) {
    TensorNetworkGraph g = make_path_graph(n);
    if (n >= 3) {
        g.edges.emplace_back(0, n - 1);
    }
    return g;
}

TensorNetworkGraph make_complete_graph(std::size_t n) {
    TensorNetworkGraph g{n, {}};
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            g.edges.emplace_back(i, j);
        }
    }
    return g;
}

// An m x m grid graph, row-major, 4-connectivity. Treewidth of an m x m
// grid is exactly m (a standard, documented graph-theory result).
TensorNetworkGraph make_grid_graph(std::size_t m) {
    TensorNetworkGraph g{m * m, {}};
    const auto flat = [&](std::size_t r, std::size_t c) { return r * m + c; };
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < m; ++c) {
            if (c + 1 < m) {
                g.edges.emplace_back(flat(r, c), flat(r, c + 1));
            }
            if (r + 1 < m) {
                g.edges.emplace_back(flat(r, c), flat(r + 1, c));
            }
        }
    }
    return g;
}

TEST(TreewidthTest, PathGraphTreewidthIsOne) {
    EXPECT_EQ(estimate_treewidth_min_fill(make_path_graph(10)).width, 1U);
}

TEST(TreewidthTest, CycleGraphTreewidthIsTwo) {
    EXPECT_EQ(estimate_treewidth_min_fill(make_cycle_graph(10)).width, 2U);
}

TEST(TreewidthTest, CompleteGraphK6TreewidthIsFive) {
    EXPECT_EQ(estimate_treewidth_min_fill(make_complete_graph(6)).width, 5U);
}

TEST(TreewidthTest, ThreeByThreeGridTreewidthIsThree) {
    EXPECT_EQ(estimate_treewidth_min_fill(make_grid_graph(3)).width, 3U);
}

TEST(TreewidthTest, FourByFourGridTreewidthIsFour) {
    EXPECT_EQ(estimate_treewidth_min_fill(make_grid_graph(4)).width, 4U);
}

// docs/BUILD_PLAN.md Phase 8 acceptance criterion 3: determinism,
// specifically "a constructed case with a genuine tie in the
// elimination heuristic confirms the fixed tie-break rule is
// exercised, not merely assumed untested." Three disjoint edges: every
// vertex starts at degree 1, so every candidate has fill=0 at every
// step -- a genuine, maximal tie at each of the 6 elimination steps,
// not merely at the first one.
TEST(TreewidthTest, ConstructedTieCaseResolvesToLowestIndexDeterministically) {
    TensorNetworkGraph g{6, {{0, 1}, {2, 3}, {4, 5}}};

    const EliminationOrder first = estimate_treewidth_min_fill(g);
    const EliminationOrder second = estimate_treewidth_min_fill(g);

    // The documented tie-break rule (lowest node index) makes this
    // exact order the only correct one, not merely "some fixed order":
    // every step ties across every remaining vertex, so the lowest
    // remaining index wins every time.
    const std::vector<std::size_t> expected_order = {0, 1, 2, 3, 4, 5};
    EXPECT_EQ(first.order, expected_order);
    EXPECT_EQ(first.width, 1U);  // three disjoint edges: treewidth 1

    // Each disjoint edge's first-eliminated endpoint has degree 1 (its
    // partner is still present); its second-eliminated endpoint has
    // degree 0 (its only neighbor is already gone) -- hand-derived,
    // matching the tensor-network engine's own per-step cost accounting
    // (src/analysis/treewidth.h's own EliminationOrder::per_step_degree
    // doc comment).
    const std::vector<std::size_t> expected_degrees = {1, 0, 1, 0, 1, 0};
    EXPECT_EQ(first.per_step_degree, expected_degrees);

    // Repeated calls on the identical graph give the identical order --
    // CLAUDE.md's determinism domain invariant applied to the
    // elimination heuristic specifically (never iteration or hash
    // order).
    EXPECT_EQ(first.order, second.order);
    EXPECT_EQ(first.width, second.width);
    EXPECT_EQ(first.per_step_degree, second.per_step_degree);
}

// Real-circuit-shaped graphs, not only abstract ones: a 2-qubit Bell
// circuit (H(0), CX(0,1)), hand-verified below.
TEST(TreewidthTest, BellCircuitOneSidedGraphMatchesHandCount) {
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});

    // qubit 0's timeline: [H(node 0), CX(node 1)] -> edge (0,1).
    // qubit 1's timeline: [CX(node 1)] only -> no edge.
    const TensorNetworkGraph graph =
        build_tensor_network_graph(circuit.num_qubits(), circuit.operations());
    ASSERT_EQ(graph.num_nodes, 2U);
    ASSERT_EQ(graph.edges.size(), 1U);
    EXPECT_EQ(graph.edges[0], (std::pair<std::size_t, std::size_t>{0, 1}));
    EXPECT_EQ(estimate_treewidth_min_fill(graph).width, 1U);
}

// The CLOSED network for the same Bell circuit against a single-qubit
// Z0 observable, hand-verified: ket nodes {H_ket, CX_ket}, bra nodes
// {H_bra, CX_bra}, one observable node (Z) on qubit 0 only (the term is
// identity on qubit 1, so qubit 1's ket and bra ends connect directly
// with no node there), plus 4 boundary nodes (one |0> and one <0| per
// qubit). Qubit 0's combined timeline
// [B0_ket, H_ket, CX_ket, Z, CX_bra, H_bra, B0_bra] gives edges
// (B0_ket,H_ket), (H_ket,CX_ket), (CX_ket,Z), (Z,CX_bra), (CX_bra,H_bra),
// (H_bra,B0_bra); qubit 1's combined timeline
// [B1_ket, CX_ket, CX_bra, B1_bra] gives edges (B1_ket,CX_ket),
// (CX_ket,CX_bra), (CX_bra,B1_bra) -- 9 nodes, 9 edges total. The 5-node
// "core" (H_ket, CX_ket, Z, CX_bra, H_bra) is exactly the same triangle-
// plus-two-pendants structure the pre-boundary-fix version of this test
// already hand-verified has treewidth 2 (triangle {CX_ket, CX_bra, Z},
// H_ket and H_bra each pendant off one triangle vertex); the 4 new
// boundary nodes are themselves degree-1 pendants (off H_ket, H_bra,
// and the two CX nodes respectively), which provably never raise
// treewidth above the core structure's own value, so the closed
// network's treewidth is still exactly 2.
TEST(TreewidthTest, BellCircuitClosedNetworkAgainstZ0MatchesHandDerivedTreewidth) {
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});

    PauliString z0;
    z0.num_qubits = 2;
    z0.x_words = {0};
    z0.z_words = {0b01};

    const TensorNetworkGraph graph =
        build_closed_contraction_graph(circuit.num_qubits(), circuit.operations(), z0);
    ASSERT_EQ(graph.num_nodes, 9U);
    ASSERT_EQ(graph.edges.size(), 9U);
    EXPECT_EQ(estimate_treewidth_min_fill(graph).width, 2U);
}

// The SAME Bell+Z0 network's own LINE GRAPH (analysis::LineGraph, docs/
// BUILD_PLAN.md Phase 8, resolved with the owner after an earlier
// attempt applied min-fill directly to the tensor-node graph above and
// found the resulting contraction did not terminate in reasonable time):
// contraction complexity is bounded by THIS graph's own treewidth, not
// the tensor-node graph's. Nodes = the same 9 wires the tensor-node
// graph's own 9 edges already are; edges connect two wires iff they meet
// at a common tensor node. Hand-derived (naming wires w1..w9 in the same
// per-qubit timeline order the test above already documents): qubit 0's
// 6 wires (B0_ket-H_ket, H_ket-CX_ket, CX_ket-Z, Z-CX_bra, CX_bra-H_bra,
// H_bra-B0_bra) plus qubit 1's 3 wires (B1_ket-CX_ket, CX_ket-CX_bra,
// CX_bra-B1_bra). Each tensor node's own incident wires form a clique:
// boundary nodes (rank 1, one wire each) contribute nothing; H_ket
// (wires 1,2) and H_bra (wires 5,6) each contribute one edge; Z (wires
// 3,4) contributes one edge; CX_ket (wires 2,3,7,8, rank 4) and CX_bra
// (wires 4,5,8,9, rank 4) each contribute a 4-clique (6 edges) -- total
// 1+1+1+6+6 = 15 edges, all distinct. Min-fill on this graph (worked by
// hand, tie-break lowest index throughout): eliminate the two rank-1
// boundary wires first (degree 1, fill 0, width so far 1); then w2
// (degree 3, neighbors {w3,w7,w8} already pairwise connected via
// CX_ket's own clique, fill 0, width 3); then w7 (degree 2, {w3,w8}
// already connected via the same clique, fill 0, width still 3); then w3
// (degree 2, {w4,w8} already connected via CX_bra's own clique, fill 0,
// width still 3); the remaining {w4,w5,w8,w9} are CX_bra's own untouched
// 4-clique, eliminated in order with fill 0 throughout (width stays 3).
// Final width 3 (not the tensor-node graph's own width 2 above -- the
// line graph's treewidth is a genuinely different quantity, expected to
// be larger since it must additionally capture the internal mixing of
// each multi-legged tensor's own indices).
TEST(TreewidthTest, BellCircuitLineGraphAgainstZ0MatchesHandDerivedTreewidth) {
    Circuit circuit(2, 0);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});

    PauliString z0;
    z0.num_qubits = 2;
    z0.x_words = {0};
    z0.z_words = {0b01};

    const ClosedTimelines timelines =
        build_closed_timelines(circuit.num_qubits(), circuit.operations(), z0);
    const LineGraph line_graph = build_line_graph(timelines);
    ASSERT_EQ(line_graph.graph.num_nodes, 9U);
    ASSERT_EQ(line_graph.wire_of.size(), 9U);
    ASSERT_EQ(line_graph.graph.edges.size(), 15U);
    EXPECT_EQ(estimate_treewidth_min_fill(line_graph.graph).width, 3U);
}

// docs/BUILD_PLAN.md Phase 8's own design conflict, resolved with the
// owner: the tensor-network engine's estimate() runs a real search
// (unlike every other engine's closed-form estimate()), so that search
// must be boundable by the job's own budget rather than running
// unconditionally to completion -- confirmed necessary by direct
// measurement (an ordinary, moderately-entangled circuit's closed
// network made the unbounded search take minutes).
TEST(TreewidthTest, TightBudgetStopsTheSearchEarlyWithAPartialOrder) {
    // K_6: true treewidth 5, verified above by CompleteGraphK6TreewidthIsFive.
    // Eliminating a degree-5 vertex materializes an intermediate over 6
    // indices (the 5 neighbours plus the one being summed away), so a
    // completed search needs 16 * 2^6 = 1024 bytes, not 16 * 2^5. These
    // constants were 512/256 while the estimator used 2^degree; they moved
    // when that was corrected to 2^(degree+1) after
    // TensorNetworkEngineTest.EstimatedPeakBytesBoundTheActualPeakAcrossRandomCircuits
    // measured execute()'s own peak and found the old figure understated it
    // by exactly a factor of two. The behaviour asserted here is unchanged;
    // only the byte figures that express it are.
    //
    // Every vertex in K_6 has degree 5 from the very first step, so a budget
    // below 1024 must stop the search after 0 completed steps.
    const TensorNetworkGraph graph = make_complete_graph(6);
    const EliminationOrder bounded = estimate_treewidth_min_fill(graph, /*max_peak_bytes=*/512);

    EXPECT_TRUE(bounded.exceeded_budget);
    EXPECT_TRUE(bounded.order.empty());
    EXPECT_TRUE(bounded.per_step_degree.empty());

    // A budget generous enough for the true width (1024 bytes exactly)
    // lets the search complete normally, matching the unbounded result.
    const EliminationOrder generous = estimate_treewidth_min_fill(graph, /*max_peak_bytes=*/1024);
    EXPECT_FALSE(generous.exceeded_budget);
    EXPECT_EQ(generous.width, 5U);
    EXPECT_EQ(generous.order.size(), 6U);

    // The default (no budget given) is unaffected -- exceeded_budget is
    // false for every existing call site that predates this parameter.
    const EliminationOrder unbounded = estimate_treewidth_min_fill(graph);
    EXPECT_FALSE(unbounded.exceeded_budget);
    EXPECT_EQ(unbounded.width, 5U);
}

// The dense-bitset search and the original std::set search must agree exactly
// -- order, per-step degrees, width and the budget flag -- on graphs of every
// density, with and without a peak-byte bound.
TEST(TreewidthTest, DenseAndSparseMinFillAgreeOnRandomGraphs) {
    std::uint64_t state = 0x2545F4914F6CDD1DULL;
    const auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 1 + next() % 90;
        const std::uint64_t density = 1 + next() % 40;  // percent
        TensorNetworkGraph g{n, {}};
        for (std::size_t a = 0; a < n; ++a) {
            for (std::size_t b = a + 1; b < n; ++b) {
                if (next() % 100 < density) {
                    g.edges.emplace_back(a, b);
                }
            }
        }
        const std::uint64_t bound =
            trial % 3 == 0 ? std::numeric_limits<std::uint64_t>::max() : (16ULL << (next() % 20));
        const EliminationOrder dense = estimate_treewidth_min_fill_dense(g, bound);
        const EliminationOrder sparse = estimate_treewidth_min_fill_sparse(g, bound);
        EXPECT_EQ(dense.order, sparse.order) << "trial " << trial;
        EXPECT_EQ(dense.per_step_degree, sparse.per_step_degree) << "trial " << trial;
        EXPECT_EQ(dense.width, sparse.width) << "trial " << trial;
        EXPECT_EQ(dense.exceeded_budget, sparse.exceeded_budget) << "trial " << trial;
    }
}

// The caller's stop predicate stops the search right after the step that
// returned true, leaving that step recorded.
TEST(TreewidthTest, StopPredicateStopsAfterTheRequestedStep) {
    const TensorNetworkGraph g = make_grid_graph(4);
    const EliminationOrder full = estimate_treewidth_min_fill(g);
    std::size_t calls = 0;
    const EliminationOrder stopped = estimate_treewidth_min_fill(
        g, std::numeric_limits<std::uint64_t>::max(), [&](std::size_t) { return ++calls == 5; });
    EXPECT_TRUE(stopped.stopped_by_caller);
    EXPECT_FALSE(full.stopped_by_caller);
    ASSERT_EQ(stopped.order.size(), 5U);
    EXPECT_TRUE(std::equal(stopped.order.begin(), stopped.order.end(), full.order.begin()));
}

}  // namespace
}  // namespace menos::analysis
