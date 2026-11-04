# Engine techniques: planner data structures

Engineering notes for maintainers; kept off the documentation site. This
file records the data structures the planner's analysis passes use after
Phase 21 step 2, why each is exact, and where the idea comes from. The
measurements and the decisions behind them are in
`docs/adr/0025-phase-21-planner-performance.md`, sections 10 and 12.
Everything here is written for MENOS; no code is copied from elsewhere.

The planner's output (the chosen plan, its cost and `explain()`) is the same
as before every change below. That is checked by 45 golden plan files and a
differential test against a frozen copy of the pre-phase planner.

## Min-fill elimination (tensor-network costing)

`src/analysis/treewidth.cpp`, `bit_adjacency.h`, `min_fill_queue.h`.

The tensor-network candidate's cost comes from a greedy elimination
order on the closed network's line graph. At each step it eliminates the
vertex whose elimination adds the fewest new edges (its *fill*), with ties
going to the lowest index. This is the classic min-fill heuristic for
treewidth upper bounds, used in sparse-matrix ordering and in tensor-network
contraction-order search.

### Bitset adjacency with exact spans

Each vertex's neighbours are a row of 64-bit words.
- fill(v) is C(deg v, 2) minus the edges already among v's neighbours.
  Those edges are counted as the sum, over neighbours a, of
  popcount(row a AND row v), halved.
- Each row keeps its exact non-zero word span `[first, last)` and its
  degree, both updated by `set`, `reset` and `clear_row`. Every loop walks
  the span only. A reset that empties an end word narrows the span past
  the empty words there.
- When a row has fewer neighbours than words in its span, the count uses
  bit tests between neighbour pairs instead of popcounts. Both give the
  same integer, and the choice is a direct comparison of the two costs.
- On x86-64 Linux with AVX2, dense rows are popcounted four words at a
  time. Each byte's count is looked up from its two nibbles with a byte
  shuffle, and the byte counts are summed into 64-bit lanes with a
  sum-of-absolute-differences instruction; this is the well-known
  nibble-lookup popcount method. It is selected once at startup with
  `__builtin_cpu_supports`. Every other platform uses `std::popcount`.

### Indexed binary heap with packed keys

The next vertex comes from an indexed min-heap: one entry per remaining
vertex, plus a table of each vertex's position, so a fill change moves the
entry up or down in place (decrease/increase-key).
- An entry is the single integer `(fill << 32) | vertex`. Integer order is
  fill first, then lowest vertex, which is exactly the tie-break of the
  original linear scan.
- The heap never holds more than n entries.
- The heap is built bottom-up in linear time (Floyd's method).

### Exact incremental fill update

After v is eliminated, only two kinds of vertex can change fill:
- **v's neighbours** are recomputed in full.
- **Any other remaining vertex u** keeps its neighbourhood unchanged, and
  the only new adjacency among its neighbours is the new fill-in edges.
  So fill(u) drops by exactly the number of new edges with both ends
  adjacent to u. The search walks the common neighbours of each new edge's
  two ends and subtracts.

When v's fill is 0, nothing outside its neighbourhood is touched. All of
this is integer counting, so each fill equals a full recompute. The
original `std::set` search, which recomputes and scans everything, is kept
as the test oracle.

## Schmidt-rank bound (MPS costing)

`src/analysis/schmidt_rank_bound.cpp`, `schmidt_rank_internal.h`.

The MPS candidate bounds each cut's bond dimension. At each cut it
multiplies the operator Schmidt ranks of the gates spanning that cut. A
rank is the matrix rank of the gate reshaped across the split, the
standard operator-Schmidt decomposition, taken by SVD.
- **Skip work that cannot matter.** Single-qubit operations and
  measurements never span a cut, so no matrix is built for them.
- **Per-call rank cache.** The key is fixed-width fields (operand count,
  left-operand count, left positions) followed by the matrix's raw bytes,
  so no two inputs alias. A rank is a deterministic function of exactly
  those bytes, so a hit equals a recompute.
- **Bounded size.** At most 16 MiB of key bytes. Reaching the cap clears
  the cache, and a key larger than the cap is computed and not stored.
  Neither can change a result.
- **Counters.** An optional build (`MENOS_PLANNER_COUNTERS`) reports
  hits, misses and peak bytes per benchmark job.

**Clifford gates (investigated, not implemented).** For a Clifford gate,
the rank across A | B equals 2 to the power of the GF(2) rank of the block
of its symplectic (Pauli-conjugation) matrix that maps A's Paulis into B.
This is the stabilizer-formalism view of Clifford dynamics
(Gottesman–Knill). It matched the SVD on every two-qubit Clifford and on
8,000 random 3- and 4-qubit cases. Since the rank cache, the SVD is about
2% of the most Clifford-heavy corpus job, so it is not implemented
(docs/adr/0025 section 12.3, C.4).
