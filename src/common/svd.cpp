#include "common/svd.h"

// This system's installed lapacke.h includes lapack.h directly, which
// only consults LAPACK_COMPLEX_CPP inside an "#ifdef HAVE_LAPACK_CONFIG_H"
// block that is not otherwise reached -- verified by reading
// /usr/include/lapack.h, not assumed from memory. The header's own
// documented alternative (its comment block above the complex-type
// section) is LAPACK_COMPLEX_CUSTOM: define lapack_complex_float and
// lapack_complex_double ourselves before including, which this system's
// lapack.h honors directly (the "#ifndef lapack_complex_double" guard
// short-circuits once we've defined it). This still keeps the LAPACKE
// dependency fully encapsulated in this one translation unit -- svd.h
// exposes only std::complex<double>, no reinterpret_cast or _Complex
// GNU extension anywhere past this file.
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#ifdef _MSC_VER
// C4190: lapacke.h declares lapack_make_complex_float/double extern "C" while
// returning the std::complex types defined above. MENOS never calls them; the
// functions it does call take only pointers and integers (docs/adr/0022).
#pragma warning(push)
#pragma warning(disable : 4190)
#endif
#include <lapacke.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <string>
#include <vector>

#include "common/error.h"
#include "common/tolerance.h"

namespace menos::common {

namespace {

// zgesdd's own workspace query (lwork = -1): the optimal complex workspace for
// an m x n jobz='S' decomposition. Reads no matrix data.
lapack_int zgesdd_optimal_lwork(lapack_int m, lapack_int n) {
    const char jobz = 'S';
    const lapack_int lda = std::max<lapack_int>(1, m);
    const lapack_int ldu = std::max<lapack_int>(1, m);
    const lapack_int ldvt = std::max<lapack_int>(1, std::min(m, n));
    const lapack_int query = -1;
    std::complex<double> optimal{0.0, 0.0};
    double rwork_dummy = 0.0;
    lapack_int iwork_dummy = 0;
    lapack_int info = 0;
    LAPACK_zgesdd(&jobz,
                  &m,
                  &n,
                  nullptr,
                  &lda,
                  nullptr,
                  nullptr,
                  &ldu,
                  nullptr,
                  &ldvt,
                  &optimal,
                  &query,
                  &rwork_dummy,
                  &iwork_dummy,
                  &info);
    ensure(info == 0, "svd: zgesdd workspace query failed with info=" + std::to_string(info));
    return static_cast<lapack_int>(optimal.real());
}

}  // namespace

SvdResult svd(std::span<const std::complex<double>> a, std::size_t m, std::size_t n) {
    ensure(a.size() == m * n, "svd: matrix size does not match m * n");
    ensure(m > 0 && n > 0, "svd: m and n must both be positive");

    const std::size_t k = std::min(m, n);
    const auto lm = static_cast<lapack_int>(m);
    const auto ln = static_cast<lapack_int>(n);
    const auto lk = static_cast<lapack_int>(k);

    // zgesdd called column-major, exactly as LAPACKE_zgesdd's row-major path
    // calls it (docs/adr/0025 section 3.3): the input transposed into a
    // column-major buffer with lda = m, U written column-major with ldu = m,
    // VT with ldvt = k, the workspace from zgesdd's own query, and the
    // outputs transposed back. zgesdd therefore receives exactly the inputs it
    // did through LAPACKE, so the results are bitwise identical
    // (tests/common/svd_test.cpp checks this against LAPACKE), but every
    // buffer is MENOS's own: none is hidden in a malloc inside LAPACKE, and
    // the caller's matrix is transposed once instead of copied and then
    // transposed. jobz='S' is the thin/economy SVD, exactly k = min(m, n)
    // singular values and vectors -- what every caller (Schmidt-rank bound,
    // MPS truncation) needs. zgesdd destroys its input, which is now the
    // transposed copy.
    std::vector<std::complex<double>> a_col(m * n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            a_col[(j * m) + i] = a[(i * n) + j];
        }
    }
    std::vector<std::complex<double>> u_col(m * k);
    std::vector<std::complex<double>> vt_col(k * n);
    SvdResult result;
    result.singular_values.resize(k);

    const lapack_int lwork = zgesdd_optimal_lwork(lm, ln);
    std::vector<std::complex<double>> work(
        static_cast<std::size_t>(std::max<lapack_int>(1, lwork)));
    // LAPACKE's size for jobz != 'N'.
    const std::size_t lrwork =
        std::max<std::size_t>(1, k * std::max((5 * k) + 7, (2 * std::max(m, n)) + (2 * k) + 1));
    std::vector<double> rwork(lrwork);
    std::vector<lapack_int> iwork(std::max<std::size_t>(1, 8 * k));

    const char jobz = 'S';
    const lapack_int lda = std::max<lapack_int>(1, lm);
    const lapack_int ldu = std::max<lapack_int>(1, lm);
    const lapack_int ldvt = std::max<lapack_int>(1, lk);
    lapack_int info = 0;
    LAPACK_zgesdd(&jobz,
                  &lm,
                  &ln,
                  a_col.data(),
                  &lda,
                  result.singular_values.data(),
                  u_col.data(),
                  &ldu,
                  vt_col.data(),
                  &ldvt,
                  work.data(),
                  &lwork,
                  rwork.data(),
                  iwork.data(),
                  &info);
    ensure(info == 0, "svd: zgesdd failed with info=" + std::to_string(info));

    // Release the input and workspace before the row-major outputs exist.
    a_col = {};
    work = {};
    rwork = {};
    iwork = {};

    result.u.resize(m * k);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            result.u[(i * k) + j] = u_col[(j * m) + i];
        }
    }
    u_col = {};
    result.vt.resize(k * n);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result.vt[(i * n) + j] = vt_col[(j * k) + i];
        }
    }
    return result;
}

std::size_t numerical_rank(const std::vector<double>& singular_values,
                           std::size_t m,
                           std::size_t n) {
    if (singular_values.empty()) {
        return 0;
    }
    const double sigma_max = singular_values.front();
    const double cutoff = sigma_max * static_cast<double>(std::max(m, n)) * kSvdRankRelativeEpsilon;
    std::size_t rank = 0;
    for (const double sigma : singular_values) {
        if (sigma > cutoff) {
            ++rank;
        }
    }
    return rank;
}

}  // namespace menos::common
