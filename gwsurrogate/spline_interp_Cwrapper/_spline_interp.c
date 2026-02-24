#include <assert.h>
#include <stddef.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#if !defined(assert)
#define assert(...)
#endif

/**
 * spline_interp.h — Fast natural cubic spline interpolation
 *
 * Design goals met:
 *   - Single heap allocation per prepare/evaluate call
 *   - FMA-friendly evaluation via Horner's method
 *   - Precomputed reciprocals for inner loops
 *   - Multi-dataset support sharing one spline_plan
 *   - restrict-annotated pointers throughout
 *   - GCC/Clang vectorization pragmas
 *   - Thread-safe (no global/static state)
 *   - Hunt algorithm for O(log n) → O(1) amortized lookup on sorted out_x
 */

/* -------------------------------------------------------------------------
 * Return codes
 * ---------------------------------------------------------------------- */
#define SPLINE_OK            0
#define SPLINE_ERR_OOM       1   /* malloc failed                          */
#define SPLINE_ERR_BOUNDS    2   /* out_x value outside [data_x[0], data_x[n-1]] */
#define SPLINE_ERR_SINGULAR  3   /* zero-length interval in data_x (degenerate) */
#define SPLINE_ERR_DATA_SIZE 4   /* data_size < 3 (need at least 3 knots)      */
#define SPLINE_ERR_OUT_SIZE  5   /* out_size <= 0                              */
#define SPLINE_ERR_NUM_DS    6   /* num_datasets <= 0                          */

/* -------------------------------------------------------------------------
 * spline_plan — opaque handle produced by spline_prepare()
 *
 * Owns one heap allocation containing:
 *   double h[n-1]          — interval widths  data_x[i+1] - data_x[i]
 *   double inv_h[n-1]      — 1/h[i], precomputed
 *   double inv_6h[n-1]     — 1/(6*h[i]), precomputed
 *   double diag[n]         — diagonal of tridiagonal system (scratch → reused
 *                            as lower-diag factor after factorisation)
 *   (the data_x pointer is stored but NOT owned — caller keeps it alive)
 * ---------------------------------------------------------------------- */
typedef struct spline_plan spline_plan;

/* -------------------------------------------------------------------------
 * spline_prepare — compute the geometry arrays from data_x.
 *
 * Parameters:
 *   data_x    — monotone increasing knot x-values, length data_size
 *   data_size — number of knots (must be >= 3)
 *
 * Returns a heap-allocated spline_plan on success, NULL on allocation failure.
 * The caller must free with spline_plan_free().
 * data_x must remain valid and unchanged for the lifetime of the plan.
 * ---------------------------------------------------------------------- */
spline_plan *spline_prepare(const double *restrict data_x, long data_size,
                            int *err_out);

/* -------------------------------------------------------------------------
 * spline_plan_free — release all memory owned by the plan.
 * ---------------------------------------------------------------------- */
void spline_plan_free(spline_plan *plan);

/* -------------------------------------------------------------------------
 * spline_interp_multi — interpolate num_datasets y-vectors using one plan.
 *
 * Parameters:
 *   plan        — from spline_prepare(); encodes data_x geometry
 *   data_y      — array of num_datasets pointers, each pointing to data_size
 *                 doubles (the y-values for that dataset at the shared data_x)
 *   out_x       — evaluation points, length out_size, sorted ascending,
 *                 must lie within [data_x[0], data_x[data_size-1]]
 *   out_y       — array of num_datasets pointers, each pointing to out_size
 *                 doubles to receive interpolated values
 *   out_size    — number of evaluation points (must be > 0)
 *   num_datasets— number of independent y-datasets
 *
 * Shared work (tridiagonal solves, hunt indices) is done once for all datasets.
 * Returns SPLINE_OK, SPLINE_ERR_OOM, SPLINE_ERR_BOUNDS, or SPLINE_ERR_SINGULAR.
 * ---------------------------------------------------------------------- */
int spline_interp_multi_impl(const spline_plan *restrict plan,
                             const double *const *restrict data_y,
                             const double *restrict out_x,
                             double *const *restrict out_y,
                             long out_size,
                             long num_datasets);

/* -------------------------------------------------------------------------
 * spline_interp — single-dataset convenience wrapper.
 *
 * Equivalent to calling spline_prepare + spline_interp_multi(num_datasets=1)
 * + spline_plan_free in sequence, with a single allocation.
 * ---------------------------------------------------------------------- */
int spline_interp_impl(long data_size, long out_size,
                       const double *restrict data_x,
                       const double *restrict data_y,
                       const double *restrict out_x,
                       double *restrict out_y);

/**
 * spline_interp.c — Fast natural cubic spline interpolation
 *
 * Natural spline => y''(x_0) = y''(x_{n-1}) = 0.
 *
 * Parameterisation
 * ----------------
 * We store c[i] = S''(x_i)  (the full second derivative).
 *
 * On [x_i, x_{i+1}], with t = x - x_i:
 *   S(x) = y[i] + b[i]*t + (c[i]/2)*t^2 + d[i]*t^3
 * where
 *   d[i] = (c[i+1] - c[i]) / (6*h[i])
 *   b[i] = (y[i+1] - y[i])/h[i] - h[i]*(2*c[i] + c[i+1])/6
 *
 * c[i] satisfies (for i = 1..n-2, natural BCs c[0]=c[n-1]=0):
 *   h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1]
 *       = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
 *
 * Thomas algorithm
 * ----------------
 * The boundary rows are NOT part of elimination (c[0],c[n-1] fixed to 0).
 * Interior system is (n-2)x(n-2).  Forward sweep starts at interior row k=1
 * (which corresponds to global index i=2), using row k=0 (global i=1) as
 * the first unmodified pivot.
 *
 * plan->diag[k] = factored diagonal for interior row k (k=0..n-3).
 *
 * Thread safety: no global/static state.
 */

#if defined(__GNUC__) || defined(__clang__)
#  define SPLINE_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define SPLINE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  define SPLINE_INLINE      __attribute__((always_inline)) static inline
#else
#  define SPLINE_LIKELY(x)   (x)
#  define SPLINE_UNLIKELY(x) (x)
#  define SPLINE_INLINE      static inline
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#  include <math.h>
#  define SPLINE_FMA(a,b,c) fma((a),(b),(c))
#else
#  define SPLINE_FMA(a,b,c) ((a)*(b)+(c))
#endif

/* ──────────────────────────────────────────────
 * Hunt algorithm for interval location.
 *
 * Given a target value `target`, find the interval index `idx` such
 * that data_x[idx] <= target <= data_x[idx + 1].
 *
 * Starts from `guess` (typically the last known interval) and
 * expands geometrically until the target is bracketed, then
 * binary searches within the bracket.
 *
 * Returns the interval index, or -1 if target is out of range.
 *
 * Reference: Numerical Recipes, "hunt" routine.
 * ────────────────────────────────────────────── */

static int hunt(const double *const restrict data_x,
                const int data_size,
                const double target,
                const int guess)
{
    const int last_interval = data_size - 2;

    /* Out of range checks */
    if (target < data_x[0] || target > data_x[data_size - 1]) {
        return -1;
    }

    /* Clamp guess to valid range */
    int lo, hi;
    int g = guess;
    if (g < 0) g = 0;
    if (g > last_interval) g = last_interval;

    /* Check if we're already in the right interval */
    if (data_x[g] <= target && target <= data_x[g + 1]) {
        return g;
    }

    /* Determine hunt direction */
    if (target >= data_x[g]) {
        /* Hunt upward */
        lo = g;
        int increment = 1;
        hi = lo + increment;
        while (hi <= last_interval && data_x[hi] < target) {
            lo = hi;
            increment *= 2;
            hi = lo + increment;
        }
        if (hi > last_interval) {
            hi = last_interval;
        }
    } else {
        /* Hunt downward */
        hi = g;
        int increment = 1;
        lo = hi - increment;
        while (lo >= 0 && data_x[lo + 1] > target) {
            hi = lo;
            increment *= 2;
            lo = hi - increment;
        }
        if (lo < 0) {
            lo = 0;
        }
    }

    /* Binary search within [lo, hi] */
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (data_x[mid] <= target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return lo;
}

/* =========================================================================
 * spline_plan (internal definition)
 * ====================================================================== */
struct spline_plan {
    const double *data_x;   /* NOT owned */
    long          n;

    void   *mem;            /* single heap block               */
    double *h;              /* h[n-1]   interval widths        */
    double *inv_h;          /* inv_h[n-1] = 1/h[i]             */
    double *diag;           /* diag[n-2] factored interior diag */
};

static size_t align_double(size_t nb) {
    const size_t a = sizeof(double);
    return (nb + a - 1u) & ~(a - 1u);
}

/* =========================================================================
 * spline_prepare
 * ====================================================================== */
spline_plan *spline_prepare(const double *restrict data_x, long data_size,
                            int *err_out)
{
    if (data_size < 3) { *err_out = SPLINE_ERR_DATA_SIZE; return NULL; }

    const long n  = data_size;
    const long nm = n - 1;   /* intervals           */
    const long ni = n - 2;   /* interior points     */

    const size_t plan_sz = align_double(sizeof(spline_plan));
    const size_t h_sz    = (size_t)nm * sizeof(double);
    const size_t diag_sz = (size_t)ni * sizeof(double);
    const size_t total   = plan_sz + 2u*h_sz + diag_sz;

    unsigned char *mem = (unsigned char *)malloc(total);
    if (SPLINE_UNLIKELY(!mem)) { *err_out = SPLINE_ERR_OOM; return NULL; }

    spline_plan *plan = (spline_plan *)mem;
    plan->mem    = mem;
    plan->data_x = data_x;
    plan->n      = n;

    unsigned char *ptr = mem + plan_sz;
    plan->h     = (double *)ptr; ptr += h_sz;
    plan->inv_h = (double *)ptr; ptr += h_sz;
    plan->diag  = (double *)ptr;

    double *restrict h     = plan->h;
    double *restrict inv_h = plan->inv_h;
    double *restrict diag  = plan->diag;

    /* Interval widths and reciprocals */
#if defined(__clang__)
#  pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#  pragma GCC ivdep
#endif
    for (long i = 0; i < nm; ++i) {
        const double hi = data_x[i+1] - data_x[i];
        if (SPLINE_UNLIKELY(hi == 0.0)) goto singular;
        h[i]     = hi;
        inv_h[i] = 1.0 / hi;
    }

    /*
     * Factor the (n-2)×(n-2) tridiagonal.
     *
     * Row k (k=0..ni-1) corresponds to interior point i=k+1.
     * diagonal[k] = 2*(h[k]+h[k+1])
     * sub-diag[k] (entry below diag[k], i.e. coeff of c[i-1] in row i) = h[k]
     * super-diag[k]                                                       = h[k+1]
     *
     * k=0: no row above it (c[0]=0 is fixed, not a pivot row).
     *   diag[0] = 2*(h[0]+h[1])     (unmodified)
     * k>=1:
     *   factor   = h[k] / diag[k-1]       (sub-diag / previous pivot)
     *   diag[k]  = 2*(h[k]+h[k+1]) - factor*h[k]
     */
    diag[0] = 2.0 * (h[0] + h[1]);

    for (long k = 1; k < ni; ++k) {
        if (SPLINE_UNLIKELY(diag[k-1] == 0.0)) goto singular;
        const double factor = h[k] / diag[k-1];
        diag[k] = 2.0*(h[k] + h[k+1]) - factor*h[k];
    }
    if (SPLINE_UNLIKELY(diag[ni-1] == 0.0)) goto singular;

    return plan;

singular:
    free(mem);
    *err_out = SPLINE_ERR_SINGULAR;
    return NULL;
}

void spline_plan_free(spline_plan *plan) {
    if (plan) free(plan->mem);
}

/* =========================================================================
 * spline_interp_multi
 * ====================================================================== */
int spline_interp_multi_impl(const spline_plan *restrict plan,
                             const double *const *restrict data_y,
                             const double *restrict out_x,
                             double *const *restrict out_y,
                             long out_size,
                             long num_datasets)
{
    if (out_size <= 0) return SPLINE_ERR_OUT_SIZE;
    if (num_datasets <= 0) return SPLINE_ERR_NUM_DS;

    const long   n        = plan->n;
    const long   ni       = n - 2;
    const double *restrict data_x   = plan->data_x;
    const double *restrict h        = plan->h;
    const double *restrict inv_h    = plan->inv_h;
    const double *restrict diag_fac = plan->diag;

    /* One allocation: c arrays + rhs scratch */
    const size_t c_sz   = (size_t)n  * (size_t)num_datasets * sizeof(double);
    const size_t rhs_sz = (size_t)ni * sizeof(double);
    const size_t idx_sz = (size_t)out_size * sizeof(int);
    unsigned char *mem = (unsigned char *)malloc(c_sz + rhs_sz + idx_sz);
    if (SPLINE_UNLIKELY(!mem)) return SPLINE_ERR_OOM;

    double *restrict c_all   = (double *)mem;
    double *restrict rhs     = (double *)(mem + c_sz);
    int    *restrict idx_buf = (int *)(mem + c_sz + rhs_sz);

    /* -------------------------------------------------------------------
     * Phase 1: Tridiagonal solve for c[1..n-2] (one per dataset).
     *
     * rhs[k] corresponds to interior point i = k+1.
     * raw_rhs[k] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
     *            = 6*(y[i+1]*inv_h[i] - y[i]*(inv_h[i]+inv_h[i-1]) + y[i-1]*inv_h[i-1])
     *
     * Forward sweep:
     *   rhs[0] = raw_rhs[0]                                 (no row above)
     *   rhs[k] = raw_rhs[k] - (h[k]/diag[k-1])*rhs[k-1]   (k>=1)
     *
     * Back-substitution:
     *   c[n-2]  = rhs[ni-1] / diag[ni-1]
     *   c[k+1]  = (rhs[k] - h[k+1]*c[k+2]) / diag[k]      (k = ni-2..0)
     * ------------------------------------------------------------------- */
    for (long d = 0; d < num_datasets; ++d) {
        const double *restrict y = data_y[d];
        double *restrict       c = c_all + d * n;

        c[0]     = 0.0;
        c[n - 1] = 0.0;

        /* k=0, i=1: no elimination needed */
        rhs[0] = 6.0 * SPLINE_FMA( y[2],  inv_h[1],
                         SPLINE_FMA(-y[1], inv_h[1] + inv_h[0],
                                     y[0] * inv_h[0]));

        /* k=1..ni-1, i=2..n-2 */
        for (long k = 1; k < ni; ++k) {
            const long   i   = k + 1;
            const double raw = 6.0 * SPLINE_FMA( y[i+1], inv_h[i],
                                        SPLINE_FMA(-y[i], inv_h[i] + inv_h[i-1],
                                                    y[i-1] * inv_h[i-1]));
            /* sub-diag of row k is h[k] (= h[i-1]) */
            const double factor = h[k] / diag_fac[k-1];
            rhs[k] = raw - factor * rhs[k-1];
        }

        /* Back-substitution */
        c[n-2] = rhs[ni-1] / diag_fac[ni-1];

        for (long k = ni-2; k >= 0; --k) {
            const long i = k + 1;
            /* super-diag of row k is h[k+1] (= h[i]) */
            c[i] = (rhs[k] - h[k+1] * c[i+1]) / diag_fac[k];
        }
    }

    /* -------------------------------------------------------------------
     * Phase 2a: Precompute interval indices for all output points.
     * ------------------------------------------------------------------- */
    {
        int prev_idx = 0;
        for (long j = 0; j < out_size; ++j) {
            const int idx = hunt(data_x, (int)n, out_x[j], prev_idx);
            if (SPLINE_UNLIKELY(idx < 0)) {
                free(mem);
                return SPLINE_ERR_BOUNDS;
            }
            idx_buf[j] = idx;
            prev_idx = idx;
        }
    }

    /* -------------------------------------------------------------------
     * Phase 2b: Evaluate spline — d-outer for cache locality on c/y rows,
     *           j-inner for stride-1 writes to out_row.
     *
     * Horner form: S = y[i] + t*(b + t*(c[i]/2 + t*d))
     * ------------------------------------------------------------------- */
    for (long d = 0; d < num_datasets; ++d) {
        const double *restrict y = data_y[d];
        const double *restrict c = c_all + d * n;
        double *restrict out_row = out_y[d];

#if defined(__clang__)
#  pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#  pragma GCC ivdep
#endif
        for (long j = 0; j < out_size; ++j) {
            const int    idx     = idx_buf[j];
            const double t       = out_x[j] - data_x[idx];
            const double inv_hi  = inv_h[idx];
            const double hi_inv6 = h[idx] * (1.0 / 6.0);

            const double ci  = c[idx];
            const double ci1 = c[idx + 1];
            const double d_coeff = (ci1 - ci) * inv_hi * (1.0/6.0);
            const double b = SPLINE_FMA(-hi_inv6,
                                        SPLINE_FMA(2.0, ci, ci1),
                                        (y[idx+1] - y[idx]) * inv_hi);

            out_row[j] = SPLINE_FMA(t,
                         SPLINE_FMA(t,
                         SPLINE_FMA(t, d_coeff, ci * 0.5),
                         b),
                         y[idx]);
        }
    }

    free(mem);
    return SPLINE_OK;
}

/* =========================================================================
 * spline_interp — single-dataset convenience wrapper
 * ====================================================================== */
int spline_interp_impl(long data_size, long out_size,
                       const double *restrict data_x,
                       const double *restrict data_y,
                       const double *restrict out_x,
                       double *restrict out_y)
{
    if (out_size <= 0) return SPLINE_ERR_OUT_SIZE;

    int err = SPLINE_OK;
    spline_plan *plan = spline_prepare(data_x, data_size, &err);
    if (SPLINE_UNLIKELY(!plan)) return err;

    const double *yd[1] = { data_y };
    double       *yo[1] = { out_y  };

    const int rc = spline_interp_multi_impl(plan, yd, out_x, yo, out_size, 1);
    spline_plan_free(plan);
    return rc;
}

int spline_interp(const long data_size, const long out_size,
                  double const * const data_x, const double  *  const data_y,
                  double const *  const out_x, double *out_y) {
  return spline_interp_impl(data_size, out_size,
                            data_x,data_y,
                            out_x, out_y);
}

int spline_interp_multi(const long data_size, const long out_size,
                        const long num_datasets,
                        const double *restrict data_x,
                        const double *const *restrict data_y,
                        const double *restrict out_x,
                        double *const *restrict out_y) {
    if (out_size <= 0) return SPLINE_ERR_OUT_SIZE;
    if (num_datasets <= 0) return SPLINE_ERR_NUM_DS;

    int err = SPLINE_OK;
    spline_plan *plan = spline_prepare(data_x, data_size, &err);
    if (SPLINE_UNLIKELY(!plan)) return err;

    const int rc = spline_interp_multi_impl(plan, data_y, out_x, out_y,
                                            out_size, num_datasets);
    spline_plan_free(plan);
    return rc;
}

/* =========================================================================
 * spline_interp_multi_complex
 *
 * Interpolate num_datasets complex128 y-vectors sharing the same x-grid.
 * Each complex double is stored as two consecutive doubles (re, im) —
 * matching NumPy complex128 / C99 double _Complex layout.
 *
 * data_y[d] points to data_size pairs of (re,im) = 2*data_size doubles
 * out_y[d]  points to out_size  pairs of (re,im) = 2*out_size  doubles
 * ====================================================================== */
int spline_interp_multi_complex(const long data_size, const long out_size,
                                const long num_datasets,
                                const double *restrict data_x,
                                const double *const *restrict data_y,
                                const double *restrict out_x,
                                double *const *restrict out_y)
{
    if (out_size <= 0) return SPLINE_ERR_OUT_SIZE;
    if (num_datasets <= 0) return SPLINE_ERR_NUM_DS;

    int err = SPLINE_OK;
    spline_plan *plan = spline_prepare(data_x, data_size, &err);
    if (SPLINE_UNLIKELY(!plan)) return err;

    const long   n        = plan->n;
    const long   ni       = n - 2;
    const double *restrict h        = plan->h;
    const double *restrict inv_h    = plan->inv_h;
    const double *restrict diag_fac = plan->diag;

    /* Allocation: 2*n*num_datasets for c arrays (re+im) + 2*ni for rhs scratch + idx buf */
    const size_t c_sz   = (size_t)(2 * n) * (size_t)num_datasets * sizeof(double);
    const size_t rhs_sz = (size_t)(2 * ni) * sizeof(double);
    const size_t idx_sz = (size_t)out_size * sizeof(int);
    unsigned char *mem = (unsigned char *)malloc(c_sz + rhs_sz + idx_sz);
    if (SPLINE_UNLIKELY(!mem)) { spline_plan_free(plan); return SPLINE_ERR_OOM; }

    double *restrict c_all   = (double *)mem;
    double *restrict rhs_buf = (double *)(mem + c_sz);
    int    *restrict idx_buf = (int *)(mem + c_sz + rhs_sz);

    /* -------------------------------------------------------------------
     * Phase 1: Tridiagonal solve for c_re and c_im (one per dataset).
     * The tridiagonal matrix depends only on x-grid (purely real).
     * We solve two independent RHS for real and imag parts.
     * ------------------------------------------------------------------- */
    for (long d = 0; d < num_datasets; ++d) {
        const double *restrict y_interleaved = data_y[d]; /* re0,im0,re1,im1,... */
        double *restrict c_re = c_all + d * (2 * n);
        double *restrict c_im = c_re + n;
        double *restrict rhs_re = rhs_buf;
        double *restrict rhs_im = rhs_buf + ni;

        c_re[0] = 0.0; c_re[n - 1] = 0.0;
        c_im[0] = 0.0; c_im[n - 1] = 0.0;

        /* k=0, i=1 */
        {
            const double y0_re = y_interleaved[0*2], y1_re = y_interleaved[1*2], y2_re = y_interleaved[2*2];
            const double y0_im = y_interleaved[0*2+1], y1_im = y_interleaved[1*2+1], y2_im = y_interleaved[2*2+1];
            rhs_re[0] = 6.0 * SPLINE_FMA( y2_re, inv_h[1],
                                 SPLINE_FMA(-y1_re, inv_h[1] + inv_h[0],
                                             y0_re * inv_h[0]));
            rhs_im[0] = 6.0 * SPLINE_FMA( y2_im, inv_h[1],
                                 SPLINE_FMA(-y1_im, inv_h[1] + inv_h[0],
                                             y0_im * inv_h[0]));
        }

        /* k=1..ni-1, i=2..n-2 */
        for (long k = 1; k < ni; ++k) {
            const long i = k + 1;
            const double yi1_re = y_interleaved[(i+1)*2], yi_re = y_interleaved[i*2], yim1_re = y_interleaved[(i-1)*2];
            const double yi1_im = y_interleaved[(i+1)*2+1], yi_im = y_interleaved[i*2+1], yim1_im = y_interleaved[(i-1)*2+1];
            const double raw_re = 6.0 * SPLINE_FMA( yi1_re, inv_h[i],
                                          SPLINE_FMA(-yi_re, inv_h[i] + inv_h[i-1],
                                                      yim1_re * inv_h[i-1]));
            const double raw_im = 6.0 * SPLINE_FMA( yi1_im, inv_h[i],
                                          SPLINE_FMA(-yi_im, inv_h[i] + inv_h[i-1],
                                                      yim1_im * inv_h[i-1]));
            const double factor = h[k] / diag_fac[k-1];
            rhs_re[k] = raw_re - factor * rhs_re[k-1];
            rhs_im[k] = raw_im - factor * rhs_im[k-1];
        }

        /* Back-substitution */
        c_re[n-2] = rhs_re[ni-1] / diag_fac[ni-1];
        c_im[n-2] = rhs_im[ni-1] / diag_fac[ni-1];

        for (long k = ni-2; k >= 0; --k) {
            const long i = k + 1;
            c_re[i] = (rhs_re[k] - h[k+1] * c_re[i+1]) / diag_fac[k];
            c_im[i] = (rhs_im[k] - h[k+1] * c_im[i+1]) / diag_fac[k];
        }
    }

    /* -------------------------------------------------------------------
     * Phase 2a: Precompute interval indices for all output points.
     * ------------------------------------------------------------------- */
    {
        int prev_idx = 0;
        for (long j = 0; j < out_size; ++j) {
            const int idx = hunt(data_x, (int)n, out_x[j], prev_idx);
            if (SPLINE_UNLIKELY(idx < 0)) {
                free(mem);
                spline_plan_free(plan);
                return SPLINE_ERR_BOUNDS;
            }
            idx_buf[j] = idx;
            prev_idx = idx;
        }
    }

    /* -------------------------------------------------------------------
     * Phase 2b: Evaluate spline — d-outer for cache locality on c/y rows,
     *           j-inner for stride-1 writes, interleaved complex output.
     * ------------------------------------------------------------------- */
    for (long d = 0; d < num_datasets; ++d) {
        const double *restrict y = data_y[d];
        const double *restrict c_re = c_all + d * (2 * n);
        const double *restrict c_im = c_re + n;
        double *restrict out_row = out_y[d];

#if defined(__clang__)
#  pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#  pragma GCC ivdep
#endif
        for (long j = 0; j < out_size; ++j) {
            const int    idx     = idx_buf[j];
            const double t       = out_x[j] - data_x[idx];
            const double inv_hi  = inv_h[idx];
            const double hi_inv6 = h[idx] * (1.0 / 6.0);

            /* Real part */
            const double ci_re  = c_re[idx],  ci1_re = c_re[idx + 1];
            const double d_re   = (ci1_re - ci_re) * inv_hi * (1.0/6.0);
            const double y_re_i = y[idx*2],   y_re_i1 = y[(idx+1)*2];
            const double b_re   = SPLINE_FMA(-hi_inv6,
                                              SPLINE_FMA(2.0, ci_re, ci1_re),
                                              (y_re_i1 - y_re_i) * inv_hi);

            /* Imaginary part */
            const double ci_im  = c_im[idx],  ci1_im = c_im[idx + 1];
            const double d_im   = (ci1_im - ci_im) * inv_hi * (1.0/6.0);
            const double y_im_i = y[idx*2+1], y_im_i1 = y[(idx+1)*2+1];
            const double b_im   = SPLINE_FMA(-hi_inv6,
                                              SPLINE_FMA(2.0, ci_im, ci1_im),
                                              (y_im_i1 - y_im_i) * inv_hi);

            out_row[j*2]     = SPLINE_FMA(t, SPLINE_FMA(t, SPLINE_FMA(t, d_re, ci_re*0.5), b_re), y_re_i);
            out_row[j*2 + 1] = SPLINE_FMA(t, SPLINE_FMA(t, SPLINE_FMA(t, d_im, ci_im*0.5), b_im), y_im_i);
        }
    }

    free(mem);
    spline_plan_free(plan);
    return SPLINE_OK;
}
