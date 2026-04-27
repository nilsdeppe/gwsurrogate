#include "QuinticHermite.h"
#include "Fornberg.h"
#include <assert.h>

/* ──────────────────────────────────────────────
 * Quintic Hermite interpolant state.
 *
 * Caches the current interval index and the first/second derivatives
 * at both endpoints. When advancing to the next interval, the left
 * derivatives become the old right derivatives (free). For arbitrary
 * jumps, both endpoints are recomputed.
 * ────────────────────────────────────────────── */

typedef struct {
    int interval;       /* current interval: [x[interval], x[interval+1]] */
    double d1_left;     /* y'  at x[interval]     */
    double d2_left;     /* y'' at x[interval]     */
    double d1_right;    /* y'  at x[interval + 1] */
    double d2_right;    /* y'' at x[interval + 1] */
} QuinticInterpState;

/* ──────────────────────────────────────────────
 * Function pointer type for the finite difference functions.
 * ────────────────────────────────────────────── */

typedef Derivatives (*fd_func_t)(const double *restrict,
                                 const double *restrict,
                                 const int, const int);

/* ──────────────────────────────────────────────
 * Compute and cache derivatives for interval [x[idx], x[idx+1]].
 * ────────────────────────────────────────────── */

static void compute_interval_derivatives(
    QuinticInterpState *state,
    const int idx,
    const double *const restrict data_x,
    const double *const restrict data_y,
    const int data_size,
    fd_func_t fd_function)
{
    if (state->interval == idx) {
        /* Already cached — nothing to do */
        return;
    }

    if (state->interval >= 0 && idx == state->interval + 1) {
        /*
         * Advancing one interval to the right: the new left endpoint
         * is the old right endpoint. Only compute the new right.
         */
        state->d1_left  = state->d1_right;
        state->d2_left  = state->d2_right;
    } else {
        /* Arbitrary jump: compute left endpoint from scratch */
        const Derivatives d_left = fd_function(data_y, data_x,
                                               idx, data_size);
        state->d1_left  = d_left.first;
        state->d2_left  = d_left.second;
    }

    /* Always compute right endpoint */
    const Derivatives d_right = fd_function(data_y, data_x,
                                            idx + 1, data_size);
    state->d1_right = d_right.first;
    state->d2_right = d_right.second;

    state->interval = idx;
}

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

/* ──────────────────────────────────────────────
 * Evaluate the quintic Hermite interpolant on a single interval.
 *
 * Given the interval [x_L, x_R] with function values and derivatives:
 *   y_L, d1_L (= y'(x_L)), d2_L (= y''(x_L))
 *   y_R, d1_R (= y'(x_R)), d2_R (= y''(x_R))
 *
 * Returns p(x) where p is the unique degree-5 polynomial satisfying
 * the six interpolation conditions.
 *
 * Derivation: Let t = (x - x_L) / h, h = x_R - x_L, so t in [0,1].
 * We express p(t) = sum of 6 Hermite basis functions:
 *
 *   p(t) = y_L  * H00(t) + y_R  * H01(t)
 *        + d1_L * H10(t) + d1_R * H11(t)    [scaled by h]
 *        + d2_L * H20(t) + d2_R * H21(t)    [scaled by h²]
 *
 * where:
 *   H00(t) =  1 - 10t³ + 15t⁴ -  6t⁵
 *   H01(t) =      10t³ - 15t⁴ +  6t⁵
 *   H10(t) = (t -  6t³ +  8t⁴ -  3t⁵) * h
 *   H11(t) = (    -4t³ +  7t⁴ -  3t⁵) * h
 *   H20(t) = (t² -  3t³ +  3t⁴ -  t⁵) * h² / 2
 *   H21(t) = (      t³ -  2t⁴ +  t⁵) * h² / 2
 *
 * These satisfy:
 *   H00(0)=1, H00(1)=0, H00'(0)=0, H00'(1)=0, H00''(0)=0, H00''(1)=0
 *   H01(0)=0, H01(1)=1, ...
 *   H10'(0)/h=1, ...  etc.
 * ────────────────────���───────────────────────── */

static double quintic_hermite_eval(
    const double x,
    const double x_left,
    const double x_right,
    const double y_left,
    const double y_right,
    const double d1_left,
    const double d1_right,
    const double d2_left,
    const double d2_right)
{
    const double h = x_right - x_left;
    const double t = (x - x_left) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double t4 = t3 * t;
    const double t5 = t4 * t;

    /* Hermite basis functions */
    const double h00 = 1.0 - 10.0*t3 + 15.0*t4 -  6.0*t5;
    const double h01 =       10.0*t3 - 15.0*t4 +  6.0*t5;
    const double h10 = t   -  6.0*t3 +  8.0*t4 -  3.0*t5;
    const double h11 =       -4.0*t3 +  7.0*t4 -  3.0*t5;
    const double h20 = t2  -  3.0*t3 +  3.0*t4 -       t5;
    const double h21 =              t3 -  2.0*t4 +       t5;

    return y_left  * h00 + y_right * h01
         + (d1_left * h10 + d1_right * h11) * h
         + (d2_left * h20 + d2_right * h21) * (h * h * 0.5);
}

/* ───────────────────────────────���──────────────
 * Shared implementation for quintic Hermite interpolation.
 *
 * Returns 0 on success, -1 if any out_x is out of range.
 * ────────────────────────────────────────────── */

static int quintic_interp_impl(
    long data_size, long out_size,
    double *data_x_raw, double *data_y_raw,
    double *out_x_raw, double *out_y_raw,
    fd_func_t fd_function)
{
    /* Promote to const restrict internally */
    const double *const restrict data_x = data_x_raw;
    const double *const restrict data_y = data_y_raw;
    const double *const restrict out_x  = out_x_raw;
    double *const restrict out_y        = out_y_raw;

    assert(data_size >= 2);

    /* Initialize the interpolation state (no interval cached yet) */
    QuinticInterpState state;
    state.interval = -1;
    state.d1_left  = 0.0;
    state.d2_left  = 0.0;
    state.d1_right = 0.0;
    state.d2_right = 0.0;

    int guess = 0;

    for (long ii = 0; ii < out_size; ii++) {
        const double x = out_x[ii];

        /* Hunt for the interval containing x */
        const int idx = hunt(data_x, (int)data_size, x, guess);
        if (idx < 0) {
            return -1;  /* out of range — stop immediately */
        }

        /* Update guess for next hunt (locality) */
        guess = idx;

        /* Ensure derivatives are cached for this interval */
        compute_interval_derivatives(&state, idx,
                                     data_x, data_y,
                                     (int)data_size,
                                     fd_function);

        /* Evaluate the quintic Hermite polynomial */
        out_y[ii] = quintic_hermite_eval(
            x,
            data_x[idx], data_x[idx + 1],
            data_y[idx], data_y[idx + 1],
            state.d1_left, state.d1_right,
            state.d2_left, state.d2_right);
    }

    return 0;
}

/* ─���────────────────────────────────────────────
 * Public API: quintic Hermite interpolation with fd2/fd4/fd6.
 *
 * Each function is a drop-in replacement for the GSL cubic spline
 * interpolation, but uses a quintic Hermite interpolant with
 * Fornberg finite difference derivatives.
 *
 * Parameters:
 *   data_size  — number of data points (must be >= stencil size)
 *   out_size   — number of output evaluation points
 *   data_x     — data x-coordinates (must be strictly increasing)
 *   data_y     — data y-values at data_x
 *   out_x      — output x-coordinates to evaluate at
 *   out_y      — output buffer for interpolated y-values
 *
 * Returns:
 *   0 on success
 *  -1 if any out_x value is outside [data_x[0], data_x[data_size-1]]
 *
 * Note: data_size must be at least 4/6/8 for fd2/fd4/fd6 respectively.
 * ────────────────────────────────────────────── */

int quintic_interp_fd2(long data_size, long out_size,
                       double *data_x, double *data_y,
                       double *out_x, double *out_y)
{
    assert(data_size >= 4);
    return quintic_interp_impl(data_size, out_size,
                               data_x, data_y,
                               out_x, out_y, fd2);
}

int quintic_interp_fd4(long data_size, long out_size,
                       double *data_x, double *data_y,
                       double *out_x, double *out_y)
{
    assert(data_size >= 6);
    return quintic_interp_impl(data_size, out_size,
                               data_x, data_y,
                               out_x, out_y, fd4);
}

int quintic_interp_fd6(long data_size, long out_size,
                       double *data_x, double *data_y,
                       double *out_x, double *out_y)
{
    assert(data_size >= 8);
    return quintic_interp_impl(data_size, out_size,
                               data_x, data_y,
                               out_x, out_y, fd6);
}
