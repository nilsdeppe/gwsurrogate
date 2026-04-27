#include <assert.h>
#include "Fornberg.h"

/* #define MAX_DERIVATIVE_ORDER 2 */
/* #define MAX_STENCIL_SIZE 8 */

/* struct Derivatives { */
/*   double first; */
/*   double second; */
/* }; */

/*
 * Compute finite difference weights using Fornberg's algorithm.
 *
 * Reference:
 *   B. Fornberg, "Generation of Finite Difference Formulas on Arbitrarily
 *   Spaced Grids", Mathematics of Computation, 51(184), pp. 699-706, 1988.
 *
 * The algorithm computes weights for approximating derivatives of order
 * 0, 1, ..., highest_derivative_order at the point grid[eval_index],
 * using the stencil points grid[stencil_start .. stencil_start + stencil_size - 1].
 *
 * Output:
 *   weights[m][j] = weight for the j-th stencil point in the m-th derivative
 *                   approximation, where j is relative to stencil_start.
 *
 * Preconditions:
 *   - All stencil points must be distinct (no duplicate x-values).
 *   - stencil_size <= MAX_STENCIL_SIZE
 *   - highest_derivative_order <= MAX_DERIVATIVE_ORDER
 */
static void fornberg_weights(
    const double *restrict grid,
    const int eval_index,
    const int stencil_start,
    const int stencil_size,
    const int highest_derivative_order,
    double weights[MAX_DERIVATIVE_ORDER + 1][MAX_STENCIL_SIZE])
{
    assert(stencil_size <= MAX_STENCIL_SIZE);
    assert(highest_derivative_order <= MAX_DERIVATIVE_ORDER);

    const double z = grid[eval_index];

    /* Zero out the weights array */
    for (int m = 0; m <= highest_derivative_order; m++) {
        for (int j = 0; j < stencil_size; j++) {
            weights[m][j] = 0.0;
        }
    }

    /*
     * Fornberg's algorithm, Table 1 from the 1988 paper.
     *
     * Notation mapping (paper -> code):
     *   x(i)  -> grid[stencil_start + i]
     *   x(0)  -> z  (the evaluation point, i.e., grid[eval_index])
     *   c1    -> c1
     *   c2    -> c2 (accumulated product)
     *   c3    -> c3 (difference x_i - x_j)
     *   c4    -> prev_x_minus_z (x_{j_prev} - z, i.e., x_{i-1} - z when j == i-1)
     *   delta -> weights
     *   n     -> stencil_size - 1
     *   M     -> highest_derivative_order
     */
    weights[0][0] = 1.0;
    double c1 = 1.0;

    for (int i = 1; i < stencil_size; i++) {
        const double xi = grid[stencil_start + i];

        /* mn = min(i, highest_derivative_order) */
        const int mn = (i < highest_derivative_order)
                     ? i : highest_derivative_order;

        double c2 = 1.0;

        for (int j = 0; j < i; j++) {
            const double xj = grid[stencil_start + j];
            const double c3 = xi - xj;
            c2 *= c3;

            /*
             * When j == i-1 (last iteration of the j loop), compute
             * weights for the NEW point i BEFORE overwriting point i-1.
             * This is critical: weights[m][i-1] is needed as input.
             */
            if (j == i - 1) {
                for (int m = mn; m >= 1; m--) {
                    weights[m][i] = c1 / c2
                        * (m * weights[m - 1][i - 1]
                           - (xj - z) * weights[m][i - 1]);
                }
                weights[0][i] = -c1 * (xj - z)
                                * weights[0][i - 1] / c2;
            }

            /* Now safe to overwrite weights for point j */
            for (int m = mn; m >= 1; m--) {
                weights[m][j] = ((xi - z) * weights[m][j]
                                 - m * weights[m - 1][j]) / c3;
            }
            weights[0][j] = (xi - z) * weights[0][j] / c3;
        }

        c1 = c2;
    }
}

/*
 * Apply a finite difference stencil to compute the first and second
 * derivatives of y at grid[eval_index].
 *
 * This function:
 *   1. Determines the stencil start index via clamping.
 *   2. Calls fornberg_weights to get the FD weights.
 *   3. Dot-products the weights with the y-values.
 */
static Derivatives apply_stencil(
    const double *restrict y,
    const double *restrict grid,
    const int eval_index,
    const int stencil_size,
    const int length)
{
    /* Clamp: center the stencil as much as possible, but keep it in bounds */
    const int half = (stencil_size - 1) / 2;
    int stencil_start = eval_index - half;
    if (stencil_start < 0) {
        stencil_start = 0;
    } else if (stencil_start > length - stencil_size) {
        stencil_start = length - stencil_size;
    }

    double weights[MAX_DERIVATIVE_ORDER + 1][MAX_STENCIL_SIZE];
    fornberg_weights(grid, eval_index, stencil_start, stencil_size,
                     MAX_DERIVATIVE_ORDER, weights);

    /* Dot product: sum weights[m][j] * y[stencil_start + j] */
    double first_derivative = 0.0;
    double second_derivative = 0.0;
    for (int j = 0; j < stencil_size; j++) {
        const double yj = y[stencil_start + j];
        first_derivative  += weights[1][j] * yj;
        second_derivative += weights[2][j] * yj;
    }

    Derivatives result;
    result.first  = first_derivative;
    result.second = second_derivative;
    return result;
}

/*
 * 2nd-order accurate finite differences (4-point stencil).
 * Guarantees at least 2nd-order accuracy for both 1st and 2nd derivatives.
 */
Derivatives fd2(const double *restrict y,
                const double *restrict x,
                const int eval_index,
                const int length)
{
    const int stencil_size = 4;
    assert(eval_index >= 0 && eval_index < length);
    assert(length >= stencil_size);
    return apply_stencil(y, x, eval_index, stencil_size, length);
}

/*
 * 4th-order accurate finite differences (6-point stencil).
 * Guarantees at least 4th-order accuracy for both 1st and 2nd derivatives.
 */
Derivatives fd4(const double *restrict y,
                const double *restrict x,
                const int eval_index,
                const int length)
{
    const int stencil_size = 6;
    assert(eval_index >= 0 && eval_index < length);
    assert(length >= stencil_size);
    return apply_stencil(y, x, eval_index, stencil_size, length);
}

/*
 * 6th-order accurate finite differences (8-point stencil).
 * Guarantees at least 6th-order accuracy for both 1st and 2nd derivatives.
 */
Derivatives fd6(const double *restrict y,
                const double *restrict x,
                const int eval_index,
                const int length)
{
    const int stencil_size = 8;
    assert(eval_index >= 0 && eval_index < length);
    assert(length >= stencil_size);
    return apply_stencil(y, x, eval_index, stencil_size, length);
}

double quintic_hermite_fd4(double const*const restrict y,
                           double  const  *const restrict x,
                           const int i,
                           const int length) {

}
