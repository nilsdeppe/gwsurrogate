#ifndef FORNBERG_FD_H
#define FORNBERG_FD_H

#define MAX_STENCIL_SIZE 8
#define MAX_DERIVATIVE_ORDER 2

typedef struct {
    double first;
    double second;
} Derivatives;

Derivatives fd2(const double *restrict y,
                const double *restrict x,
                const int eval_index,
                const int length);

Derivatives fd4(const double *restrict y,
                const double *restrict x,
                const int eval_index,
                const int length);

Derivatives fd6(const double *restrict y,
                const double *restrict x,
                const int eval_index,
                const int length);

#endif /* FORNBERG_FD_H */
