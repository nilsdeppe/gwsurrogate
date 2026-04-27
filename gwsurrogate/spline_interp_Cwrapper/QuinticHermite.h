#ifndef QUINTIC_INTERP_H
#define QUINTIC_INTERP_H

int quintic_interp_fd2(long data_size, long out_size,
                       double *data_x, double *data_y,
                       double *out_x, double *out_y);

int quintic_interp_fd4(long data_size, long out_size,
                       double *data_x, double *data_y,
                       double *out_x, double *out_y);

int quintic_interp_fd6(long data_size, long out_size,
                       double *data_x, double *data_y,
                       double *out_x, double *out_y);

#endif /* QUINTIC_INTERP_H */
