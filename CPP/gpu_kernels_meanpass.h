/*

This is a collection of the CPU kernels specific to the mean field estimation based marginal solver.
This is to make these functions easier to find and give a bit more flexibility with naming.

Note that some functions may have multiple signatures. This is for handling data of different dimensionality.
Many functions operate point-wise and thus only one signature is needed regardless of image dimensionality.

*/

#ifndef GPU_KERNELS_MEANPASS
#define GPU_KERNELS_MEANPASS

#include <cuda.h>
#include <cuda_runtime.h>
#include "algorithm.h"

//Functions specifically for the mean field forward calculation
void change_to_diff(const CUDA_DEVICE& dev, float* transfer, float* diff, const int n_s, const float tau);
void get_effective_reg(const CUDA_DEVICE& dev, float* const r_eff, const float* const u_b, const float *const *const r, const int dim, const int* const n, const int n_c);

//Functions specifically for helping with mean field gradient calculation
void populate_reg_mean_gradients_and_add(const CUDA_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau);
void get_gradient_for_u(const CUDA_DEVICE & dev, const float* dy, const float *const *const r, float* const du, const int dim, const int* const n, const int n_c, const float tau);

#endif