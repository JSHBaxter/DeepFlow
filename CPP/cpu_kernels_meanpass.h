/*

This is a collection of the CPU kernels specific to the mean field estimation based marginal solver.
This is to make these functions easier to find and give a bit more flexibility with naming.

Note that some functions may have multiple signatures. This is for handling data of different dimensionality.
Many functions operate point-wise and thus only one signature is needed regardless of image dimensionality.

*/

#ifndef CPU_KERNELS_MEANPASS
#define CPU_KERNELS_MEANPASS

#include "algorithm.h"

//Functions specifically for the mean field forward calculation
void change_to_diff(const CPU_DEVICE & dev,float* buffer, float* update, const int n_s, const float alpha);
void get_effective_reg(const CPU_DEVICE & dev, float* const r_eff, const float* const u, const float *const *const r, const int dim, const int* const n, const int n_c);

//Functions for mean field backward calculation
void untangle_softmax(const CPU_DEVICE & dev,const float* g, const float* u, float* dy, const int n_s, const int n_c);
void untangle_sigmoid(const CPU_DEVICE & dev,const float* g, const float* u, float* dy, const int n_s);
void get_gradient_for_u(const CPU_DEVICE & dev,const float* dy, const float *const *const r, float* const du, const int dim, const int* const n, const int n_c, const float tau);
void populate_reg_mean_gradients(const CPU_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c);
void populate_reg_mean_gradients_and_add(const CPU_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau);

//void get_reg_gradients(const CPU_DEVICE & dev,const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau);


#endif