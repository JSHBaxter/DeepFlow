/*

This is a collection of the CPU kernels specific to the Augmented Lagrangian maximum a posteriori solver.
This is to make these functions easier to find and give a bit more flexibility with naming.

Note that some functions may have multiple signatures. This is for handling data of different dimensionality.
Many functions operate point-wise and thus only one signature is needed regardless of image dimensionality.

*/

#ifndef GPU_KERNELS_AUGLAG
#define GPU_KERNELS_AUGLAG

#include <cuda.h>
#include <cuda_runtime.h>
#include "algorithm.h"

//Utilities for augmented lagrangian calculation
void init_flows_potts(const CUDA_DEVICE& dev, const float* data, float* ps, float* pt, float* u, const int n_s, const int n_c);
void init_flows_binary(const CUDA_DEVICE& dev, const float* data, float* ps, float* pt, float* u, const int n_s);
void update_source_flows(const CUDA_DEVICE& dev, float* ps, const float* pt, const float* div, const float* u, float icc, const int n_c, const int n_s);
void update_sink_flows(const CUDA_DEVICE& dev, const float* ps, float* pt, const float* div, const float* u, const float* d, float icc, const int n_c, const int n_s);
void update_multiplier(const CUDA_DEVICE& dev, const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int n_c, const int n_s);
void update_source_sink_multiplier_potts(const CUDA_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s);
void update_source_sink_multiplier_binary(const CUDA_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s);
void update_multiplier_hmf(const CUDA_DEVICE& dev, float* const* const ps_ind, const float* div, const float* pt, float* u, float* erru, const int n_s, const int n_c, const float cc);
void find_min_constraint(const CUDA_DEVICE& dev, float* output, const float* neg_constraint, const int n_c, const int n_s);
void calc_capacity_potts(const CUDA_DEVICE& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void calc_capacity_binary(const CUDA_DEVICE& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau);
void calc_capacity_potts_source_separate(const CUDA_DEVICE& dev, float* g, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void calc_capacity_hmf(const CUDA_DEVICE& dev, float* g, float* const* const ps_ind, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void update_spatial_flows(const CUDA_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const int dim, const int* const n, const int n_c);
void update_spatial_star_flows(const CUDA_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const float *const *const l, const int dim, const int* const n, const int n_c);
void update_flow_hmf(const CUDA_DEVICE& dev, float** g_ind, float* g_s, float* g, float** ps_ind, float* ps, float* pt, const float* div, const float* u, const float icc, const int* p_c, const int s_c, const int n_s, const int n_c);
void divide_out_and_store_hmf(const CUDA_DEVICE& dev, const float* g_s, const float* g, float* ps, float* pt, const int* p_c, const int s_c, const int n_s, const int n_c);
void prep_flow_hmf(const CUDA_DEVICE& dev, float* g, float* const* const ps_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c);
void compute_parents_flow_hmf(const CUDA_DEVICE& dev, float** g_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c);

#endif