/*

The purpose of this collection of functions is to abstract the details of GPGPU implementation in CUDA away
from the mathematical logic of the solution algorithm.

Note that some functions may have multiple signatures. This is for handling data of different dimensionality.
Many functions operate point-wise and thus only one signature is needed regardless of image dimensionality.

*/

#ifndef GPU_KERNELS
#define GPU_KERNELS

#include <cuda.h>
#include <cuda_runtime.h>
#include "algorithm.h"

//General GPU utilities
void let_device_catchup(const CUDA_DEVICE& dev);
void* allocate_on_gpu(const CUDA_DEVICE& dev, size_t amount);
void deallocate_on_gpu(const CUDA_DEVICE& dev, void* ptr);
void send_to_gpu(const CUDA_DEVICE& dev, const void* source, void* dest, size_t amount);
void get_from_gpu(const CUDA_DEVICE& dev, const void* source, void* dest, size_t amount);
void print_buffer(const CUDA_DEVICE& dev, const float* buffer, const int n_s);

void clear_buffer(const CUDA_DEVICE& dev, float* buffer, const int size);
void clear_buffer(const CUDA_DEVICE& dev, int* buffer, const int size);
void set_buffer(const CUDA_DEVICE& dev, float* buffer, const float number, const int size);

void copy_buffer(const CUDA_DEVICE& dev, const float* source, float* dest, const int n_s);
void copy_buffer_clean(const CUDA_DEVICE& dev, const float* source, float* dest, const int n_s);
void copy_buffer_clip(const CUDA_DEVICE& dev, const float* source, float* dest, const int n_s, float value);
void rep_buffer(const CUDA_DEVICE& dev, const float* input, float* output, const int n_c, const int n_s);

void inc_buffer(const CUDA_DEVICE& dev, const float* inc, float* acc, const int n_s);
void inc_buffer(const CUDA_DEVICE& dev, const float inc, float* acc, const int n_s);
void ninc_buffer(const CUDA_DEVICE& dev, const float* inc, float* acc, const int n_s);
void inc_mult_buffer(const CUDA_DEVICE& dev, const float* inc, float* acc, const int n_s, const float multi);
void inc2_mult_buffer(const CUDA_DEVICE& dev, const float* inc_m1, const float* inc_m2, float* acc, const int n_s, const float multi);
void inc_inc_minc_buffer(const CUDA_DEVICE& dev, const float* inc1, const float* inc2, const float* minc, const float multi, float* acc, const int n_s);
void m_inc_inc_ninc_minc_buffer(const CUDA_DEVICE& dev, const float* inc1, const float* inc2, const float* ninc, const float* minc, const float multi_end, const float multi_all, float* acc, const int n_s);

void div_buffer(const CUDA_DEVICE& dev, const float number, float* res, const int n_s);
void div_buffer(const CUDA_DEVICE& dev, const float* div, float* res, const int n_s);
void mult_buffer(const CUDA_DEVICE& dev, const float mult, float* res, const int n_s);
void mult_buffer(const CUDA_DEVICE& dev, const float mult, const float* input, float* res, const int n_s);
void log_buffer(const CUDA_DEVICE& dev, const float* in, float* out, const int n_s);

void mark_neg_equal(const CUDA_DEVICE& dev, const float* buffer_s, const float* buffer_l, float* u, const int n_s, const int n_c);

void exp_and_inc_buffer(const CUDA_DEVICE& dev, const float* max, float* cost, float* acc, const int n_s);

void add_store_then_max_buffer(const CUDA_DEVICE& dev, const float* comp1, const float* comp2, float* store, float* res, const int n_s);
void add_then_store(const CUDA_DEVICE& dev, const float* addend1, const float* addend2, float* sum, const int size);
void add_then_store(const CUDA_DEVICE& dev, const float* addend1, const float* addend2, float* sum1, float* sum2, const int size);

float max_of_buffer(const CUDA_DEVICE& dev, const float* buffer, const int n_s);
float mean_of_buffer(const CUDA_DEVICE& dev, const float* buffer, const int n_s);
float spat_max_of_buffer(const CUDA_DEVICE& dev, const float* buffer, const int n_s, const int n_c);

void aggregate_bottom_up(const CUDA_DEVICE& dev, float** p_ind, float* buffer, const float* org, const int n_s, const int n_c, const int n_r);

//Utilities for augmented lagrangian calculation
void update_source_flows(const CUDA_DEVICE& dev, float* ps, const float* pt, const float* div, const float* u, float icc, const int n_c, const int n_s);
void update_sink_flows(const CUDA_DEVICE& dev, const float* ps, float* pt, const float* div, const float* u, const float* d, float icc, const int n_c, const int n_s);
void update_multiplier(const CUDA_DEVICE& dev, const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int n_c, const int n_s);
void update_source_sink_multiplier_potts(const CUDA_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s);
void update_source_sink_multiplier_binary(const CUDA_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s);
void update_multiplier_hmf(const CUDA_DEVICE& dev, float* const* const ps_ind, const float* div, const float* pt, float* u, float* erru, const int n_s, const int n_c, const float cc);
void find_min_constraint(const CUDA_DEVICE& dev, float* output, const float* neg_constraint, const int n_c, const int n_s);
void init_flows_potts(const CUDA_DEVICE& dev, const float* data, float* ps, float* pt, float* u, const int n_s, const int n_c);
void init_flows_binary(const CUDA_DEVICE& dev, const float* data, float* ps, float* pt, float* u, const int n_s);
void calc_capacity_potts(const CUDA_DEVICE& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void calc_capacity_binary(const CUDA_DEVICE& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau);
void calc_capacity_potts_source_separate(const CUDA_DEVICE& dev, float* g, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void calc_capacity_hmf(const CUDA_DEVICE& dev, float* g, float* const* const ps_ind, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void update_spatial_flows(const CUDA_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const int dim, const int* const n, const int n_c);
void update_spatial_star_flows(const CUDA_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const float *const *const l, const int dim, const int* const n, const int n_c);
void abs_constrain(const CUDA_DEVICE& dev, float* buffer, const float* constrain, const int n_s);
void max_neg_constrain(const CUDA_DEVICE& dev, float* buffer, const float* constrain, const int n_s);
void binary_constrain(const CUDA_DEVICE& dev, float* buffer, const int n_s);

void update_flow_hmf(const CUDA_DEVICE& dev, float** g_ind, float* g_s, float* g, float** ps_ind, float* ps, float* pt, const float* div, const float* u, const float icc, const int* p_c, const int s_c, const int n_s, const int n_c);
void divide_out_and_store_hmf(const CUDA_DEVICE& dev, const float* g_s, const float* g, float* ps, float* pt, const int* p_c, const int s_c, const int n_s, const int n_c);
void prep_flow_hmf(const CUDA_DEVICE& dev, float* g, float* const* const ps_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c);
void compute_parents_flow_hmf(const CUDA_DEVICE& dev, float** g_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c);

//Functions specifically for the mean field forward calculation
void softmax(const CUDA_DEVICE& dev, const float* e1, const float* e2, float* u, const int n_s, const int n_c);
void softmax(const CUDA_DEVICE& dev, const float* bufferrin, float* bufferout, const int n_s, const int n_c);
void neg_softmax(const CUDA_DEVICE& dev, const float* e, float* u, const int n_s, const int n_c);
void sigmoid(const CUDA_DEVICE& dev, const float* e1, const float* e2, float* u, const int n_s);
void exp(const CUDA_DEVICE& dev, const float* e1, float* u, const int n_s);
void change_to_diff(const CUDA_DEVICE& dev, float* transfer, float* diff, const int n_s, const float tau);
void get_effective_reg(const CUDA_DEVICE& dev, float* const r_eff, const float* const u_b, const float *const *const r, const int dim, const int* const n, const int n_c);
void parity_mask(const CUDA_DEVICE& dev, float* buffer, const int dim, const int* const n, const int n_c, const int parity);
void parity_mask(const CUDA_DEVICE& dev, float* buffer, const float* other, const int dim, const int* const n, const int n_c, const int parity);



//Functions specifically for helping with mean field gradient calculation
void untangle_softmax(const CUDA_DEVICE& dev, const float* du_i, const float* u, float* loc, const int n_s, const int n_c);
void untangle_sigmoid(const CUDA_DEVICE& dev, const float* du_i, const float* u, float* loc, const int n_s);
void populate_reg_mean_gradients_and_add(const CUDA_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau);
void get_gradient_for_u(const CUDA_DEVICE & dev, const float* dy, const float *const *const r, float* const du, const int dim, const int* const n, const int n_c, const float tau);

#endif // GPU_KERNELS
