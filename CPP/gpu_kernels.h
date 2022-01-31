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

//General GPU utilities
void* allocate_on_gpu(const cudaStream_t& dev, size_t amount);
void deallocate_on_gpu(const cudaStream_t& dev, void* ptr);
void send_to_gpu(const cudaStream_t& dev, const void* source, void* dest, size_t amount);
void get_from_gpu(const cudaStream_t& dev, const void* source, void* dest, size_t amount);
void print_buffer(const cudaStream_t& dev, const float* buffer, const int n_s);

void clear_buffer(const cudaStream_t& dev, float* buffer, const int size);
void clear_buffer(const cudaStream_t& dev, int* buffer, const int size);
void set_buffer(const cudaStream_t& dev, float* buffer, const float number, const int size);

void copy_buffer(const cudaStream_t& dev, const float* source, float* dest, const int n_s);
void copy_buffer_clean(const cudaStream_t& dev, const float* source, float* dest, const int n_s);
void copy_buffer_clip(const cudaStream_t& dev, const float* source, float* dest, const int n_s, float value);
void rep_buffer(const cudaStream_t& dev, const float* input, float* output, const int n_c, const int n_s);

void inc_buffer(const cudaStream_t& dev, const float* inc, float* acc, const int n_s);
void ninc_buffer(const cudaStream_t& dev, const float* inc, float* acc, const int n_s);
void inc_mult_buffer(const cudaStream_t& dev, const float* inc, float* acc, const int n_s, const float multi);
void inc2_mult_buffer(const cudaStream_t& dev, const float* inc_m1, const float* inc_m2, float* acc, const int n_s, const float multi);
void inc_inc_minc_buffer(const cudaStream_t& dev, const float* inc1, const float* inc2, const float* minc, const float multi, float* acc, const int n_s);

void div_buffer(const cudaStream_t& dev, const float number, float* res, const int n_s);
void div_buffer(const cudaStream_t& dev, const float* div, float* res, const int n_s);
void mult_buffer(const cudaStream_t& dev, const float mult, float* res, const int n_s);
void mult_buffer(const cudaStream_t& dev, const float mult, const float* input, float* res, const int n_s);
void log_buffer(const cudaStream_t& dev, const float* in, float* out, const int n_s);

void mark_neg_equal(const cudaStream_t& dev, const float* buffer_s, const float* buffer_l, float* u, const int n_s, const int n_c);

void exp_and_inc_buffer(const cudaStream_t& dev, const float* max, float* cost, float* acc, const int n_s);

void add_store_then_max_buffer(const cudaStream_t& dev, const float* comp1, const float* comp2, float* store, float* res, const int n_s);
void add_then_store(const cudaStream_t& dev, const float* addend1, const float* addend2, float* sum, const int size);
void add_then_store_2(const cudaStream_t& dev, const float* addend1, const float* addend2, float* sum1, float* sum2, const int size);

float max_of_buffer(const cudaStream_t& dev, const float* buffer, const int n_s);

void aggregate_bottom_up(const cudaStream_t& dev, float** p_ind, float* buffer, const float* org, const int n_s, const int n_c, const int n_r);

//Utilities for augmented lagrangian calculation
void update_source_flows(const cudaStream_t& dev, float* ps, const float* pt, const float* div, const float* u, float icc, const int n_c, const int n_s);
void update_sink_flows(const cudaStream_t& dev, const float* ps, float* pt, const float* div, const float* u, const float* d, float icc, const int n_c, const int n_s);
void update_multiplier(const cudaStream_t& dev, const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int n_c, const int n_s);
void update_source_sink_multiplier_potts(const cudaStream_t& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s);
void update_source_sink_multiplier_binary(const cudaStream_t& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s);
void update_multiplier_hmf(const cudaStream_t& dev, float* const* const ps_ind, const float* div, const float* pt, float* u, float* erru, const int n_s, const int n_c, const float cc);
void find_min_constraint(const cudaStream_t& dev, float* output, const float* neg_constraint, const int n_c, const int n_s);
void init_flows_binary(const cudaStream_t& dev, const float* data, float* ps, float* pt, const int n_s);
void calc_capacity_potts(const cudaStream_t& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void calc_capacity_binary(const cudaStream_t& dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau);
void calc_capacity_potts_source_separate(const cudaStream_t& dev, float* g, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void calc_capacity_hmf(const cudaStream_t& dev, float* g, float* const* const ps_ind, const float* div, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void update_spatial_flows(const cudaStream_t& dev, const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_t);
void update_spatial_flows(const cudaStream_t& dev, const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_x, const int n_y, const int n_t);
void update_spatial_flows(const cudaStream_t& dev, const float* g, float* div, float* px, const float* rx, const int n_x, const int n_t);
void abs_constrain(const cudaStream_t& dev, float* buffer, const float* constrain, const int n_s);
void max_neg_constrain(const cudaStream_t& dev, float* buffer, const float* constrain, const int n_s);
void binary_constrain(const cudaStream_t& dev, float* buffer, const int n_s);

void update_flow_hmf(const cudaStream_t& dev, float** g_ind, float* g_s, float* g, float** ps_ind, float* ps, float* pt, const float* div, const float* u, const float icc, const int* p_c, const int s_c, const int n_s, const int n_c);
void divide_out_and_store_hmf(const cudaStream_t& dev, const float* g_s, const float* g, float* ps, float* pt, const int* p_c, const int s_c, const int n_s, const int n_c);
void prep_flow_hmf(const cudaStream_t& dev, float* g, float* const* const ps_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c);
void compute_parents_flow_hmf(const cudaStream_t& dev, float** g_ind, const float* pt, const float* div, const float* u, const float icc, const int n_s, const int n_c);

void populate_data_gradient(const cudaStream_t& dev, const float* g, const float* u, float* output, const int n_s);

void populate_reg_gradients(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c);


//Functions specifically for the mean field forward calculation
void softmax(const cudaStream_t& dev, const float* e1, const float* e2, float* u, const int n_s, const int n_c);
void neg_softmax(const cudaStream_t& dev, const float* e, float* u, const int n_s, const int n_c);
void sigmoid(const cudaStream_t& dev, const float* e1, const float* e2, float* u, const int n_s);
void exp(const cudaStream_t& dev, const float* e1, float* u, const int n_s);
void change_to_diff(const cudaStream_t& dev, float* transfer, float* diff, const int n_s, const float tau);
void get_effective_reg(const cudaStream_t& dev, float* r_eff, const float* u_b, const float* rx_b, const float* ry_b, const float* rz_b, const int n_x, const int n_y, const int n_z, const int n_c);
void get_effective_reg(const cudaStream_t& dev, float* r_eff, const float* u_b, const float* rx_b, const float* ry_b, const int n_x, const int n_y, const int n_c);
void get_effective_reg(const cudaStream_t& dev, float* r_eff, const float* u_b, const float* rx_b, const int n_x, const int n_c);
void parity_mask(const cudaStream_t& dev, float* buffer, const int n_x, const int n_y, const int n_z, const int n_c, const int parity);
void parity_mask(const cudaStream_t& dev, float* buffer, const int n_x, const int n_y, const int n_c, const int parity);
void parity_mask(const cudaStream_t& dev, float* buffer, const int n_x, const int n_c, const int parity);
void parity_mask(const cudaStream_t& dev, float* buffer, const float* other, const int n_x, const int n_y, const int n_z, const int n_c, const int parity);
void parity_mask(const cudaStream_t& dev, float* buffer, const float* other, const int n_x, const int n_y, const int n_c, const int parity);
void parity_mask(const cudaStream_t& dev, float* buffer, const float* other, const int n_x, const int n_c, const int parity);



//Functions specifically for helping with mean field gradient calculation
void softmax_gradient(const cudaStream_t& dev, const float* g, const float* u, float* g_d, const int n_s, const int n_c);
void untangle_softmax(const cudaStream_t& dev, const float* du_i, const float* u, float* loc, const int n_s, const int n_c);
void untangle_sigmoid(const cudaStream_t& dev, const float* du_i, const float* u, float* loc, const int n_s);

void populate_reg_mean_gradients(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c);
void populate_reg_mean_gradients(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c);
void populate_reg_mean_gradients(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, const int n_x, const int n_c);
void populate_reg_mean_gradients_and_add(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau);
void populate_reg_mean_gradients_and_add(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau);
void populate_reg_mean_gradients_and_add(const cudaStream_t& dev, const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau);

void get_gradient_for_u(const cudaStream_t& dev, const float* dy, float* du, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau);
void get_gradient_for_u(const cudaStream_t& dev, const float* dy, float* du, const float* rx, const float* ry, const int n_x, const int n_y, const int n_c, const float tau);
void get_gradient_for_u(const cudaStream_t& dev, const float* dy, float* du, const float* rx, const int n_x, const int n_c, const float tau);

// Misc. functions
void taylor_series_channels_first(const cudaStream_t& dev, const float* input, const float* coeffs, float* output, int n_b, int n_s, int n_c, int n_i);
void taylor_series_channels_last(const cudaStream_t& dev, const float* input, const float* coeffs, float* output, int n_b, int n_s, int n_c, int n_i);
void taylor_series_grad_channels_first(const cudaStream_t& dev, const float* input, const float* coeffs, const float* grad, float* g_input, float* g_coeffs, int n_b, int n_s, int n_c, int n_i);
void taylor_series_grad_channels_last(const cudaStream_t& dev, const float* input, const float* coeffs, const float* grad, float* g_input, float* g_coeffs, int n_b, int n_s, int n_c, int n_i);

#endif // GPU_KERNELS
