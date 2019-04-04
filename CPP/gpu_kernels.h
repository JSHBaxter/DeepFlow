#ifndef GPU_KERNELS
#define GPU_KERNELS

#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op_kernel.h"

void get_from_gpu(const Eigen::GpuDevice& dev, const void* source, void* dest, size_t amount);

void clear_buffer(const Eigen::GpuDevice& dev, float* buffer, const int size);
void clear_buffer(const Eigen::GpuDevice& dev, int* buffer, const int size);


void update_source_flows(const Eigen::GpuDevice& dev, float* ps, const float* pt, const float* div, const float* u, float icc, const int n_c, const int n_s);


void update_sink_flows(const Eigen::GpuDevice& dev, const float* ps, float* pt, const float* div, const float* u, const float* d, float icc, const int n_c, const int n_s);


void update_multiplier(const Eigen::GpuDevice& dev, const float* ps, const float* pt, const float* div, float* u, float* erru, float cc, const int n_c, const int n_s);


void find_min_constraint(const Eigen::GpuDevice& dev, float* output, const float* neg_constraint, const int n_c, const int n_s);


void rep_buffer(const Eigen::GpuDevice& dev, const float* input, float* output, const int n_c, const int n_s);


void calc_divergence(const Eigen::GpuDevice& dev, float* div, const float* px, const float* py, const float* pz, const int n_x, const int n_y, const int n_z, const int n_c);


void update_spatial_flows(const Eigen::GpuDevice& dev, float* g, const float* div, float* px, float* py, float* pz, const float* ps, const float* pt, const float* u, const int n_x, const int n_y, const int n_z, const int n_c, float icc, float tau);


void abs_constrain(const Eigen::GpuDevice& dev, float* buffer, const float* constrain, const int n_s);


void binary_constrain(const Eigen::GpuDevice& dev, float* buffer, const int n_s);


float max_of_buffer(const Eigen::GpuDevice& dev, float* buffer, const int n_s);


void populate_data_gradient(const Eigen::GpuDevice& dev, const float* g, const float* u, float* output, const int n_s);


void populate_reg_gradients(const Eigen::GpuDevice& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c);


void softmax(const Eigen::GpuDevice& dev, const float* e1, const float* e2, float* u, const int n_s, const int n_c);


void change_to_diff(const Eigen::GpuDevice& dev, float* transfer, float* diff, const int n_s, const float tau);


void softmax_gradient(const Eigen::GpuDevice& dev, const float* g, const float* u, float* g_d, const int n_s, const int n_c);


void populate_reg_mean_gradients(const Eigen::GpuDevice& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c);
void populate_reg_mean_gradients_and_add(const Eigen::GpuDevice& dev, const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c);
            
void get_effective_reg(const Eigen::GpuDevice& dev, float* r_eff, const float* u_b, const float* rx_b, const float* ry_b, const float* rz_b, const int n_x, const int n_y, const int n_z, const int n_c);

void copy_buffer(const Eigen::GpuDevice& dev, const float* source, float* dest, const int n_s);
void copy_buffer_clean(const Eigen::GpuDevice& dev, const float* source, float* dest, const int n_s);
void copy_buffer_clip(const Eigen::GpuDevice& dev, const float* source, float* dest, const int n_s, float value);

void inc_buffer(const Eigen::GpuDevice& dev, const float* inc, float* acc, const int n_s);
void inc_mult_buffer(const Eigen::GpuDevice& dev, const float* inc, float* acc, const int n_s, const float multi);
void inc2_mult_buffer(const Eigen::GpuDevice& dev, const float* inc_m1, const float* inc_m2, float* acc, const int n_s, const float multi);

void div_buffer(const Eigen::GpuDevice& dev, const float* div, float* res, const int n_s);
void mult_buffer(const Eigen::GpuDevice& dev, const float mult, float* res, const int n_s);

void add_store_then_max_buffer(const Eigen::GpuDevice& dev, const float* comp1, const float* comp2, float* store, float* res, const int n_s);

void exp_and_inc_buffer(const Eigen::GpuDevice& dev, const float* max, float* cost, float* acc, const int n_s);

void add_then_store(const Eigen::GpuDevice& dev, const float* addend1, const float* addend2, float* sum, const int size);
void add_then_store_2(const Eigen::GpuDevice& dev, const float* addend1, const float* addend2, float* sum1, float* sum2, const int size);

void process_grad_potts(const Eigen::GpuDevice& dev, const float* du_i, const float* u, float* loc, const int n_s, const int n_c, const float tau);
void get_gradient_for_u(const Eigen::GpuDevice& dev, const float* dy, float* du, const float* rx, const float* ry, const float* rz, const int n_x, const int n_y, const int n_z, const int n_c);

#endif // GPU_KERNELS