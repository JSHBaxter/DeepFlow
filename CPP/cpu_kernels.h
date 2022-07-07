/*

The purpose of this collection of functions is to abstract the details of buffer iteration away
from the mathematical logic of the solution algorithm.

Note that some functions may have multiple signatures. This is for handling data of different dimensionality.
Many functions operate point-wise and thus only one signature is needed regardless of image dimensionality.

*/

#ifndef CPU_KERNELS
#define CPU_KERNELS

#include "hmf_trees.h"
#include "algorithm.h"

//General use
void print_buffer(const CPU_DEVICE & dev,const float* buffer, const int n_s);
void set_buffer(const CPU_DEVICE & dev,float* buffer, const float number, const int n_s);
void clear_buffer(const CPU_DEVICE & dev,float* buffer, const int n_s);
void clear_buffer(const CPU_DEVICE & dev,float* buffer1, float* buffer2, const int n_s);
void clear_buffer(const CPU_DEVICE & dev,float* buffer1, float* buffer2, float* buffer3, const int n_s);
void copy_buffer(const CPU_DEVICE & dev,const float* bufferin, float* bufferout, const int n_s);


void add_then_store(const CPU_DEVICE& dev, const float* addend1, const float* addend2, float* sum, const int size);
void add_then_store(const CPU_DEVICE& dev, const float* addend1, const float* addend2, float* sum1, float* sum2, const int size);
void inc_mult_buffer(const CPU_DEVICE& dev, const float* inc, float* acc, const int n_s, const float multi);
void inc_buffer(const CPU_DEVICE & dev,const float* inc, float* acc, const int n_s);
void inc_buffer(const CPU_DEVICE & dev,const float inc, float* acc, const int n_s);
void ninc_buffer(const CPU_DEVICE & dev,const float* inc, float* acc, const int n_s);
void inc_inc_minc_buffer(const CPU_DEVICE& dev, const float* inc1, const float* inc2, const float* minc, const float multi, float* acc, const int n_s);
void m_inc_inc_ninc_minc_buffer(const CPU_DEVICE& dev, const float* inc1, const float* inc2, const float* ninc, const float* minc, const float multi_end, const float multi_all, float* acc, const int n_s);

void mult_buffer(const CPU_DEVICE & dev, const float number, float* buffer, const int n_s);
void div_buffer(const CPU_DEVICE & dev, const float number, float* buffer, const int n_s);
void constrain(const CPU_DEVICE & dev,float* buffer, const float* constraint, const int n_s);
void max_neg_constrain(const CPU_DEVICE& dev, float* buffer, const float* constrain, const int n_s);
float max_of_buffer(const CPU_DEVICE & dev,const float* buffer, const int n_s);
float max_diff(const CPU_DEVICE & dev,const float* buffer, const int n_c, const int n_s);
float spat_max_of_buffer(const CPU_DEVICE& dev, const float* buffer, const int n_s, const int n_c);
void log_buffer(const CPU_DEVICE & dev, const float* bufferin, float* bufferout, const int n_s);
void div_buffer(const CPU_DEVICE & dev,float* buffer, const float number, const int n_s);
void mult_buffer(const CPU_DEVICE & dev,float* buffer, const float number, const int n_s);
void unfold_buffer(const CPU_DEVICE & dev,float* buffer, const int n_s, const int n_c, const int n_r);
void refold_buffer(const CPU_DEVICE & dev,float* buffer, const int n_s, const int n_c, const int n_r);
void sigmoid(const CPU_DEVICE & dev,const float* bufferin, float* bufferout, const int n_s);
void sigmoid(const CPU_DEVICE & dev,const float* bufferin1, const float* bufferin2, float* bufferout, const int n_s);
void exp(const CPU_DEVICE & dev,const float* bufferin, float* bufferout, const int n_s);
void softmax(const CPU_DEVICE & dev,const float* bufferin, float* bufferout, const int n_s, const int n_c);
void softmax(const CPU_DEVICE & dev, const float* bufferin1, const float* bufferin2, float* bufferout, const int n_s, const int n_c);
float* transpose(const CPU_DEVICE & dev,const float* bufferin, float* bufferout, const int n_d1, const int n_d2);

//Tree iteration
void aggregate_bottom_up(const CPU_DEVICE & dev,const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list);
void aggregate_bottom_up(const CPU_DEVICE & dev,float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list);
void aggregate_top_down(const CPU_DEVICE & dev,float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list);
void aggregate_bottom_up_channels_first(const CPU_DEVICE & dev,const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list);
void aggregate_bottom_up_channels_first(const CPU_DEVICE & dev,float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list);
void aggregate_top_down_channels_first(const CPU_DEVICE & dev,float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list);
void aggregate_bottom_up_channels_first(const CPU_DEVICE & dev,const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const DAGNode* const* bottom_up_list);
void aggregate_bottom_up_channels_first(const CPU_DEVICE & dev,float* buffer, const int n_s, const int n_r, const DAGNode* const* bottom_up_list);
void aggregate_top_down_channels_first(const CPU_DEVICE & dev,float* buffer, const int n_s, const int n_r, const DAGNode* const* bottom_up_list);

//Functions speficially for augmented lagrangian
void compute_source_flow(const CPU_DEVICE & dev, const float* u, float* ps, const float* pt, const float* div, const float icc, const int n_c, const int n_s);
void compute_sink_flow(const CPU_DEVICE & dev, const float* u, const float* ps, float* pt, const float* div, const float* d, const float icc, const int n_c, const int n_s);
void compute_multipliers(const CPU_DEVICE & dev, float* erru, float* u, const float* ps, const float* pt, const float* div, const float cc, const int n_c, const int n_s);
void update_source_sink_multiplier_potts(const CPU_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s);
void update_source_sink_multiplier_binary(const CPU_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s);
void calc_capacity_potts(const CPU_DEVICE & dev,float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void calc_capacity_binary(const CPU_DEVICE & dev,float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau);
void update_spatial_flows(const CPU_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const int dim, const int* const n, const int n_c);
void update_spatial_star_flows(const CPU_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const float *const *const l, const int dim, const int* const n, const int n_c);
void init_flows_binary(const CPU_DEVICE & dev,const float* d, float* ps, float* pt, float* u, const int n_s);
void init_flows_potts(const CPU_DEVICE & dev,const float* d, float* ps, float* pt, float* u, const int n_s, const int n_c);
void init_flows(const CPU_DEVICE & dev,const float* d, float* ps, const int n_c, const int n_s);

//Functions specifically for the mean field forward calculation
void parity_mask(const CPU_DEVICE & dev,float* const buffer, const int dim, const int* const n, const int n_c, const int parity);
void parity_mask(const CPU_DEVICE & dev,float* const buffer, const float* other, const int dim, const int* n, const int n_c, const int parity);
void change_to_diff(const CPU_DEVICE & dev,float* buffer, float* update, const int n_s, const float alpha);
void get_effective_reg(const CPU_DEVICE & dev, float* const r_eff, const float* const u, const float *const *const r, const int dim, const int* const n, const int n_c);

//Functions for mean field backward calculation
void untangle_softmax(const CPU_DEVICE & dev,const float* g, const float* u, float* dy, const int n_s, const int n_c);
void untangle_sigmoid(const CPU_DEVICE & dev,const float* g, const float* u, float* dy, const int n_s);
void get_gradient_for_u(const CPU_DEVICE & dev,const float* dy, const float *const *const r, float* const du, const int dim, const int* const n, const int n_c, const float tau);
void populate_reg_mean_gradients(const CPU_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c);
void populate_reg_mean_gradients_and_add(const CPU_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau);

//void get_reg_gradients(const CPU_DEVICE & dev,const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau);

#endif //CPU_KERNELS