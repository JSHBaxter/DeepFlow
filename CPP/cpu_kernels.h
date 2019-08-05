/*

The purpose of this collection of functions is to abstract the details of buffer iteration away
from the mathematical logic of the solution algorithm.

Note that some functions may have multiple signatures. This is for handling data of different dimensionality.
Many functions operate point-wise and thus only one signature is needed regardless of image dimensionality.

*/

#ifndef CPU_KERNELS
#define CPU_KERNELS

#include "hmf_trees.h"
//General use
void clear(float* buffer, const int n_s);
void print_buffer(float* buffer, const int n_s);
void set(float* buffer, const float number, const int n_s);
void clear(float* buffer1, float* buffer2, const int n_s);
void clear(float* buffer1, float* buffer2, float* buffer3, const int n_s);
void copy(const float* bufferin, float* bufferout, const int n_s);
void inc(const float* inc, float* acc, const int n_s);
void ninc(const float* inc, float* acc, const int n_s);
void inc(const float* inc, float* acc, const float alpha, const int n_s);
void constrain(float* buffer, const float* constraint, const int n_s);
float maxabs(const float* buffer, const int n_s);
float max_diff(const float* buffer, const int n_c, const int n_s);
void log_buffer(float* buffer, const int n_s);
void div_buffer(float* buffer, const float number, const int n_s);
void mult_buffer(float* buffer, const float number, const int n_s);
void unfold_buffer(float* buffer, const int n_s, const int n_c, const int n_r);
void refold_buffer(float* buffer, const int n_s, const int n_c, const int n_r);

//Tree iteration
void aggregate_bottom_up(const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list);
void aggregate_bottom_up(float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list);
void aggregate_top_down(float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list);

//Functions speficially for augmented lagrangian
void compute_source_flow( const float* u, float* ps, const float* pt, const float* div, const float icc, const int n_c, const int n_s);
void compute_sink_flow( const float* u, const float* ps, float* pt, const float* div, const float* d, const float icc, const int n_c, const int n_s);
void compute_multipliers( float* erru, float* u, const float* ps, const float* pt, const float* div, const float cc, const int n_c, const int n_s);
void compute_source_sink_multipliers( float* erru, float* u, float* ps, float* pt, const float* div, const float* d, const float cc, const float icc, const int n_c, const int n_s);
void compute_capacity_potts(float* g, const float* u, const float* ps, const float* pt, const float* div, const int n_s, const int n_c, const float tau, const float icc);
void compute_flows(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const int n_c, const int n_x, const int n_y, const int n_z);
void compute_flows(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_c, const int n_x, const int n_y);
void compute_flows(const float* g, float* div, float* px, const float* rx, const int n_c, const int n_x);
void compute_flows_channels_first(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const int n_c, const int n_x, const int n_y, const int n_z);
void compute_flows_channels_first(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_c, const int n_x, const int n_y);
void compute_flows_channels_first(const float* g, float* div, float* px, const float* rx, const int n_c, const int n_x);
void init_flows(const float* d, float* ps, float* pt, const int n_c, const int n_s);
void init_flows(const float* d, float* ps, const int n_c, const int n_s);
void init_flows_channels_first(const float* d, float* ps, const int n_c, const int n_s);

//Functions specifically for the mean field forward calculation
void softmax(const float* bufferin, float* bufferout, const int n_s, const int n_c);
void softmax_update(const float* bufferin, float* bufferout, const int n_s, const int n_c, const float alpha);
float softmax_with_convergence(const float* bufferin, float* bufferout, const int n_s, const int n_c, const float alpha);
void sigmoid(const float* bufferin, float* bufferout, const int n_s);
void sigmoid_update(const float* bufferin, float* bufferout, const int n_s, const float alpha);
float sigmoid_with_convergence(const float* bufferin, float* bufferout, const int n_s, const float alpha);

void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c);
void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* u, const int n_x, const int n_y, const int n_c);
void calculate_r_eff(float* r_eff, const float* rx, const float* u, const int n_x, const int n_c);

//Functions for mean field backward calculation
void untangle_softmax(const float* g, const float* u, float* dy, const int n_s, const int n_c);
void untangle_sigmoid(const float* g, const float* u, float* dy, const int n_s);
void get_gradient_for_u(const float* dy, const float* rx, const float* ry, const float* rz, float* du, const int n_x, const int n_y, const int n_z, const int n_c, const float tau);
void get_gradient_for_u(const float* dy, const float* rx, const float* ry, float* du, const int n_x, const int n_y, const int n_c, const float tau);
void get_gradient_for_u(const float* dy, const float* rxs, float* du, const int n_x, const int n_c, const float tau);
void get_reg_gradients(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau);
void get_reg_gradients(const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau);
void get_reg_gradients(const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau);

#endif //CPU_KERNELS