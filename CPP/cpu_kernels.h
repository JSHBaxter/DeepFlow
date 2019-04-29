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
void clear(float* buffer1, float* buffer2, const int n_s);
void clear(float* buffer1, float* buffer2, float* buffer3, const int n_s);
void copy(const float* bufferin, float* bufferout, const int n_s);
void inc(const float* inc, float* acc, const int n_s);
void inc(const float* inc, float* acc, const float alpha, const int n_s);
float maxabs(const float* buffer, const int n_s);
void log_buffer(float* buffer, const int n_s);

//Tree iteration
void aggregate_bottom_up(const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list);

//Functions speficially for augmented lagrangian
void compute_source_flow( const float* u, float* ps, const float* pt, const float* div, const float icc, const int n_c, const int n_s);
void compute_sink_flow( const float* u, const float* ps, float* pt, const float* div, const float* d, const float icc, const int n_c, const int n_s);
void compute_multipliers( float* erru, float* u, const float* ps, const float* pt, const float* div, const float cc, const int n_c, const int n_s);
void compute_source_sink_multipliers( float* erru, float* u, float* ps, float* pt, const float* div, const float* d, const float cc, const float icc, const int n_c, const int n_s);
void compute_flows( float* g, const float* u, const float* ps, const float* pt, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const float tau, const float icc, const int n_c, const int n_x, const int n_y, const int n_z);

//Functions specifically for the mean field forward calculation
void softmax(const float* bufferin, float* bufferout, const int n_s, const int n_c);
float softmax_with_convergence(const float* bufferin, float* bufferout, const int n_s, const int n_c, const float alpha);

void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c);
void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* u, const int n_x, const int n_y, const int n_c);
void calculate_r_eff(float* r_eff, const float* rx, const float* u, const int n_x, const int n_c);


#endif //CPU_KERNELS