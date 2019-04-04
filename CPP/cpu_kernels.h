#ifndef CPU_KERNELS
#define CPU_KERNELS

#include "hmf_trees.h"

void softmax(const float* bufferin, float* bufferout, const int n_s, const int n_c);
float softmax_with_convergence(const float* bufferin, float* bufferout, const int n_s, const int n_c, const float alpha);

void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c);

void aggregate_bottom_up(const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list);

#endif //CPU_KERNELS