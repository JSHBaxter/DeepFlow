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
void parity_mask(const CPU_DEVICE & dev,float* const buffer, const int dim, const int* const n, const int n_c, const int parity);
void parity_mask(const CPU_DEVICE & dev,float* const buffer, const float* other, const int dim, const int* n, const int n_c, const int parity);

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

#endif //CPU_KERNELS