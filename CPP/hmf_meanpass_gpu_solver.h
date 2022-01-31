    

#ifndef HMF_MEANPASS_GPU_SOLVER_H
#define HMF_MEANPASS_GPU_SOLVER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "hmf_trees.h"

class HMF_MEANPASS_GPU_SOLVER_BASE
{
private:

protected:
    const cudaStream_t & dev;
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const data;
    float* const u;
    float* const r_eff;
    float* const u_full;
    
    float** u_ind;
    float** reg_ind;
    
    // optimization constants
	const float beta = 0.001f;
	const float epsilon = 0.0001f;
	const float tau = 0.5f;
    
    virtual int min_iter_calc() = 0;
    virtual void update_spatial_flow_calc() = 0;
    virtual void parity_mask_buffer(float* buffer, const int parity) = 0;
    virtual void parity_merge_buffer(float* buffer, const float* other, const int parity) = 0;
    void block_iter(const int parity);
    
public:
    HMF_MEANPASS_GPU_SOLVER_BASE(
        const cudaStream_t & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* const data_cost,
        const float* const init_u,
        float* const u,
        float** full_buff,
        float** img_buff);
    ~HMF_MEANPASS_GPU_SOLVER_BASE();
    
    void operator()();
};

class HMF_MEANPASS_GPU_GRADIENT_BASE
{
private:

protected:
    const cudaStream_t & dev;
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const logits;
    const float* const grad;
    float* const g_data;
    float* const u;
    float* const tmp;
    float* const dy;
    float* const du;
    
    // optimization constants
	const float beta = 0.0001f;
	const float epsilon = 0.0001f;
	const float tau = 0.25f;
    
    virtual int min_iter_calc() = 0;
	virtual void clear_variables() = 0;
    virtual void get_reg_gradients_and_push(float tau) = 0;
    void block_iter();
    
public:
    HMF_MEANPASS_GPU_GRADIENT_BASE(
        const cudaStream_t & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* const u,
        const float* const g,
        float* const g_d,
        float** full_buff);
    
    void operator()();
};

#endif //HMF_MEANPASS_GPU_SOLVER_H
