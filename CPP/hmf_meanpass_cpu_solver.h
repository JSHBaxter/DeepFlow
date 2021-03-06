    

#ifndef HMF_MEANPASS_CPU_SOLVER_H
#define HMF_MEANPASS_CPU_SOLVER_H

#include "hmf_trees.h"

class HMF_MEANPASS_CPU_SOLVER_BASE
{
private:

protected:
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const data;
    float* const u;
    float* r_eff;
    float* u_tmp;
    
    // optimization constants
    const float tau = 1.0f;
    const float beta = 0.001f;
    const float epsilon = 10e-5f;
    
    virtual int min_iter_calc() = 0;
    virtual void update_spatial_flow_calc() = 0;
    virtual void parity_mask_buffer(float* buffer, const int parity) = 0;
    virtual void parity_merge_buffer(float* buffer, const float* other, const int parity) = 0;
    float block_iter(const int parity, bool last);
    
public:
    HMF_MEANPASS_CPU_SOLVER_BASE(
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* data_cost,
        const float* const init_u,
        float* u);
        
    ~HMF_MEANPASS_CPU_SOLVER_BASE();
    
    void operator()();
};

class HMF_MEANPASS_CPU_GRADIENT_BASE
{
private:

protected:
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const logits;
    const float* const grad;
    float* g_data;
    float* dy;
    float* u;
    float* g_u;
    
    // optimization constants
    const float tau = 0.1f;
    const float beta = 0.005f;
    const float epsilon = 10e-5f;
    
    virtual int min_iter_calc() = 0;
    virtual void update_spatial_flow_calc(bool use_tau) = 0;
    float block_iter();
    
public:
    HMF_MEANPASS_CPU_GRADIENT_BASE(
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* u,
        const float* g,
        float* g_d );
        
    ~HMF_MEANPASS_CPU_GRADIENT_BASE();
    
    void operator()();
};

#endif //HMF_MEANPASS_CPU_SOLVER_H
