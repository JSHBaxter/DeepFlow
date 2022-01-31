    

#ifndef HMF_MEANPASS_CPU_SOLVER_H
#define HMF_MEANPASS_CPU_SOLVER_H

#include "hmf_trees.h"

class HMF_MEANPASS_CPU_SOLVER_BASE
{
private:

protected:
    TreeNode const* const* bottom_up_list;
    const bool channels_first;
    const int b;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* data;
    const float* data_b;
    float* u;
    float* r_eff;
    float* u_tmp;
    
    // optimization constants
	const float beta = 0.001f;
	const float epsilon = 0.0001f;
	const float tau = 0.5f;
    
    virtual int min_iter_calc() = 0;
    virtual void init_reg_info() = 0;
    virtual void clean_up() = 0;
    virtual void update_spatial_flow_calc() = 0;
    virtual void parity_mask_buffer(float* buffer, const int parity) = 0;
    virtual void parity_merge_buffer(float* buffer, const float* other, const int parity) = 0;
    float block_iter(const int parity, bool last);
    
public:
    HMF_MEANPASS_CPU_SOLVER_BASE(
        const bool channels_first,
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
    const bool channels_first;
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
    float* g_u_l;
    
    // optimization constants
	const float beta = 0.0001f;
	const float epsilon = 0.0001f;
	const float tau = 0.25f;
    
    virtual int min_iter_calc() = 0;
    virtual void init_reg_info() = 0;
    virtual void clean_up() = 0;
    virtual void get_reg_gradients_and_push(float tau) = 0;
    void block_iter();
    
public:
    HMF_MEANPASS_CPU_GRADIENT_BASE(
        const bool channels_first,
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
