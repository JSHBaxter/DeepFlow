    

#ifndef HMF_MEANPASS_SOLVER_H
#define HMF_MEANPASS_SOLVER_H

#include "hmf_trees.h"
#include "algorithm.h"
#include "spatial_meanpass.h"

template<typename DEV>
class HMF_MEANPASS_SOLVER : public MAXFLOW_ALGORITHM<DEV>
{
private:

protected:
    SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>* spatial_flow;
    TreeNode const* const* bottom_up_list;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const data;
    const float* const init_u;
    float* u;
    float* r_eff;
    float* u_full;
    
    float** u_ind;
    float** reg_ind;
    float* num_children;
    
    // optimization constants
	const float beta = 0.01f;
	const float epsilon = 0.0001f;
	const float tau = 0.5f;
    
    void block_iter(const int parity);
    
public:
    HMF_MEANPASS_SOLVER(
        const DEV & dev,
        TreeNode** bottom_up_list,
        const int dim,
        const int* dims,
        const int n_c,
        const int n_r,
        const float* const* const inputs,
        float* const u);
    ~HMF_MEANPASS_SOLVER();
    
    void run();
    
    void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    int get_buffer_size();
};

template class HMF_MEANPASS_SOLVER<CPU_DEVICE>;
#ifdef USE_CUDA
template class HMF_MEANPASS_SOLVER<CUDA_DEVICE>;
#endif

template<typename DEV>
class HMF_MEANPASS_GRADIENT : public MAXFLOW_ALGORITHM<DEV>
{
private:

protected:
    SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>* spatial_flow;
    TreeNode const* const* bottom_up_list;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const logits;
    const float* const grad;
    float* const g_data;
    float* u;
    float* tmp;
    float* dy;
    float* du;
    
    // optimization constants
	const float beta = 0.0001f;
	const float epsilon = 0.0001f;
	const float tau = 0.25f;
    
    void block_iter();
    
public:
    HMF_MEANPASS_GRADIENT(
        const DEV & dev,
        TreeNode** bottom_up_list,
        const int dim,
        const int* dims,
        const int n_c,
        const int n_r,
        const float *const *const inputs,
        float *const *const outputs);
    ~HMF_MEANPASS_GRADIENT();
    
    void run();
    
    void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    int get_buffer_size();
};

template class HMF_MEANPASS_GRADIENT<CPU_DEVICE>;
#ifdef USE_CUDA
template class HMF_MEANPASS_GRADIENT<CUDA_DEVICE>;
#endif


#endif //HMF_MEANPASS_SOLVER_H
