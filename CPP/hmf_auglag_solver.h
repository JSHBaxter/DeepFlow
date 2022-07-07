    

#ifndef HMF_AUGLAG_SOLVER_H
#define HMF_AUGLAG_SOLVER_H

#include "algorithm.h"
#include "spatial_auglag.h"
#include "hmf_trees.h"

template<typename DEV>
class HMF_AUGLAG_SOLVER : public MAXFLOW_ALGORITHM<DEV>
{
private:
    
protected:
    SPATIAL_AUGLAG_SOLVER<DEV>* spatial_flow;
    TreeNode const* const* bottom_up_list;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const data;
    float* u;
    float* ps;
    float* pt;
    float* u_tmp;
    float* div;
    float* g;
    
    float* g_ps;
    float** ps_ind;
    float** g_ind;
    float* num_children;
    
    // optimization constants
    const float tau = 0.1f;
    const float beta = 0.01f;
    const float epsilon = 10e-5f;
    const float cc = 1.0f;
    const float icc = 1.0f/cc;
    
    void block_iter();
    
public:
    HMF_AUGLAG_SOLVER(
        const DEV & dev,
        TreeNode** bottom_up_list,
        const bool star,
        const int dim,
        const int* dims,
        const int n_c,
        const int n_r,
        const float * const * const inputs,
        float* u);
        
    ~HMF_AUGLAG_SOLVER();
    
    void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    int get_buffer_size();
    
    void run();
};

template class HMF_AUGLAG_SOLVER<CPU_DEVICE>;
#ifdef USE_CUDA
template class HMF_AUGLAG_SOLVER<CUDA_DEVICE>;
#endif

#endif //HMF_AUGLAG_SOLVER_H
