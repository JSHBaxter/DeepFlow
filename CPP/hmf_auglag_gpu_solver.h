    

#ifndef HMF_AUGLAG_GPU_SOLVER_H
#define HMF_AUGLAG_GPU_SOLVER_H

#include "hmf_trees.h"
#include <cuda.h>
#include <cuda_runtime.h>

class HMF_AUGLAG_GPU_SOLVER_BASE
{
private:
    
protected:
    const cudaStream_t & dev;
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* data;
    float* u;
    float* ps;
    float* pt;
    float* u_tmp;
    float* div;
    float* g;
    float* g_ps;
    float** ps_ind;
    float** g_ind;
    int* num_children;
    
    // optimization constants
    const float tau = 0.1f;
    const float beta = 0.001f;
    const float epsilon = 10e-5f;
    const float cc = 1.0f;
    const float icc = 1.0f/cc;
    
    virtual int min_iter_calc() = 0;
    virtual void clear_spatial_flows() = 0;
    virtual void update_spatial_flow_calc() = 0;
    void block_iter();
    
public:
    HMF_AUGLAG_GPU_SOLVER_BASE(
        const cudaStream_t & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* data_cost,
        float* u,
        float** full_buff,
        float** img_buff);
    ~HMF_AUGLAG_GPU_SOLVER_BASE();
    
    void operator()();
};

#endif //HMF_AUGLAG_GPU_SOLVER_H
