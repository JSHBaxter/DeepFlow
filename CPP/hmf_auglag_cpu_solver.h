    

#ifndef HMF_AUGLAG_CPU_SOLVER_H
#define HMF_AUGLAG_CPU_SOLVER_H

#include "hmf_trees.h"

class HMF_AUGLAG_CPU_SOLVER_BASE
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
    float* data_b;
    float* u;
    float* ps;
    float* pt;
    float* u_tmp;
    float* div;
    float* g;
    
    // optimization constants
    const float tau = 0.1f;
    const float beta = 0.001f;
    const float epsilon = 10e-5f;
    const float cc = 10.0f;//0.25f;
    const float icc = 1.0f/cc;
    
    virtual int min_iter_calc() = 0;
    virtual void clear_spatial_flows() = 0;
    virtual void update_spatial_flow_calc() = 0;
    virtual void clean_up() = 0;
    void block_iter();
    
public:
    ~HMF_AUGLAG_CPU_SOLVER_BASE();
    HMF_AUGLAG_CPU_SOLVER_BASE(
        const bool channels_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* data_cost,
        float* u) ;
    
    void operator()();
};

#endif //HMF_AUGLAG_CPU_SOLVER_H
