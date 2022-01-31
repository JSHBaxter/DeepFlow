#ifndef HMF_AUGLAG1D_CPU_SOLVER_H
#define HMF_AUGLAG1D_CPU_SOLVER_H

#include "hmf_auglag_cpu_solver.h"
#include "cpu_kernels.h"
#include "hmf_trees.h"


class HMF_AUGLAG_CPU_SOLVER_1D : public HMF_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* rx;
    float* px;
    const float* rx_b;

protected:
    int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();
    void clean_up();
    
public:
    ~HMF_AUGLAG_CPU_SOLVER_1D();
    HMF_AUGLAG_CPU_SOLVER_1D(
        const bool channels_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[1],
        const float* data_cost,
        const float* rx_cost,
        float* u );
};

#endif

