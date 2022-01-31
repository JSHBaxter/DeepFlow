#ifndef HMF_AUGLAG3D_CPU_SOLVER_H
#define HMF_AUGLAG3D_CPU_SOLVER_H

#include "hmf_auglag_cpu_solver.h"
#include "hmf_trees.h"


class HMF_AUGLAG_CPU_SOLVER_3D : public HMF_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* rx;
    const float* ry;
    const float* rz;
    float* px;
    float* py;
    float* pz;
    const float* rx_b;
    const float* ry_b;
    const float* rz_b;

protected:
    int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();
    void clean_up();
    
public:
    ~HMF_AUGLAG_CPU_SOLVER_3D();
    HMF_AUGLAG_CPU_SOLVER_3D(
        const bool channel_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[3],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* u );
    
};

#endif
