
#ifndef POTTS_AUGLAG_CPU_SOLVER_3D_
#define POTTS_AUGLAG_CPU_SOLVER_3D_

#include "potts_auglag_cpu_solver.h"

class POTTS_AUGLAG_CPU_SOLVER_3D : public POTTS_AUGLAG_CPU_SOLVER_BASE
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

protected:
    virtual int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();
    virtual void clean_up();

public:
	POTTS_AUGLAG_CPU_SOLVER_3D(
        const bool channels_first,
        const int batch,
        const int n_c,
        const int sizes[3],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* u 
	);
};

#endif
