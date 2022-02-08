#ifndef POTTS_AUGLAG1D_CPU_SOLVER_H
#define POTTS_AUGLAG1D_CPU_SOLVER_H

#include "potts_auglag_cpu_solver.h"

class POTTS_AUGLAG_CPU_SOLVER_1D : public POTTS_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* rx;
	float* px;

protected:
    virtual int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();
    virtual void clean_up();

public:
    ~POTTS_AUGLAG_CPU_SOLVER_1D();
	POTTS_AUGLAG_CPU_SOLVER_1D(
        const int channels_first,
        const int batch,
        const int n_c,
        const int sizes[1],
        const float* data_cost,
        const float* rx_cost,
        float* u 
	);
};

#endif
