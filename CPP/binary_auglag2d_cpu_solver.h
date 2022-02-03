#ifndef BINARY_AUGLAG2D_CPU_SOLVER_H
#define BINARY_AUGLAG2D_CPU_SOLVER_H

#include "binary_auglag_cpu_solver.h"

class BINARY_AUGLAG_CPU_SOLVER_2D : public BINARY_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float * const rx;
    const float * const ry;
	float* px;
	float* py;

protected:
    virtual int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();
    virtual void clean_up();

    
public:
	BINARY_AUGLAG_CPU_SOLVER_2D(
        const bool channels_first,
        const int batch,
        const int n_c,
        const int sizes[2],
        const float * const data_cost,
        const float * const rx_cost,
        const float * const ry_cost,
        float* u 
	);
};

#endif

