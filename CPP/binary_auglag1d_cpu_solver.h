#ifndef BINARY_AUGLAG1D_CPU_SOLVER_H
#define BINARY_AUGLAG1D_CPU_SOLVER_H

#include "binary_auglag_cpu_solver.h"
#include <functional>

class BINARY_AUGLAG_CPU_SOLVER_1D : public BINARY_AUGLAG_CPU_SOLVER_BASE
{
private:
    const BINARY_AUGLAG_CPU_SOLVER_1D* self;
    const int n_x;
    const float * const rx;
	float* px;

protected:
    int min_iter_calc();
    void clear_spatial_flows();
    void update_spatial_flow_calc();
    void clean_up();

public:
	BINARY_AUGLAG_CPU_SOLVER_1D(
        const bool channels_first,
        const int batch,
        const int n_c,
        const int sizes[1],
        const float * const data_cost,
        const float * const rx_cost,
        float* u 
	);
};

#endif
