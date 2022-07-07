#ifndef SPATIAL_AUGLAG1D_CPU_SOLVER_H
#define SPATIAL_AUGLAG1D_CPU_SOLVER_H

#include "spatial_auglag_cpu.h"

class SPATIAL_AUGLAG_CPU_SOLVER_1D : SPATIAL_AUGLAG_CPU_SOLVER_BASE
{
protected:
    const int n_x;
    const float* const rx;
    float* const px;
public:
	SPATIAL_AUGLAG_CPU_SOLVER_1D(
        const bool channels_first,
        const int n_channels,
        const int img_size[1],
        float* const g,
        float* const div,
        const float* const rx,
        float* const px
	);
    
    int get_min_iter();
    int get_max_iter();
    
    static int get_number_buffer_full();
    
    void init();
    void run();
    void deinit();
};

#endif
