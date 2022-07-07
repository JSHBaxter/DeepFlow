#ifndef SPATIAL_AUGLAG2D_CPU_SOLVER_H
#define SPATIAL_AUGLAG2D_CPU_SOLVER_H

#include "spatial_auglag_cpu.h"

class SPATIAL_AUGLAG_CPU_SOLVER_2D : SPATIAL_AUGLAG_CPU_SOLVER_BASE
{
protected:
    const int n_x;
    const int n_y;
    const float* const rx;
    const float* const ry;
    float* const px;
    float* const py;

public:
	SPATIAL_AUGLAG_CPU_SOLVER_2D(
        const bool channels_first,
        const int n_channels,
        const int img_size[2],
        float* const g,
        float* const div,
        const float* const rx,
        const float* const ry,
        float* const px,
        float* const py
	);
    
    int get_min_iter();
    int get_max_iter();
    
    static int get_number_buffer_full();
    
    void init();
    void run();
    void deinit();
};

#endif
