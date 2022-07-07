#ifndef SPATIAL_AUGLAG3D_GPU_SOLVER_H
#define SPATIAL_AUGLAG3D_GPU_SOLVER_H

#include "spatial_auglag_gpu.h"

class SPATIAL_AUGLAG_GPU_SOLVER_3D : SPATIAL_AUGLAG_GPU_SOLVER_BASE
{
protected:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* const rx;
    const float* const ry;
    const float* const rz;
    float* const px;
    float* const py;
    float* const pz;

public:
	SPATIAL_AUGLAG_GPU_SOLVER_3D(
        const cudaStream_t & dev,
        const bool channels_first,
        const int n_channels,
        const int img_size[3],
        float* const g,
        float* const div,
        const float* const rx,
        const float* const ry,
        const float* const rz,
        float* const px,
        float* const py,
        float* const pz
	);
    
    int get_min_iter();
    int get_max_iter();
    
    int get_number_buffer_full();
    
    void init();
    void run();
    void deinit();
};

#endif
