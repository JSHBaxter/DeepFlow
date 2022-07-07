#include "spatial_auglag2d_gpu.h"
#include <math.h>
#include "gpu_kernels.h"

SPATIAL_AUGLAG_GPU_SOLVER_2D::SPATIAL_AUGLAG_GPU_SOLVER_2D(
    const cudaStream_t & dev,
    const bool channels_first,
    const int n_channels,
    const int img_size[1],
    float* const g,
    float* const div,
    const float* const rx,
    const float* const ry,
    float* const px,
    float* const py
) :
SPATIAL_AUGLAG_GPU_SOLVER_BASE(dev,channels_first,n_channels,img_size[0]*img_size[1]),
n_x(img_size[0]),
n_y(img_size[1]),
rx(rx),
ry(ry),
px(px),
py(py)
{
    //nothing to do here
}

int SPATIAL_AUGLAG_GPU_SOLVER_2D::get_min_iter(){
    return 10;
}

int SPATIAL_AUGLAG_GPU_SOLVER_2D::get_max_iter(){
    return std::max(n_x,n_y);
}

int SPATIAL_AUGLAG_GPU_SOLVER_2D::get_number_buffer_full(){
    return 2;
}

void SPATIAL_AUGLAG_GPU_SOLVER_2D::init(){
    //clear flows
    clear_buffer(dev, px, n_s*n_c);
    clear_buffer(dev, py, n_s*n_c);
    clear_buffer(dev, div, n_s*n_c);
}

void SPATIAL_AUGLAG_GPU_SOLVER_2D::run(){
    if(channels_first){
        update_spatial_flows(dev, g, div, px, py, rx, ry, n_x, n_y, n_s*n_c);
    }else{
        update_spatial_flows_channels_last(dev, g, div, px, py, rx, ry, n_x, n_y, n_c, n_s*n_c);
    }
}

void SPATIAL_AUGLAG_GPU_SOLVER_2D::deinit(){
    //nothing to do here
}

