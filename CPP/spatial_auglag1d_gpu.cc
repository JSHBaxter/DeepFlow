#include "spatial_auglag1d_gpu.h"
#include "gpu_kernels.h"

SPATIAL_AUGLAG_GPU_SOLVER_1D::SPATIAL_AUGLAG_GPU_SOLVER_1D(
    const cudaStream_t & dev,
    const bool channels_first,
    const int n_channels,
    const int img_size[1],
    float* const g,
    float* const div,
    const float* const rx,
    float* const px
) :
SPATIAL_AUGLAG_GPU_SOLVER_BASE(dev,channels_first,n_channels,img_size[0]),
n_x(img_size[0]),
rx(rx),
px(px)
{
    //nothing to do here
}

int SPATIAL_AUGLAG_GPU_SOLVER_1D::get_min_iter(){
    return 10;
}

int SPATIAL_AUGLAG_GPU_SOLVER_1D::get_max_iter(){
    return n_x;
}

int SPATIAL_AUGLAG_GPU_SOLVER_1D::get_number_buffer_full(){
    return 1;
}

void SPATIAL_AUGLAG_GPU_SOLVER_1D::init(){
    //clear px
    clear_buffer(dev, px, n_s*n_c);
    clear_buffer(dev, div, n_s*n_c);
}

void SPATIAL_AUGLAG_GPU_SOLVER_1D::run(){
    if(channels_first){
    if(channels_first){
        update_spatial_flows(dev, g, div, px, rx, n_x, n_s*n_c);
    }else{
        update_spatial_flows_channels_last(dev, g, div, px, rx, n_x, n_c, n_s*n_c);
    }
}

void SPATIAL_AUGLAG_GPU_SOLVER_1D::deinit(){
    //nothing to do here
}

