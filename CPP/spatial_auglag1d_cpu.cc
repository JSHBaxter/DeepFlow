#include "spatial_auglag1d_cpu.h"
#include "cpu_kernels.h"

SPATIAL_AUGLAG_CPU_SOLVER_1D::SPATIAL_AUGLAG_CPU_SOLVER_1D(
    const bool channels_first,
    const int n_channels,
    const int img_size[1],
    float* const g,
    float* const div,
    const float* const rx,
    float* const px
) :
SPATIAL_AUGLAG_CPU_SOLVER_BASE(channels_first,n_channels,img_size[0],g,div,p),
n_x(img_size[0]),
rx(rx),
px(px)
{
    //nothing to do here
}

int SPATIAL_AUGLAG_CPU_SOLVER_1D::get_min_iter(){
    return 10;
}

int SPATIAL_AUGLAG_CPU_SOLVER_1D::get_max_iter(){
    return n_x;
}

int SPATIAL_AUGLAG_CPU_SOLVER_1D::get_number_buffer_full(){
    return 1;
}

void SPATIAL_AUGLAG_CPU_SOLVER_1D::init(){
    //clear px
    clear(px, n_s*n_c);
    clear(div, n_s*n_c);
}

void SPATIAL_AUGLAG_CPU_SOLVER_1D::run(){
    if(channels_first){
        compute_flows_channels_first(g, div, px, rx, n_c, n_x);
    }else{
        compute_flows(g, div, px, rx, n_c, n_x);
    }
}

void SPATIAL_AUGLAG_CPU_SOLVER_1D::deinit(){
    //nothing to do here
}

