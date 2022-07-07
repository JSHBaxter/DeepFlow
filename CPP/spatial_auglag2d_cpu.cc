#include "spatial_auglag2d_cpu.h"
#include <math.h>
#include "cpu_kernels.h"

SPATIAL_AUGLAG_CPU_SOLVER_2D::SPATIAL_AUGLAG_CPU_SOLVER_2D(
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
SPATIAL_AUGLAG_CPU_SOLVER_BASE(channels_first,n_channels,img_size[0]*img_size[1],g,div,p),
n_x(img_size[0]),
n_y(img_size[1]),
rx(rx),
ry(ry),
px(px),
py(py)
{
    //nothing to do here
}

int SPATIAL_AUGLAG_CPU_SOLVER_2D::get_min_iter(){
    return 10;
}

int SPATIAL_AUGLAG_CPU_SOLVER_2D::get_max_iter(){
    return std::max(n_x,n_y);
}

int SPATIAL_AUGLAG_CPU_SOLVER_2D::get_number_buffer_full(){
    return 2;
}

void SPATIAL_AUGLAG_CPU_SOLVER_2D::init(){
    //clear flows
    clear(px, n_s*n_c);
    clear(py, n_s*n_c);
    clear(div, n_s*n_c);
}

void SPATIAL_AUGLAG_CPU_SOLVER_2D::run(){
    if(channels_first){
        compute_flows_channels_first( g, div, px, py, rx, ry, n_c, n_x, n_y);
    }else{
        compute_flows(g, div, px, py, rx, ry, n_c, n_x, n_y);
    }
}

void SPATIAL_AUGLAG_CPU_SOLVER_2D::deinit(){
    //nothing to do here
}

