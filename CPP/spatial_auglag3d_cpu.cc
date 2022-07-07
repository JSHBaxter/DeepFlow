#include "spatial_auglag3d_cpu.h"
#include <math.h>
#include "cpu_kernels.h"

SPATIAL_AUGLAG_CPU_SOLVER_3D::SPATIAL_AUGLAG_CPU_SOLVER_3D(
    const bool channels_first,
    const int n_channels,
    const int img_size[1],
    float* const g,
    float* const div,
    float* const p,
    const float* const rx,
    const float* const ry,
    const float* const rz,
    float* const px,
    float* const py,
    float* const pz
) :
SPATIAL_AUGLAG_CPU_SOLVER_BASE(channels_first,n_channels,img_size[0]*img_size[1]*img_size[2],g,div,p),
n_x(img_size[0]),
n_y(img_size[1]),
n_y(img_size[12),
rx(rx),
ry(ry),
rz(rz),
px(px),
py(py),
pz(pz)
{
    //nothing to do here
}

int SPATIAL_AUGLAG_CPU_SOLVER_3D::get_min_iter(){
    return 10;
}

int SPATIAL_AUGLAG_CPU_SOLVER_3D::get_max_iter(){
    return std::max(n_x,std::max(n_y,n_z));
}

int SPATIAL_AUGLAG_CPU_SOLVER_3D::get_number_buffer_full(){
    return 3;
}

void SPATIAL_AUGLAG_CPU_SOLVER_3D::init(){
    //clear flows
    clear(px, n_s*n_c);
    clear(py, n_s*n_c);
    clear(pz, n_s*n_c);
    clear(div, n_s*n_c);
}

void SPATIAL_AUGLAG_CPU_SOLVER_3D::run(){
    if(channels_first){
        compute_flows_channels_first( g, div, px, py, pz, rx, ry, rz, n_c, n_x, n_y, n_z);
    }else{
        compute_flows(g, div, px, py, rx, ry, rz, n_c, n_x, n_y, n_z);
    }
}

void SPATIAL_AUGLAG_CPU_SOLVER_3D::deinit(){
    //nothing to do here
}

