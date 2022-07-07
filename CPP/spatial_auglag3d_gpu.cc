#include "spatial_auglag3d_gpu.h"
#include <math.h>
#include "gpu_kernels.h"

SPATIAL_AUGLAG_GPU_SOLVER_3D::SPATIAL_AUGLAG_GPU_SOLVER_3D(
    const cudaStream_t & dev,
    const bool channels_first,
    const int n_channels,
    const int img_size[1],
    float* const g,
    float* const div,
    const float* const rx,
    const float* const ry,
    const float* const rz,
    float* const px,
    float* const py,
    float* const pz
) :
SPATIAL_AUGLAG_GPU_SOLVER_BASE(dev,channels_first,n_channels,img_size[0]*img_size[1]*img_size[2]),
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

int SPATIAL_AUGLAG_GPU_SOLVER_3D::get_min_iter(){
    return 10;
}

int SPATIAL_AUGLAG_GPU_SOLVER_3D::get_max_iter(){
    return std::max(n_x,std::max(n_y,n_z));
}

int SPATIAL_AUGLAG_GPU_SOLVER_3D::get_number_buffer_full(){
    return 3;
}

void SPATIAL_AUGLAG_GPU_SOLVER_3D::init(){
    //clear flows
    clear_buffer(dev, px, n_s*n_c);
    clear_buffer(dev, py, n_s*n_c);
    clear_buffer(dev, pz, n_s*n_c);
    clear_buffer(dev, div, n_s*n_c);
}

void SPATIAL_AUGLAG_GPU_SOLVER_3D::run(){
    if(channels_first){
        update_spatial_flows(dev, g, div, px, py, pz, rx, ry, rz, n_x, n_y, n_z, n_s*n_c);
    }else{
        update_spatial_flows_channels_last(dev, g, div, px, py, pz, rx, ry, rz, n_x, n_y, n_z, n_c, n_s*n_c);
    }
}

void SPATIAL_AUGLAG_GPU_SOLVER_3D::deinit(){
    //nothing to do here
}

