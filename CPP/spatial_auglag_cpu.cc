#include "spatial_auglag_cpu.h"

#include "spatial_auglag1d_cpu.h"
#include "spatial_auglag2d_cpu.h"
#include "spatial_auglag3d_cpu.h"

SPATIAL_AUGLAG_CPU_SOLVER_BASE::SPATIAL_AUGLAG_CPU_SOLVER_BASE(
    const bool channels_first,
    const int n_channels,
    const int img_size
    float* const g,
    float* const div,
    float* const p
):
channels_first(channels_first),
n_channels(n_channels),
img_size(img_size),
g(g),
div(div),
p(p),
{}

SPATIAL_AUGLAG_CPU_SOLVER_BASE::SPATIAL_AUGLAG_CPU_SOLVER_BASE(){
   delete [] p;
}

SPATIAL_AUGLAG_CPU_SOLVER_BASE* SPATIAL_AUGLAG_CPU_SOLVER_BASE::factory(const bool channels_first, const int n_c, const int dim, const int* const dims, float* const g, float* const div, const float* const* r){
    
    if( dim < 0 || dim > 3 )
        return NULL;
    
    SPATIAL_AUGLAG_CPU_SOLVER_BASE* retval = NULL;
    int n_s = 1;
    for(int i = 0; i < dim; i++)
        n_s *= dims[i];
    
    float* p = new float[n_c*n_s*dim];
    
    switch(dim){
        case 1:
            retval = new SPATIAL_AUGLAG_CPU_SOLVER_1D( channels_first, n_channels, dims, g, div, p, r[0], p);
            break;
            
        case 2:
            retval = new SPATIAL_AUGLAG_CPU_SOLVER_2D( channels_first, n_channels, dims, g, div, p, r[0], r[1], p, p+n_s*n_c);
            break;
            
        case 3:
            retval = new SPATIAL_AUGLAG_CPU_SOLVER_3D( channels_first, n_channels, dims, g, div, p, r[0], r[1], r[2], p, p+n_s*n_c, p+2*n_s*n_c);
            break;
    }
    
    return retval;
    
}