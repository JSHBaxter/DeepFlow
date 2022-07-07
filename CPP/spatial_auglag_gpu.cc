#include "spatial_auglag_gpu.h"

SPATIAL_AUGLAG_GPU_SOLVER_BASE::SPATIAL_AUGLAG_GPU_SOLVER_BASE(
    const cudaStream_t & dev,
    const bool channels_first,
    const int n_channels,
    const int img_size
    float* const g,
    float* const div,
):
dev(dev),
channels_first(channels_first),
n_channels(n_channels),
img_size(img_size),
g(g),
div(div)
{}