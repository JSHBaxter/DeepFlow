
#include "binary_auglag3d_gpu_solver.h"
#include "gpu_kernels.h"

int BINARY_AUGLAG_GPU_SOLVER_3D::min_iter_calc(){
	return std::max(std::max(n_x,n_y), n_z);
}

void BINARY_AUGLAG_GPU_SOLVER_3D::clear_spatial_flows(){
	clear_buffer(dev, px, n_c*n_s);
	clear_buffer(dev, py, n_c*n_s);
	clear_buffer(dev, pz, n_c*n_s);
}

void BINARY_AUGLAG_GPU_SOLVER_3D::update_spatial_flow_calc(){
	update_spatial_flows(dev, g, div, px, py, pz, rx, ry, rz, n_x, n_y, n_z, n_s*n_c);
}

BINARY_AUGLAG_GPU_SOLVER_3D::BINARY_AUGLAG_GPU_SOLVER_3D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[3],
	const float* data_cost,
	const float* rx_cost,
	const float* ry_cost,
	const float* rz_cost,
	float* u,
	float** buffers_full,
	float** buffers_img
):
BINARY_AUGLAG_GPU_SOLVER_BASE(dev, batch, sizes[0]*sizes[1]*sizes[2], n_c, data_cost, u, buffers_full, buffers_img),
n_x(sizes[0]),
n_y(sizes[1]),
n_z(sizes[2]),
rx(rx_cost),
ry(ry_cost),
rz(rz_cost),
px(buffers_full[4]),
py(buffers_full[5]),
pz(buffers_full[6])
{}
