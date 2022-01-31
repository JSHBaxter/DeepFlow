
#include "binary_auglag2d_gpu_solver.h"
#include "gpu_kernels.h"

int BINARY_AUGLAG_GPU_SOLVER_2D::min_iter_calc(){
	return std::max(n_x,n_y);
}

void BINARY_AUGLAG_GPU_SOLVER_2D::clear_spatial_flows(){
	clear_buffer(dev, px, n_c*n_s);
	clear_buffer(dev, py, n_c*n_s);
}

void BINARY_AUGLAG_GPU_SOLVER_2D::update_spatial_flow_calc(){
	update_spatial_flows(dev, g, div, px, py, rx, ry, n_x, n_y, n_s*n_c);
}

BINARY_AUGLAG_GPU_SOLVER_2D::BINARY_AUGLAG_GPU_SOLVER_2D(
	const cudaStream_t & dev,
	const int batch,
    const int n_c,
	const int sizes[2],
	const float* data_cost,
	const float* rx_cost,
	const float* ry_cost,
	float* u,
	float** buffers_full,
	float** buffers_img
):
BINARY_AUGLAG_GPU_SOLVER_BASE(dev, batch, sizes[0]*sizes[1], n_c, data_cost, u, buffers_full, buffers_img),
n_x(sizes[0]),
n_y(sizes[1]),
rx(rx_cost),
ry(ry_cost),
px(buffers_full[4]),
py(buffers_full[5])
{
	//std::cout << rx << std::endl;
	//std::cout << ry << std::endl;
	//std::cout << px << std::endl;
	//std::cout << py << std::endl;
}

