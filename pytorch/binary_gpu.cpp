
#include "../CPP/binary_auglag_gpu_solver.h"
#include "../CPP/binary_auglag1d_gpu_solver.h"
#include "../CPP/binary_auglag2d_gpu_solver.h"
#include "../CPP/binary_auglag3d_gpu_solver.h"

#include "../CPP/binary_meanpass_gpu_solver.h"
#include "../CPP/binary_meanpass1d_gpu_solver.h"
#include "../CPP/binary_meanpass2d_gpu_solver.h"
#include "../CPP/binary_meanpass3d_gpu_solver.h"

#include "../CPP/gpu_kernels.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <c10/cuda/CUDAStream.h>

#include <iostream>
#include <algorithm>

void binary_auglag_1d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = BINARY_AUGLAG_GPU_SOLVER_1D::num_buffers_full();
	int num_buffers_img = BINARY_AUGLAG_GPU_SOLVER_1D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (num_buffers_full*n_s+num_buffers_img*(n_s/n_c))*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}
	for(int b = 0; b < num_buffers_img; b++)
	{
		buffers_img[b] = buffer_ptr;
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto solver = BINARY_AUGLAG_GPU_SOLVER_1D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, out_buf+b*n_s, buffers_full, buffers_img);
		solver();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
}

void binary_auglag_2d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_s = n_c*n_x*n_y;
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = BINARY_AUGLAG_GPU_SOLVER_2D::num_buffers_full();
	int num_buffers_img = BINARY_AUGLAG_GPU_SOLVER_2D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (num_buffers_full*n_s+num_buffers_img*(n_s/n_c))*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}
	for(int b = 0; b < num_buffers_img; b++)
	{
		buffers_img[b] = buffer_ptr;
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto solver = BINARY_AUGLAG_GPU_SOLVER_2D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, out_buf+b*n_s, buffers_full, buffers_img);
		solver();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
}

void binary_auglag_3d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());
	
  
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_z = data.size(4);
	int n_s = n_c*n_x*n_y*n_z;
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();
	const float * const rz_buf = rz.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = BINARY_AUGLAG_GPU_SOLVER_3D::num_buffers_full();
	int num_buffers_img = BINARY_AUGLAG_GPU_SOLVER_3D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (num_buffers_full*n_s+num_buffers_img*(n_s/n_c))*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}
	for(int b = 0; b < num_buffers_img; b++)
	{
		buffers_img[b] = buffer_ptr;
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto solver = BINARY_AUGLAG_GPU_SOLVER_3D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, out_buf+b*n_s, buffers_full, buffers_img);
		solver();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
}

void binary_meanpass_1d_gpu(torch::Tensor data, torch::Tensor rx,  torch::Tensor out) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = std::max(BINARY_AUGLAG_GPU_SOLVER_1D::num_buffers_full(),BINARY_MEANPASS_GPU_SOLVER_1D::num_buffers_full());
	int num_buffers_img = BINARY_AUGLAG_GPU_SOLVER_1D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (n_s+num_buffers_full*n_s+num_buffers_img*(n_s/n_c))*sizeof(float));
	float* u_init_buf = buffer;
	float* buffer_ptr = buffer+n_s;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}
	for(int b = 0; b < num_buffers_img; b++)
	{
		buffers_img[b] = buffer_ptr;
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = BINARY_AUGLAG_GPU_SOLVER_1D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, u_init_buf, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = BINARY_MEANPASS_GPU_SOLVER_1D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, u_init_buf, out_buf+b*n_s, buffers_full);
		solver_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
}


void binary_meanpass_1d_gpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor g_data, torch::Tensor g_rx) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(u.get_device());

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const u_buf = u.data_ptr<float>();
	const float * const g_buf = g.data_ptr<float>();
	
	//get the temporary buffers
	cudaSetDevice(u.get_device());
	int num_buffers_full = BINARY_MEANPASS_GPU_GRADIENT_1D::num_buffers_full();
	float* buffer = 0;
	cudaMalloc(&buffer, num_buffers_full*n_s*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = BINARY_MEANPASS_GPU_GRADIENT_1D(dev, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s, buffers_full);
		grad_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
}

void binary_meanpass_2d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_s = n_c*n_x*n_y;
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = std::max(BINARY_AUGLAG_GPU_SOLVER_2D::num_buffers_full(),BINARY_MEANPASS_GPU_SOLVER_2D::num_buffers_full());
	int num_buffers_img = BINARY_AUGLAG_GPU_SOLVER_2D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (n_s+num_buffers_full*n_s+num_buffers_img*(n_s/n_c))*sizeof(float));
	float* u_init_buf = buffer;
	float* buffer_ptr = buffer+n_s;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}
	for(int b = 0; b < num_buffers_img; b++)
	{
		buffers_img[b] = buffer_ptr;
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto solver_auglag = BINARY_AUGLAG_GPU_SOLVER_2D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, u_init_buf, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = BINARY_MEANPASS_GPU_SOLVER_2D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, u_init_buf, out_buf+b*n_s, buffers_full);
		solver_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
}


void binary_meanpass_2d_gpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(u.get_device());

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_s = n_c*n_x*n_y;
	
	//get input buffers
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();
	const float * const u_buf = u.data_ptr<float>();
	const float * const g_buf = g.data_ptr<float>();
	
	//get the temporary buffers
	cudaSetDevice(u.get_device());
	int num_buffers_full = BINARY_MEANPASS_GPU_GRADIENT_2D::num_buffers_full();
	float* buffer = 0;
	cudaMalloc(&buffer, num_buffers_full*n_s*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = BINARY_MEANPASS_GPU_GRADIENT_2D(dev, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s, g_ry_buf+b*n_s, buffers_full);
		grad_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
}

void binary_meanpass_3d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());
  
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_z = data.size(4);
	int n_s = n_c*n_x*n_y*n_z;
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();
	const float * const rz_buf = rz.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = std::max(BINARY_AUGLAG_GPU_SOLVER_3D::num_buffers_full(),BINARY_MEANPASS_GPU_SOLVER_3D::num_buffers_full());
	int num_buffers_img = BINARY_AUGLAG_GPU_SOLVER_3D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (n_s+num_buffers_full*n_s+num_buffers_img*(n_s/n_c))*sizeof(float));
	float* u_init_buf = buffer;
	float* buffer_ptr = buffer+n_s;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}
	for(int b = 0; b < num_buffers_img; b++)
	{
		buffers_img[b] = buffer_ptr;
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto solver_auglag = BINARY_AUGLAG_GPU_SOLVER_3D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, u_init_buf, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = BINARY_MEANPASS_GPU_SOLVER_3D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, u_init_buf, out_buf+b*n_s, buffers_full);
		solver_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
}


void binary_meanpass_3d_gpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor g_rz) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(u.get_device());

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_z = u.size(4);
	int n_s = n_c*n_x*n_y*n_z;
	
	//get input buffers
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();
	const float * const rz_buf = rz.data_ptr<float>();
	const float * const u_buf = u.data_ptr<float>();
	const float * const g_buf = g.data_ptr<float>();
	
	//get the temporary buffers
	cudaSetDevice(u.get_device());
	int num_buffers_full = BINARY_MEANPASS_GPU_GRADIENT_3D::num_buffers_full();
	float* buffer = 0;
	cudaMalloc(&buffer, num_buffers_full*n_s*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();
	float* g_rz_buf = g_rz.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = BINARY_MEANPASS_GPU_GRADIENT_3D(dev, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s, g_ry_buf+b*n_s, g_rz_buf+b*n_s,buffers_full);
		grad_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
}

void binary_gpu_bindings(py::module & m) {
  m.def("binary_gpu_auglag_1d_forward", &binary_auglag_1d_gpu, "deepflow binary_gpu_auglag_1d_forward");
  m.def("binary_gpu_auglag_2d_forward", &binary_auglag_2d_gpu, "deepflow binary_gpu_auglag_2d_forward");
  m.def("binary_gpu_auglag_3d_forward", &binary_auglag_3d_gpu, "deepflow binary_gpu_auglag_3d_forward");
  m.def("binary_gpu_meanpass_1d_forward", &binary_meanpass_1d_gpu, "deepflow binary_gpu_meanpass_1d_forward");
  m.def("binary_gpu_meanpass_2d_forward", &binary_meanpass_2d_gpu, "deepflow binary_gpu_meanpass_2d_forward");
  m.def("binary_gpu_meanpass_3d_forward", &binary_meanpass_3d_gpu, "deepflow binary_gpu_meanpass_3d_forward");
  m.def("binary_gpu_meanpass_1d_backward", &binary_meanpass_1d_gpu_back, "deepflow binary_gpu_meanpass_1d_backward");
  m.def("binary_gpu_meanpass_2d_backward", &binary_meanpass_2d_gpu_back, "deepflow binary_gpu_meanpass_2d_backward");
  m.def("binary_gpu_meanpass_3d_backward", &binary_meanpass_3d_gpu_back, "deepflow binary_gpu_meanpass_3d_backward");
}
