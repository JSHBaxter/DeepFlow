#include "../CPP/hmf_auglag1d_gpu_solver.h"
#include "../CPP/hmf_auglag_gpu_solver.h"
#include "../CPP/hmf_meanpass1d_gpu_solver.h"
#include "../CPP/hmf_meanpass_gpu_solver.h"
#include "../CPP/gpu_kernels.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>
#include <algorithm>

void hmf_meanpass_1d_gpu(torch::Tensor data, torch::Tensor rx,  torch::Tensor out, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());
	
	//ensure Tensor is of float type and 3 dimensional	
	if (int(data.ndimension()) != 3)
	{
		std::cerr << "Data term is the wrong dimensionality." << std::endl;
		return;
	}
	if (int(rx.ndimension()) != 3)
	{
		std::cerr << "Smoohness term is the wrong dimensionality." << std::endl;
		return;
	}

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;
	for(int i = 0; i < 3; i++)
		if (i != 1)
			if (data.size(i) != rx.size(i))
			{
				std::cerr << "Term sizes do not match." << std::endl;
				return;
			}
	
	//build the tree
        int* parentage_b = new int[n_r];
        get_from_gpu(dev, parentage.data_ptr<int>(), parentage_b, n_r*sizeof(int));
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage_b, n_r, n_c);
	
	//get input buffers
	float* data_buf = data.data_ptr<float>();
	float* rx_buf = rx.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = std::max(HMF_AUGLAG_GPU_SOLVER_1D::num_buffers_full(),HMF_MEANPASS_GPU_SOLVER_1D::num_buffers_full());
	int num_buffers_img = HMF_AUGLAG_GPU_SOLVER_1D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (n_s+num_buffers_full*n_sr+num_buffers_img*(n_sr/n_r))*sizeof(float));
	float* u_init_buf = buffer;
	float* buffer_ptr = buffer+n_s;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_sr;
	}
	for(int b = 0; b < num_buffers_img; b++)
	{
		buffers_img[b] = buffer_ptr;
		buffer_ptr += n_sr/n_r;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [5] = {n_b,n_c,n_x,n_r,n_x};
	for(int b = 0; b < n_b; b++){
		std::cout << b << std::endl;
		
		auto solver_auglag = HMF_AUGLAG_GPU_SOLVER_1D(dev, bottom_up_list, b, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, u_init_buf, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = HMF_MEANPASS_GPU_SOLVER_1D(dev, bottom_up_list, b, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, u_init_buf, out_buf+b*n_s, buffers_full, buffers_img);
		solver_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}


void hmf_meanpass_1d_gpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(u.get_device());
	
	//ensure Tensor is of float type and 3 dimensional
	if (int(rx.ndimension()) != 3)
	{
		std::cerr << "Smoohness term is the wrong dimensionality." << std::endl;
		return;
	}
	if (int(u.ndimension()) != 3)
	{
		std::cerr << "Forward pass output is the wrong dimensionality." << std::endl;
		return;
	}
	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;
	for(int i = 0; i < 3; i++)
		if (i != 1)
			if (u.size(i) != rx.size(i))
			{
				std::cerr << "Term sizes do not match." << std::endl;
				return;
			}
	
	//build the tree
	cudaSetDevice(u.get_device());
        int* parentage_b = new int[n_r];
        get_from_gpu(dev, parentage.data_ptr<int>(), parentage_b, n_r*sizeof(int));
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage_b, n_r, n_c);
		
	//get input buffers
	float* rx_buf = rx.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();
	
	//get the temporary buffers
	int num_buffers_full = HMF_MEANPASS_GPU_GRADIENT_1D::num_buffers_full();
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
	int data_sizes [5] = {n_b,n_c,n_x,n_r,n_x};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = HMF_MEANPASS_GPU_GRADIENT_1D(dev, bottom_up_list, b, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_sr, g_data_buf+b*n_s, g_rx_buf+b*n_sr, buffers_full);
		grad_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

PYBIND11_MODULE(hmf_meanpass_1d_gpu, m) {
  m.def("forward", &hmf_meanpass_1d_gpu, "hmf_meanpass_1d_gpu forward");
  m.def("backward", &hmf_meanpass_1d_gpu_back, "hmf_meanpass_1d_gpu backward");
}
