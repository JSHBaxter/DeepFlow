#include "../CPP/hmf_auglag_gpu_solver.h"
#include "../CPP/hmf_auglag1d_gpu_solver.h"
#include "../CPP/hmf_auglag2d_gpu_solver.h"
#include "../CPP/hmf_auglag3d_gpu_solver.h"

#include "../CPP/hmf_meanpass_gpu_solver.h"
#include "../CPP/hmf_meanpass1d_gpu_solver.h"
#include "../CPP/hmf_meanpass2d_gpu_solver.h"
#include "../CPP/hmf_meanpass3d_gpu_solver.h"

#include "../CPP/gpu_kernels.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>

void hmf_auglag_1d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;

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
	int num_buffers_full = HMF_AUGLAG_GPU_SOLVER_1D::num_buffers_full();
	int num_buffers_img = HMF_AUGLAG_GPU_SOLVER_1D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (num_buffers_full*n_sr+num_buffers_img*(n_sr/n_r))*sizeof(float));
	float* buffer_ptr = buffer;
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
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto solver = HMF_AUGLAG_GPU_SOLVER_1D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, out_buf+b*n_s, buffers_full, buffers_img);
		solver();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_auglag_2d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;

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
	float* ry_buf = ry.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = HMF_AUGLAG_GPU_SOLVER_2D::num_buffers_full();
	int num_buffers_img = HMF_AUGLAG_GPU_SOLVER_2D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (num_buffers_full*n_sr+num_buffers_img*(n_s/n_c))*sizeof(float));
	float* buffer_ptr = buffer;
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
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto solver = HMF_AUGLAG_GPU_SOLVER_2D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, out_buf+b*n_s, buffers_full, buffers_img);
		solver();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}


void hmf_auglag_3d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());
	
  
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_z = data.size(4);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y*n_z;
	int n_sr = n_r*n_x*n_y*n_z;
	
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
	float* ry_buf = ry.data_ptr<float>();
	float* rz_buf = rz.data_ptr<float>();

	//get the temporary buffers
	cudaSetDevice(data.get_device());
	int num_buffers_full = HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_full();
	int num_buffers_img = HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (num_buffers_full*n_sr+num_buffers_img*(n_s/n_c))*sizeof(float));
	float* buffer_ptr = buffer;
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
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto solver = HMF_AUGLAG_GPU_SOLVER_3D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, out_buf+b*n_s, buffers_full, buffers_img);
		solver();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_meanpass_1d_gpu(torch::Tensor data, torch::Tensor rx,  torch::Tensor out, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;
	
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
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = HMF_AUGLAG_GPU_SOLVER_1D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, u_init_buf, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = HMF_MEANPASS_GPU_SOLVER_1D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, u_init_buf, out_buf+b*n_s, buffers_full, buffers_img);
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
    
	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;
	
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
	cudaMalloc(&buffer, num_buffers_full*n_sr*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_sr;
	}

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = HMF_MEANPASS_GPU_GRADIENT_1D(dev, bottom_up_list, b, n_c, n_r, data_sizes, rx_buf+b*n_sr, u_buf+b*n_s, g_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_sr, buffers_full);
		grad_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
    
}

void hmf_meanpass_2d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;
	
	//build the tree
	cudaSetDevice(data.get_device());
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
	float* ry_buf = ry.data_ptr<float>();

	//get the temporary buffers
	int num_buffers_full = std::max(HMF_AUGLAG_GPU_SOLVER_2D::num_buffers_full(),HMF_MEANPASS_GPU_SOLVER_2D::num_buffers_full());
	int num_buffers_img = HMF_AUGLAG_GPU_SOLVER_2D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (n_s+num_buffers_full*n_sr+num_buffers_img*(n_s/n_c))*sizeof(float));
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
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto solver_auglag = HMF_AUGLAG_GPU_SOLVER_2D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, u_init_buf, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = HMF_MEANPASS_GPU_SOLVER_2D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, u_init_buf, out_buf+b*n_s, buffers_full, buffers_img);
		solver_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}


void hmf_meanpass_2d_gpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(u.get_device());

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;
	
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
	float* ry_buf = ry.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();
	
	//get the temporary buffers
	int num_buffers_full = HMF_MEANPASS_GPU_GRADIENT_2D::num_buffers_full();
	float* buffer = 0;
	cudaMalloc(&buffer, num_buffers_full*n_sr*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_sr;
	}

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = HMF_MEANPASS_GPU_GRADIENT_2D(dev, bottom_up_list, b, n_c, n_r, data_sizes, rx_buf+b*n_sr, ry_buf+b*n_sr, u_buf+b*n_s, g_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_sr, g_ry_buf+b*n_sr, buffers_full);
		grad_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_meanpass_3d_gpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(data.get_device());
  
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_z = data.size(4);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y*n_z;
	int n_sr = n_r*n_x*n_y*n_z;

	//build the tree
	cudaSetDevice(data.get_device());
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
	float* ry_buf = ry.data_ptr<float>();
	float* rz_buf = rz.data_ptr<float>();

	//get the temporary buffers
	int num_buffers_full = std::max(HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_full(),HMF_MEANPASS_GPU_SOLVER_3D::num_buffers_full());
	int num_buffers_img = HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_images();
	float* buffer = 0;
	cudaMalloc( &buffer, (n_s+num_buffers_full*n_sr+num_buffers_img*(n_s/n_c))*sizeof(float));
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
		buffer_ptr += n_s/n_c;
	}
	
	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto solver_auglag = HMF_AUGLAG_GPU_SOLVER_3D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, u_init_buf, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = HMF_MEANPASS_GPU_SOLVER_3D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, u_init_buf, out_buf+b*n_s, buffers_full, buffers_img);
		solver_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
	delete(buffers_img);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}


void hmf_meanpass_3d_gpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor g_rz, torch::Tensor parentage) {
	cudaStream_t dev = c10::cuda::getCurrentCUDAStream(u.get_device());

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_z = u.size(4);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y*n_z;
	int n_sr = n_r*n_x*n_y*n_z;
	
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
	float* ry_buf = ry.data_ptr<float>();
	float* rz_buf = ry.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();
	
	//get the temporary buffers
	int num_buffers_full = HMF_MEANPASS_GPU_GRADIENT_3D::num_buffers_full();
	float* buffer = 0;
	cudaMalloc(&buffer, num_buffers_full*n_sr*sizeof(float));
	float* buffer_ptr = buffer;
	float** buffers_full = new float* [num_buffers_full];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_sr;
	}

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();
	float* g_rz_buf = g_ry.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = HMF_MEANPASS_GPU_GRADIENT_3D(dev, bottom_up_list, b, n_c, n_r, data_sizes, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, u_buf+b*n_s, g_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_sr, g_ry_buf+b*n_sr, g_rz_buf+b*n_sr,buffers_full);
		grad_meanpass();
	}
	
	//free temporary memory
	cudaFree(buffer);
	delete(buffers_full);
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}


PYBIND11_MODULE(hmf_gpu, m) {
  m.def("auglag_1d_forward", &hmf_auglag_1d_gpu, "hmf_gpu auglag_1d_forward");
  m.def("auglag_2d_forward", &hmf_auglag_2d_gpu, "hmf_gpu auglag_2d_forward");
  m.def("auglag_3d_forward", &hmf_auglag_3d_gpu, "hmf_gpu auglag_3d_forward");
  m.def("meanpass_1d_forward", &hmf_meanpass_1d_gpu, "hmf_gpu meanpass_1d_forward");
  m.def("meanpass_2d_forward", &hmf_meanpass_2d_gpu, "hmf_gpu meanpass_2d_forward");
  m.def("meanpass_3d_forward", &hmf_meanpass_3d_gpu, "hmf_gpu meanpass_3d_forward");
  m.def("meanpass_1d_backward", &hmf_meanpass_1d_gpu_back, "hmf_gpu meanpass_1d_backward");
  m.def("meanpass_2d_backward", &hmf_meanpass_2d_gpu_back, "hmf_gpu meanpass_2d_backward");
  m.def("meanpass_3d_backward", &hmf_meanpass_3d_gpu_back, "hmf_gpu meanpass_3d_backward");
}
