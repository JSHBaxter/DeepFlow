
#include "../CPP/hmf_auglag_cpu_solver.h"
#include "../CPP/hmf_auglag1d_cpu_solver.h"
#include "../CPP/hmf_auglag2d_cpu_solver.h"
#include "../CPP/hmf_auglag3d_cpu_solver.h"

#include "../CPP/hmf_meanpass_cpu_solver.h"
#include "../CPP/hmf_meanpass1d_cpu_solver.h"
#include "../CPP/hmf_meanpass2d_cpu_solver.h"
#include "../CPP/hmf_meanpass3d_cpu_solver.h"

#include "../CPP/cpu_kernels.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

void hmf_auglag_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out, torch::Tensor parentage) {
    
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;

	//build the tree
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();

	//make output tensor
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto solver = HMF_AUGLAG_CPU_SOLVER_1D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, out_buf+b*n_s);
		solver();
	}

	//free temporary memory
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_auglag_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out, torch::Tensor parentage) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;

	//build the tree
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto solver = HMF_AUGLAG_CPU_SOLVER_2D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, out_buf+b*n_s);
		solver();
	}
	//free temporary memory
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_auglag_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out, torch::Tensor parentage) {
	
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
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);

	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();
	const float * const rz_buf = rz.data_ptr<float>();

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto solver = HMF_AUGLAG_CPU_SOLVER_3D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, out_buf+b*n_s);
		solver();
	}

	//free temporary memory
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_meanpass_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out, torch::Tensor parentage) {
    
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;
		
	//build the tree
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = HMF_AUGLAG_CPU_SOLVER_1D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = HMF_MEANPASS_CPU_SOLVER_1D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
	}
	
	//clean up temp buffers
	delete(u_init_buf);
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}



void hmf_meanpass_1d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor parentage) {

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;
	
	//build the tree
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);
	
	//get input buffers
	const float * const rx_buf = rx.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = HMF_MEANPASS_CPU_GRADIENT_1D(true,bottom_up_list, b, n_c, n_r, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_sr, g_data_buf+b*n_s, g_rx_buf+b*n_sr);
		grad_meanpass();
	}
	
	//clean up temp buffers
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_meanpass_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out, torch::Tensor parentage) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;

	//build the tree
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = HMF_AUGLAG_CPU_SOLVER_2D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = HMF_MEANPASS_CPU_SOLVER_2D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
		
	}
	
	//clean up temp buffers
	delete(u_init_buf);
}

void hmf_meanpass_2d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor parentage) {

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;
	
	//build the tree
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);
	
	//get input buffers
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();
    
	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = HMF_MEANPASS_CPU_GRADIENT_2D(true,bottom_up_list, b, n_c, n_r, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, g_data_buf+b*n_s, g_rx_buf+b*n_sr, g_ry_buf+b*n_sr);
		grad_meanpass();
	}
	
	//clean up temp buffers
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_meanpass_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out, torch::Tensor parentage) {
	
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
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);

	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();
	const float * const rz_buf = rz.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = HMF_AUGLAG_CPU_SOLVER_3D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = HMF_MEANPASS_CPU_SOLVER_3D(true,bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
	}
	
	//clean up temp buffers
	delete(u_init_buf);
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_meanpass_3d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor g_rz, torch::Tensor parentage) {

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
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), n_r, n_c);

	//get input buffers
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();
	const float * const rz_buf = rz.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();
	float* g_rz_buf = g_rz.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = HMF_MEANPASS_CPU_GRADIENT_3D(true,bottom_up_list, b, n_c, n_r, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, g_data_buf+b*n_s, g_rx_buf+b*n_sr, g_ry_buf+b*n_sr, g_rz_buf+b*n_sr);
		grad_meanpass();
	}
	
	//clean up temp buffers
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void hmf_cpu_bindings(py::module & m) {
  m.def("hmf_cpu_auglag_1d_forward", &hmf_auglag_1d_cpu, "deepflow hmf_cpu_auglag_1d_forward");
  m.def("hmf_cpu_auglag_2d_forward", &hmf_auglag_2d_cpu, "deepflow hmf_cpu_auglag_2d_forward");
  m.def("hmf_cpu_auglag_3d_forward", &hmf_auglag_3d_cpu, "deepflow hmf_cpu_auglag_3d_forward");
  m.def("hmf_cpu_meanpass_1d_forward", &hmf_meanpass_1d_cpu, "deepflow hmf_cpu_meanpass_1d_forward");
  m.def("hmf_cpu_meanpass_2d_forward", &hmf_meanpass_2d_cpu, "deepflow hmf_cpu_meanpass_2d_forward");
  m.def("hmf_cpu_meanpass_3d_forward", &hmf_meanpass_3d_cpu, "deepflow hmf_cpu_meanpass_3d_forward");
  m.def("hmf_cpu_meanpass_1d_backward", &hmf_meanpass_1d_cpu_back, "deepflow hmf_cpu_meanpass_1d_backward");
  m.def("hmf_cpu_meanpass_2d_backward", &hmf_meanpass_2d_cpu_back, "deepflow hmf_cpu_meanpass_2d_backward");
  m.def("hmf_cpu_meanpass_3d_backward", &hmf_meanpass_3d_cpu_back, "deepflow hmf_cpu_meanpass_3d_backward");
}