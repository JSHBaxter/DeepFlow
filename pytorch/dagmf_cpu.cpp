
#include "../CPP/dagmf_auglag_solver.h"
#include "../CPP/dagmf_meanpass_solver.h"

#include "../CPP/cpu_kernels.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

void dagmf_auglag_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };
    
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;

	//build the tree
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();

	//make output tensor
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {data_buf + b*n_s, rx_buf + b*n_sr};
		auto solver = new DAGMF_AUGLAG_SOLVER<CPU_DEVICE>(dev, bottom_up_list, false, 1, data_sizes, n_c, n_r, inputs, out_buf+b*n_s);
		solver->run();
        delete solver;
	}

	//free temporary memory
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_auglag_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;

	//build the tree
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const ry_buf = ry.data_ptr<float>();

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {data_buf + b*n_s, rx_buf + b*n_sr, ry_buf + b*n_sr};
		auto solver = new DAGMF_AUGLAG_SOLVER<CPU_DEVICE>(dev, bottom_up_list, false, 2, data_sizes, n_c, n_r, inputs, out_buf+b*n_s);
		solver->run();
        delete solver;
	}
	//free temporary memory
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_auglag_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };
	
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
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);

	
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
        const float* const inputs[] = {data_buf + b*n_s, rx_buf + b*n_sr, ry_buf + b*n_sr, rz_buf + b*n_sr};
		auto solver = new DAGMF_AUGLAG_SOLVER<CPU_DEVICE>(dev, bottom_up_list, false, 3, data_sizes, n_c, n_r, inputs, out_buf+b*n_s);
		solver->run();
        delete solver;
	}

	//free temporary memory
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_meanpass_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };
    
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;
		
	//build the tree
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
        float* const buffers[] = {u_init_buf, data_buf + b*n_s, rx_buf + b*n_sr};
		
		auto solver_auglag = new DAGMF_AUGLAG_SOLVER<CPU_DEVICE>(dev, bottom_up_list, false, 1, data_sizes, n_c, n_r, buffers+1, u_init_buf);
		solver_auglag->run();
        delete solver_auglag;
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = new DAGMF_MEANPASS_SOLVER<CPU_DEVICE>(dev, bottom_up_list, 1, data_sizes, n_c, n_r, buffers, out_buf+b*n_s);
		solver_meanpass->run();
        delete solver_meanpass;
	}
	
	//clean up temp buffers
	delete [] u_init_buf;
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_meanpass_1d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_r = rx.size(1);
	int n_s = n_c*n_x;
	int n_sr = n_r*n_x;
	
	//build the tree
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);
	
	//get input buffers
	const float* const rx_buf = rx.data_ptr<float>();
	const float* const u_buf = u.data_ptr<float>();
	const float* const g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {g_buf + b*n_s, u_buf + b*n_s*n_c, rx_buf + b*n_sr};
        float* const outputs[] = {g_data_buf + b*n_s, g_rx_buf + b*n_sr};
		auto grad_meanpass = new DAGMF_MEANPASS_GRADIENT<CPU_DEVICE>(dev, bottom_up_list, 1, data_sizes, n_c, n_r, inputs, outputs);
		grad_meanpass->run();
        delete grad_meanpass;
	}
	
	//clean up temp buffers
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_meanpass_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;

	//build the tree
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
        float* const buffers[] = {u_init_buf, data_buf + b*n_s, rx_buf + b*n_sr, ry_buf + b*n_sr};
		
		auto solver_auglag = new DAGMF_AUGLAG_SOLVER<CPU_DEVICE>(dev, bottom_up_list, false, 2, data_sizes, n_c, n_r, buffers+1, u_init_buf);
		solver_auglag->run();
        delete solver_auglag;
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = new DAGMF_MEANPASS_SOLVER<CPU_DEVICE>(dev, bottom_up_list, 2, data_sizes, n_c, n_r, buffers, out_buf+b*n_s);
		solver_meanpass->run();
		delete solver_meanpass;
	}
	
	//clean up temp buffers
	delete [] u_init_buf;
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_meanpass_2d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_r = rx.size(1);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;
	
	//build the tree
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);
	
	//get input buffers
	const float* const rx_buf = rx.data_ptr<float>();
	const float* const ry_buf = ry.data_ptr<float>();
	const float* const u_buf = u.data_ptr<float>();
	const float* const g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();
    
	//create and run the solver
	int data_sizes [2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {g_buf + b*n_s, u_buf + b*n_s*n_c, rx_buf + b*n_sr, ry_buf + b*n_sr};
        float* const outputs[] = {g_data_buf + b*n_s, g_rx_buf + b*n_sr, g_ry_buf + b*n_sr};
		auto grad_meanpass = new DAGMF_MEANPASS_GRADIENT<CPU_DEVICE>(dev, bottom_up_list, 2, data_sizes, n_c, n_r, inputs, outputs);
		grad_meanpass->run();
        delete grad_meanpass;
	}
	
	//clean up temp buffers
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_meanpass_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };
	
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
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);

	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();
	float * const rz_buf = rz.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
        float* const buffers[] = {u_init_buf, data_buf + b*n_s, rx_buf + b*n_sr, ry_buf + b*n_sr, rz_buf + b*n_sr};
		
		auto solver_auglag = new DAGMF_AUGLAG_SOLVER<CPU_DEVICE>(dev, bottom_up_list, false, 3, data_sizes, n_c, n_r, buffers+1, u_init_buf);
		solver_auglag->run();
        delete solver_auglag;
		exp(dev,u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = new DAGMF_MEANPASS_SOLVER<CPU_DEVICE>(dev, bottom_up_list, 3, data_sizes, n_c, n_r, buffers, out_buf+b*n_s);
		solver_meanpass->run();
        delete solver_meanpass;
	}
	
	//clean up temp buffers
	delete [] u_init_buf;
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_meanpass_3d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor g_rz, torch::Tensor parentage) {
    CPU_DEVICE dev = { .channels_first = true };

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
	DAGNode* node = NULL;
	DAGNode** children = NULL;
	DAGNode** bottom_up_list = NULL;
	DAGNode** top_down_list = NULL;
	DAGNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<float>(), n_r, n_c);

	//get input buffers
	const float* const rx_buf = rx.data_ptr<float>();
	const float* const ry_buf = ry.data_ptr<float>();
	const float* const rz_buf = rz.data_ptr<float>();
	const float* const u_buf = u.data_ptr<float>();
	const float* const g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();
	float* g_rz_buf = g_rz.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {g_buf + b*n_s, u_buf + b*n_s*n_c, rx_buf + b*n_sr, ry_buf + b*n_sr, rz_buf + b*n_sr};
        float* const outputs[] = {g_data_buf + b*n_s, g_rx_buf + b*n_sr, g_ry_buf + b*n_sr, g_rz_buf + b*n_sr};
		auto grad_meanpass = new DAGMF_MEANPASS_GRADIENT<CPU_DEVICE>(dev, bottom_up_list, 3, data_sizes, n_c, n_r, inputs, outputs);
		grad_meanpass->run();
        delete grad_meanpass;
	}
	
	//clean up temp buffers
    DAGNode::free_tree(node, children, bottom_up_list, top_down_list);
}

void dagmf_cpu_bindings(py::module & m) {
  m.def("dagmf_auglag_1d_cpu_forward", &dagmf_auglag_1d_cpu, "deepflow dagmf_auglag_1d_cpu_forward");
  m.def("dagmf_auglag_2d_cpu_forward", &dagmf_auglag_2d_cpu, "deepflow dagmf_auglag_2d_cpu_forward");
  m.def("dagmf_auglag_3d_cpu_forward", &dagmf_auglag_3d_cpu, "deepflow dagmf_auglag_3d_cpu_forward");
  m.def("dagmf_meanpass_1d_cpu_forward", &dagmf_meanpass_1d_cpu, "deepflow dagmf_meanpass_1d_cpu_forward");
  m.def("dagmf_meanpass_2d_cpu_forward", &dagmf_meanpass_2d_cpu, "deepflow dagmf_meanpass_2d_cpu_forward");
  m.def("dagmf_meanpass_3d_cpu_forward", &dagmf_meanpass_3d_cpu, "deepflow dagmf_meanpass_3d_cpu_forward");
  m.def("dagmf_meanpass_1d_cpu_backward", &dagmf_meanpass_1d_cpu_back, "deepflow dagmf_meanpass_1d_cpu_backward");
  m.def("dagmf_meanpass_2d_cpu_backward", &dagmf_meanpass_2d_cpu_back, "deepflow dagmf_meanpass_2d_cpu_backward");
  m.def("dagmf_meanpass_3d_cpu_backward", &dagmf_meanpass_3d_cpu_back, "deepflow dagmf_meanpass_3d_cpu_backward");
}