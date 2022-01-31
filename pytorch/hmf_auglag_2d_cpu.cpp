#include "../CPP/hmf_auglag2d_cpu_solver.h"
#include "../CPP/hmf_auglag_cpu_solver.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <iostream>

void hmf_auglag_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out, torch::Tensor parentage, torch::Tensor data_index) {
	
	//ensure Tensor is of float type and 3 dimensional	
	if (int(data.ndimension()) != 4)
	{
		std::cerr << "Data term is the wrong dimensionality." << std::endl;
		return;
	}
	if (int(rx.ndimension()) != 4)
	{
		std::cerr << "Smoohness term is the wrong dimensionality." << std::endl;
		return;
	}
	if (int(ry.ndimension()) != 4)
	{
		std::cerr << "Smoohness term is the wrong dimensionality." << std::endl;
		return;
	}

	//get tensor sizing information
	int n_b = data.size(0);
	int n_x = data.size(1);
	int n_y = data.size(2);
	int n_c = data.size(3);
	int n_r = rx.size(3);
	int n_s = n_c*n_x*n_y;
	int n_sr = n_r*n_x*n_y;
	for(int i = 0; i < 4; i++)
		if (i == 3){
			if (rx.size(i) != ry.size(i))
			{
				std::cerr << "Term sizes do not match." << std::endl;
				return;
			}

		}else{
			if (data.size(i) != rx.size(i) || data.size(i) != ry.size(i))
			{
				std::cerr << "Term sizes do not match." << std::endl;
				return;
			}
		}

	//build the tree
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage.data_ptr<int>(), data_index.data_ptr<int>(), n_r, n_c);
	
	//get input buffers
	float* data_buf = data.data_ptr<float>();
	float* rx_buf = rx.data_ptr<float>();
	float* ry_buf = ry.data_ptr<float>();

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [6] = {n_b,n_x,n_y,n_c,n_x,n_r};
	for(int b = 0; b < n_b; b++){
		auto solver = HMF_AUGLAG_CPU_SOLVER_2D(bottom_up_list, b, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, out_buf+b*n_s);
		solver();
	}
	//free temporary memory
        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
}

PYBIND11_MODULE(hmf_auglag_2d_cpu, m) {
  m.def("forward", &hmf_auglag_2d_cpu, "hmf_auglag_2d_cpu forward");
}