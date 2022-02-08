
#include "../CPP/binary_auglag_cpu_solver.h"
#include "../CPP/binary_auglag1d_cpu_solver.h"
#include "../CPP/binary_auglag2d_cpu_solver.h"
#include "../CPP/binary_auglag3d_cpu_solver.h"

#include "../CPP/binary_meanpass_cpu_solver.h"
#include "../CPP/binary_meanpass1d_cpu_solver.h"
#include "../CPP/binary_meanpass2d_cpu_solver.h"
#include "../CPP/binary_meanpass3d_cpu_solver.h"

#include "../CPP/cpu_kernels.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

void binary_auglag_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out) {
    
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();
    
	//create and run the solver
	int data_sizes[1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto solver = BINARY_AUGLAG_CPU_SOLVER_1D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, out_buf+b*n_s);
		solver();
	}
}

void binary_auglag_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out) {

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

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto solver = BINARY_AUGLAG_CPU_SOLVER_2D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, out_buf+b*n_s);
		solver();
	}
}

void binary_auglag_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out) {

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

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto solver = BINARY_AUGLAG_CPU_SOLVER_3D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, out_buf+b*n_s);
		solver();
	}
}

void binary_meanpass_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	const float * const data_buf = data.data_ptr<float>();
	const float * const rx_buf = rx.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float * const u_init_buf = new float[n_s];

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[1] = {n_x};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = BINARY_AUGLAG_CPU_SOLVER_1D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = BINARY_MEANPASS_CPU_SOLVER_1D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
	}
	
	//clean up temp buffers
	delete(u_init_buf);
}

void binary_meanpass_1d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor g_data, torch::Tensor g_rx) {

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	const float * const rx_buf = rx.data_ptr<float>();
	const float * const u_buf = u.data_ptr<float>();
	const float * const g_buf = g.data_ptr<float>();

	//make output tensor  
	float * const g_data_buf = g_data.data_ptr<float>();
	float * const g_rx_buf = g_rx.data_ptr<float>();

	//create and run the solver
	int data_sizes[1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = BINARY_MEANPASS_CPU_GRADIENT_1D(true, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s);
		grad_meanpass();
	}
}

void binary_meanpass_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_s = n_c*n_x*n_y;
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float * const u_init_buf = new float[n_s];

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = BINARY_AUGLAG_CPU_SOLVER_2D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = BINARY_MEANPASS_CPU_SOLVER_2D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
		
	}
	
	//clean up temp buffers
	delete(u_init_buf);
}

void binary_meanpass_2d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry) {
	
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

	//make output tensor  
	float * const g_data_buf = g_data.data_ptr<float>();
	float * const g_rx_buf = g_rx.data_ptr<float>();
	float * const g_ry_buf = g_ry.data_ptr<float>();

	//create and run the solver
	int data_sizes[2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = BINARY_MEANPASS_CPU_GRADIENT_2D(true, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s, g_ry_buf+b*n_s);
		grad_meanpass();
	}
	
	//return
	return;
}

void binary_meanpass_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out) {

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

	//get buffer for MAP solution (used as initialisation)
	float * const u_init_buf = new float[n_s];

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = BINARY_AUGLAG_CPU_SOLVER_3D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = BINARY_MEANPASS_CPU_SOLVER_3D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
	}
	
	//clean up temp buffers
	delete(u_init_buf);
}

void binary_meanpass_3d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor g_rz) {
	
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

	//make output tensor  
	float * const g_data_buf = g_data.data_ptr<float>();
	float * const g_rx_buf = g_rx.data_ptr<float>();
	float * const g_ry_buf = g_ry.data_ptr<float>();
	float * const g_rz_buf = g_rz.data_ptr<float>();

	//create and run the solver
	int data_sizes[3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = BINARY_MEANPASS_CPU_GRADIENT_3D(true, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s, g_ry_buf+b*n_s, g_rz_buf+b*n_s);
		grad_meanpass();
	}
}

void binary_cpu_bindings(py::module & m) {
  m.def("binary_cpu_auglag_1d_forward", &binary_auglag_1d_cpu, "deepflow binary_cpu_auglag_1d_forward");
  m.def("binary_cpu_auglag_2d_forward", &binary_auglag_2d_cpu, "deepflow binary_cpu_auglag_2d_forward");
  m.def("binary_cpu_auglag_3d_forward", &binary_auglag_3d_cpu, "deepflow binary_cpu_auglag_3d_forward");
  m.def("binary_cpu_meanpass_1d_forward", &binary_meanpass_1d_cpu, "deepflow binary_cpu_meanpass_1d_forward");
  m.def("binary_cpu_meanpass_2d_forward", &binary_meanpass_2d_cpu, "deepflow binary_cpu_meanpass_2d_forward");
  m.def("binary_cpu_meanpass_3d_forward", &binary_meanpass_3d_cpu, "deepflow binary_cpu_meanpass_3d_forward");
  m.def("binary_cpu_meanpass_1d_backward", &binary_meanpass_1d_cpu_back, "deepflow binary_cpu_meanpass_1d_backward");
  m.def("binary_cpu_meanpass_2d_backward", &binary_meanpass_2d_cpu_back, "deepflow binary_cpu_meanpass_2d_backward");
  m.def("binary_cpu_meanpass_3d_backward", &binary_meanpass_3d_cpu_back, "deepflow binary_cpu_meanpass_3d_backward");
}
