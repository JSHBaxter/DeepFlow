
#include "../CPP/potts_auglag_cpu_solver.h"
#include "../CPP/potts_auglag1d_cpu_solver.h"
#include "../CPP/potts_auglag2d_cpu_solver.h"
#include "../CPP/potts_auglag3d_cpu_solver.h"

#include "../CPP/potts_meanpass_cpu_solver.h"
#include "../CPP/potts_meanpass1d_cpu_solver.h"
#include "../CPP/potts_meanpass2d_cpu_solver.h"
#include "../CPP/potts_meanpass3d_cpu_solver.h"

#include "../CPP/cpu_kernels.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <iostream>

void potts_auglag_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	float* data_buf = data.data_ptr<float>();
	float* rx_buf = rx.data_ptr<float>();

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto solver = POTTS_AUGLAG_CPU_SOLVER_1D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, out_buf+b*n_s);
		solver();
	}
}

void potts_auglag_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_s = n_c*n_x*n_y;
	
	//get input buffers
	float* data_buf = data.data_ptr<float>();
	float* rx_buf = rx.data_ptr<float>();
	float* ry_buf = ry.data_ptr<float>();

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x, n_y};
	for(int b = 0; b < n_b; b++){
		auto solver = POTTS_AUGLAG_CPU_SOLVER_2D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, out_buf+b*n_s);
		solver();
	}
}

void potts_auglag_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_z = data.size(4);
	int n_s = n_c*n_x*n_y*n_z;
    
	//get input buffers
	float* data_buf = data.data_ptr<float>();
	float* rx_buf = rx.data_ptr<float>();
	float* ry_buf = ry.data_ptr<float>();
	float* rz_buf = rz.data_ptr<float>();

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x, n_y, n_z};
	for(int b = 0; b < n_b; b++){
		auto solver = POTTS_AUGLAG_CPU_SOLVER_3D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, out_buf+b*n_s);
		solver();
	}
}

void potts_meanpass_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out) {
	
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	float* data_buf = data.data_ptr<float>();
	float* rx_buf = rx.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = POTTS_AUGLAG_CPU_SOLVER_1D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = POTTS_MEANPASS_CPU_SOLVER_1D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
	}
	
	//clean up temp buffers
	delete(u_init_buf);
}



void potts_meanpass_1d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor g_data, torch::Tensor g_rx) {
	
	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_s = n_c*n_x;
	
	//get input buffers
	float* rx_buf = rx.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();

	//create and run the solver
	int data_sizes [1] = {n_x};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = POTTS_MEANPASS_CPU_GRADIENT_1D(true, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s);
		grad_meanpass();
	}
}

void potts_meanpass_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_s = n_c*n_x*n_y;
	
	//get input buffers
	float* data_buf = data.data_ptr<float>();
	float* rx_buf = rx.data_ptr<float>();
	float* ry_buf = ry.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x, n_y};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = POTTS_AUGLAG_CPU_SOLVER_2D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = POTTS_MEANPASS_CPU_SOLVER_2D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
		
	}
	
	//clean up temp buffers
	delete(u_init_buf);
}

void potts_meanpass_2d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry) {

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_s = n_c*n_x*n_y;
	
	//get input buffers
	float* rx_buf = rx.data_ptr<float>();
	float* ry_buf = ry.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();

	//create and run the solver
	int data_sizes [2] = {n_x, n_y};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = POTTS_MEANPASS_CPU_GRADIENT_2D(true, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s, g_ry_buf+b*n_s);
		grad_meanpass();
	}
	
	//return
	return;
}

void potts_meanpass_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out) {

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_z = data.size(4);
	int n_s = n_c*n_x*n_y*n_z;
	
	//get input buffers
	float* data_buf = data.data_ptr<float>();
	float* rx_buf = rx.data_ptr<float>();
	float* ry_buf = ry.data_ptr<float>();
	float* rz_buf = rz.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf = new float[n_s];

	//make output tensor  
	float* out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x, n_y, n_z};
	for(int b = 0; b < n_b; b++){
		
		auto solver_auglag = POTTS_AUGLAG_CPU_SOLVER_3D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, u_init_buf);
		solver_auglag();
		exp(u_init_buf,u_init_buf,n_s);
		
		auto solver_meanpass = POTTS_MEANPASS_CPU_SOLVER_3D(true, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, u_init_buf, out_buf+b*n_s);
		solver_meanpass();
	}
	
	//clean up temp buffers
	delete(u_init_buf);
}

void potts_meanpass_3d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor g_rz) {

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_z = u.size(4);
	int n_s = n_c*n_x*n_y*n_z;
	
	//get input buffers
	float* rx_buf = rx.data_ptr<float>();
	float* ry_buf = ry.data_ptr<float>();
	float* rz_buf = ry.data_ptr<float>();
	float* u_buf = u.data_ptr<float>();
	float* g_buf = g.data_ptr<float>();

	//make output tensor  
	float* g_data_buf = g_data.data_ptr<float>();
	float* g_rx_buf = g_rx.data_ptr<float>();
	float* g_ry_buf = g_ry.data_ptr<float>();
	float* g_rz_buf = g_rz.data_ptr<float>();

	//create and run the solver
	int data_sizes [3] = {n_x, n_y, n_z};
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = POTTS_MEANPASS_CPU_GRADIENT_3D(true, b, n_c, data_sizes, u_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s, g_ry_buf+b*n_s, g_rz_buf+b*n_s);
		grad_meanpass();
	}
}

PYBIND11_MODULE(potts_cpu, m) {
  m.def("auglag_1d_forward", &potts_auglag_1d_cpu, "potts_cpu auglag_1d_forward");
  m.def("auglag_2d_forward", &potts_auglag_2d_cpu, "potts_cpu auglag_2d_forward");
  m.def("auglag_3d_forward", &potts_auglag_3d_cpu, "potts_cpu auglag_3d_forward");
  m.def("meanpass_1d_forward", &potts_meanpass_1d_cpu, "potts_cpu meanpass_1d_forward");
  m.def("meanpass_2d_forward", &potts_meanpass_2d_cpu, "potts_cpu meanpass_2d_forward");
  m.def("meanpass_3d_forward", &potts_meanpass_3d_cpu, "potts_cpu meanpass_3d_forward");
  m.def("meanpass_1d_backward", &potts_meanpass_1d_cpu_back, "potts_cpu meanpass_1d_backward");
  m.def("meanpass_2d_backward", &potts_meanpass_2d_cpu_back, "potts_cpu meanpass_2d_backward");
  m.def("meanpass_3d_backward", &potts_meanpass_3d_cpu_back, "potts_cpu meanpass_3d_backward");
}