
#include "../CPP/potts_auglag_solver.h"
#include "../CPP/potts_meanpass_solver.h"

#include "../CPP/algorithm.h"
#include "../CPP/cpu_kernels.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

void potts_auglag_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out) {
    CPU_DEVICE dev = { .channels_first = true };
    
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_s = n_x;
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();
    
	//create and run the solver
	int data_sizes[1] = {n_x};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {data_buf + b*n_s*n_c, rx_buf + b*n_s*n_c};
        float* const out_buff_curr = out_buf + b*n_s*n_c;
		auto solver = new POTTS_AUGLAG_SOLVER<CPU_DEVICE>(dev, false, 1, data_sizes, n_c, inputs, out_buff_curr);
		solver->run();
        delete solver;
	}
}

void potts_auglag_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out) {
    CPU_DEVICE dev = { .channels_first = true };

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_s = n_x*n_y;
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {data_buf + b*n_s*n_c, rx_buf + b*n_s*n_c, ry_buf + b*n_s*n_c};
        float* const out_buff_curr = out_buf + b*n_s*n_c;
		auto solver = new POTTS_AUGLAG_SOLVER<CPU_DEVICE>(dev, false, 2, data_sizes, n_c, inputs, out_buff_curr);
		solver->run();
        delete solver;
	}
}

void potts_auglag_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out) {
    CPU_DEVICE dev = { .channels_first = true };

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_z = data.size(4);
	int n_s = n_x*n_y*n_z;
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();
	float * const rz_buf = rz.data_ptr<float>();

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {data_buf + b*n_s*n_c, rx_buf + b*n_s*n_c, ry_buf + b*n_s*n_c, rz_buf + b*n_s*n_c};
        float* const out_buff_curr = out_buf + b*n_s*n_c;
		auto solver = new POTTS_AUGLAG_SOLVER<CPU_DEVICE>(dev, false, 3, data_sizes, n_c, inputs, out_buff_curr);
		solver->run();
        delete solver;
	}
}

void potts_meanpass_1d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor out) {
    CPU_DEVICE dev = { .channels_first = true };

	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_s = n_x;
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf =new float[n_c*n_s];

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[1] = {n_x};
	for(int b = 0; b < n_b; b++){
        float* const buffers[] = {u_init_buf, data_buf + b*n_s*n_c, rx_buf + b*n_s*n_c};
		
		auto solver_auglag = new POTTS_AUGLAG_SOLVER<CPU_DEVICE>(dev, false, 1, data_sizes, n_c, buffers+1, buffers[0]);
		solver_auglag->run();
		exp(dev,u_init_buf,u_init_buf,n_s*n_c);
        delete solver_auglag;
		
		auto solver_meanpass = new POTTS_MEANPASS_SOLVER<CPU_DEVICE>(dev, 1, data_sizes, n_c, buffers, out_buf+b*n_s*n_c);
		solver_meanpass->run();
        delete solver_meanpass;
	}
	
	//clean up temp buffers
	delete [] u_init_buf;
}

void potts_meanpass_1d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor g_data, torch::Tensor g_rx) {
    CPU_DEVICE dev = { .channels_first = true };

	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_s = n_x;
	
	//get input buffers
	float * const rx_buf = rx.data_ptr<float>();
	float * const u_buf = u.data_ptr<float>();
	float * const g_buf = g.data_ptr<float>();
    
	//make output tensor  
	float * const g_data_buf = g_data.data_ptr<float>();
	float * const g_rx_buf = g_rx.data_ptr<float>();

	//create and run the solver
	int data_sizes[1] = {n_x};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {g_buf + b*n_s*n_c, u_buf + b*n_s*n_c, rx_buf + b*n_s*n_c};
        float* const outputs[] = {g_data_buf + b*n_s*n_c, g_rx_buf + b*n_s*n_c};
		auto grad_meanpass = new POTTS_MEANPASS_GRADIENT<CPU_DEVICE>(dev, 1, data_sizes, n_c, inputs, outputs);
		grad_meanpass->run();
        delete grad_meanpass;
	}
}

void potts_meanpass_2d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor out) {
    CPU_DEVICE dev = { .channels_first = true };
    
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_s = n_x*n_y;
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf =new float[n_c*n_s];

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
        float* const buffers[] = {u_init_buf, data_buf + b*n_s*n_c, rx_buf + b*n_s*n_c, ry_buf + b*n_s*n_c};
		
		auto solver_auglag = new POTTS_AUGLAG_SOLVER<CPU_DEVICE>(dev, false, 2, data_sizes, n_c, buffers+1, buffers[0]);
		solver_auglag->run();
		exp(dev,u_init_buf,u_init_buf,n_s*n_c);
        delete solver_auglag;
		
		auto solver_meanpass = new POTTS_MEANPASS_SOLVER<CPU_DEVICE>(dev, 2, data_sizes, n_c, buffers, out_buf+b*n_s*n_c);
		solver_meanpass->run();
        delete solver_meanpass;
	}
	
	//clean up temp buffers
	delete [] u_init_buf;
}

void potts_meanpass_2d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry) {
    CPU_DEVICE dev = { .channels_first = true };
    
	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_s = n_x*n_y;
	
	//get input buffers
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();
	float * const u_buf = u.data_ptr<float>();
	float * const g_buf = g.data_ptr<float>();

	//make output tensor  
	float * const g_data_buf = g_data.data_ptr<float>();
	float * const g_rx_buf = g_rx.data_ptr<float>();
	float * const g_ry_buf = g_ry.data_ptr<float>();

	//create and run the solver
	int data_sizes[2] = {n_x,n_y};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {g_buf + b*n_s*n_c, u_buf + b*n_s*n_c, rx_buf + b*n_s*n_c, ry_buf + b*n_s*n_c};
        float* const outputs[] = {g_data_buf + b*n_s*n_c, g_rx_buf + b*n_s*n_c, g_ry_buf + b*n_s*n_c};
		auto grad_meanpass = new POTTS_MEANPASS_GRADIENT<CPU_DEVICE>(dev, 2, data_sizes, n_c, inputs, outputs);
		grad_meanpass->run();
        delete grad_meanpass;
	}
	
	//return
	return;
}

void potts_meanpass_3d_cpu(torch::Tensor data, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor out) {
    CPU_DEVICE dev = { .channels_first = true };
    
	//get tensor sizing information
	int n_b = data.size(0);
	int n_c = data.size(1);
	int n_x = data.size(2);
	int n_y = data.size(3);
	int n_z = data.size(4);
	int n_s = n_x*n_y*n_z;
	
	//get input buffers
	float * const data_buf = data.data_ptr<float>();
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();
	float * const rz_buf = rz.data_ptr<float>();

	//get buffer for MAP solution (used as initialisation)
	float* u_init_buf =new float[n_c*n_s];

	//make output tensor  
	float * const out_buf = out.data_ptr<float>();

	//create and run the solver
	int data_sizes[3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
        float* const buffers[] = {u_init_buf, data_buf + b*n_s*n_c, rx_buf + b*n_s*n_c, ry_buf + b*n_s*n_c, rz_buf + b*n_s*n_c};
		
		auto solver_auglag = new POTTS_AUGLAG_SOLVER<CPU_DEVICE>(dev, false, 3, data_sizes, n_c, buffers+1, buffers[0]);
		solver_auglag->run();
		exp(dev,u_init_buf,u_init_buf,n_s*n_c);
        delete solver_auglag;
		
		auto solver_meanpass = new POTTS_MEANPASS_SOLVER<CPU_DEVICE>(dev, 3, data_sizes, n_c, buffers, out_buf+b*n_s*n_c);
		solver_meanpass->run();
        delete solver_meanpass;
	}
	
	//clean up temp buffers
	delete [] u_init_buf;
}

void potts_meanpass_3d_cpu_back(torch::Tensor g, torch::Tensor u, torch::Tensor rx, torch::Tensor ry, torch::Tensor rz, torch::Tensor g_data, torch::Tensor g_rx, torch::Tensor g_ry, torch::Tensor g_rz) {
    CPU_DEVICE dev = { .channels_first = true };
    
	//get tensor sizing information
	int n_b = u.size(0);
	int n_c = u.size(1);
	int n_x = u.size(2);
	int n_y = u.size(3);
	int n_z = u.size(4);
	int n_s = n_x*n_y*n_z;
	
	//get input buffers
	float * const rx_buf = rx.data_ptr<float>();
	float * const ry_buf = ry.data_ptr<float>();
	float * const rz_buf = rz.data_ptr<float>();
	float * const u_buf = u.data_ptr<float>();
	float * const g_buf = g.data_ptr<float>();

	//make output tensor  
	float * const g_data_buf = g_data.data_ptr<float>();
	float * const g_rx_buf = g_rx.data_ptr<float>();
	float * const g_ry_buf = g_ry.data_ptr<float>();
	float * const g_rz_buf = g_rz.data_ptr<float>();

	//create and run the solver
	int data_sizes[3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
        const float* const inputs[] = {g_buf + b*n_s*n_c, u_buf + b*n_s*n_c, rx_buf + b*n_s*n_c, ry_buf + b*n_s*n_c, rz_buf + b*n_s*n_c};
        float* const outputs[] = {g_data_buf + b*n_s*n_c, g_rx_buf + b*n_s*n_c, g_ry_buf + b*n_s*n_c, g_rz_buf + b*n_s*n_c};
		auto grad_meanpass = new POTTS_MEANPASS_GRADIENT<CPU_DEVICE>(dev, 3, data_sizes, n_c, inputs, outputs);
		grad_meanpass->run();
        delete grad_meanpass;
	}
}

void potts_cpu_bindings(py::module & m) {
  m.def("potts_auglag_1d_cpu_forward", &potts_auglag_1d_cpu, "deepflow potts_auglag_1d_cpu_forward");
  m.def("potts_auglag_2d_cpu_forward", &potts_auglag_2d_cpu, "deepflow potts_auglag_2d_cpu_forward");
  m.def("potts_auglag_3d_cpu_forward", &potts_auglag_3d_cpu, "deepflow potts_auglag_3d_cpu_forward");
  m.def("potts_meanpass_1d_cpu_forward", &potts_meanpass_1d_cpu, "deepflow potts_meanpass_1d_cpu_forward");
  m.def("potts_meanpass_2d_cpu_forward", &potts_meanpass_2d_cpu, "deepflow potts_meanpass_2d_cpu_forward");
  m.def("potts_meanpass_3d_cpu_forward", &potts_meanpass_3d_cpu, "deepflow potts_meanpass_3d_cpu_forward");
  m.def("potts_meanpass_1d_cpu_backward", &potts_meanpass_1d_cpu_back, "deepflow potts_meanpass_1d_cpu_backward");
  m.def("potts_meanpass_2d_cpu_backward", &potts_meanpass_2d_cpu_back, "deepflow potts_meanpass_2d_cpu_backward");
  m.def("potts_meanpass_3d_cpu_backward", &potts_meanpass_3d_cpu_back, "deepflow potts_meanpass_3d_cpu_backward");
}
