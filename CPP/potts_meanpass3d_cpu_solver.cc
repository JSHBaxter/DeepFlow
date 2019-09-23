#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "potts_meanpass_cpu_solver.h"
#include "cpu_kernels.h"
#include <algorithm>

class POTTS_MEANPASS_CPU_SOLVER_3D : public POTTS_MEANPASS_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* rx;
    const float* ry;
    const float* rz;
	
protected:
    int min_iter_calc(){
		return std::max(std::max(n_x,n_y), n_z);
	}
    void init_vars(){}
    void calculate_regularization(){
		calculate_r_eff(r_eff, rx, ry, rz, u, n_x, n_y, n_z, n_c);
	}
    void clean_up(){}
	
public:
	POTTS_MEANPASS_CPU_SOLVER_3D(
        const int batch,
        const int sizes[5],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        const float* init_u,
        float* u 
	):
	POTTS_MEANPASS_CPU_SOLVER_BASE(batch, sizes[1]*sizes[2]*sizes[3], sizes[4], data_cost, init_u, u),
	n_x(sizes[1]),
	n_y(sizes[2]),
	n_z(sizes[3]),
	rx(rx_cost),
	ry(ry_cost),
	rz(rz_cost)
	{}
};

class POTTS_MEANPASS_CPU_GRADIENT_3D : public POTTS_MEANPASS_CPU_GRADIENT_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* const rx;
    const float* const ry;
    const float* const rz;
    float* const g_rx;
    float* const g_ry;
    float* const g_rz;
	
protected:
    int min_iter_calc(){
		return n_x+n_y+n_z;
	}
    void init_vars(){
		clear(g_rx, g_ry, g_rz, n_c*n_s);
	}
	void get_reg_gradients_and_push(float tau){
		get_reg_gradients(d_y, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c, tau);
		get_gradient_for_u(d_y, rx, ry, rz, g_u, n_x, n_y, n_z, n_c, tau);
	}
    void clean_up(){}

public:
	POTTS_MEANPASS_CPU_GRADIENT_3D(
        const int batch,
        const int sizes[5],
        const float* u,
        const float* g,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* g_d,
        float* g_rx,
        float* g_ry,
        float* g_rz
	) :
	POTTS_MEANPASS_CPU_GRADIENT_BASE(batch, sizes[1]*sizes[2]*sizes[3], sizes[4], u, g, g_d),
	n_x(sizes[1]),
	n_y(sizes[2]),
	n_z(sizes[3]),
	rx(rx_cost),
	ry(ry_cost),
	rz(rz_cost),
	g_rx(g_rx),
	g_ry(g_ry),
	g_rz(g_rz)
	{}
};

template <>
struct PottsMeanpass3dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[5],
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* rz_cost,
      const float* init_u,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    int n_batches = sizes[0];
	int n_s = sizes[1]*sizes[2]*sizes[3];
	int n_c = sizes[4];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(POTTS_MEANPASS_CPU_SOLVER_3D(b, sizes,
																  data_cost + b*n_s*n_c,
																  rx_cost + b*n_s*n_c,
																  ry_cost + b*n_s*n_c,
																  rz_cost + b*n_s*n_c,
                                                                  init_u + (init_u ? b*n_s*n_c : 0),
																  u + b*n_s*n_c));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
  }
  int num_buffers_full(){ return 0; }
  int num_buffers_images(){ return 0; }
};

template <>
struct PottsMeanpass3dGradFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[5],
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* rz_cost,
      const float* u,
      const float* g,
      float* g_data,
      float* g_rx,
      float* g_ry,
      float* g_rz,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/
  ){
	  
    int n_batches = sizes[0];
	int n_s = sizes[1]*sizes[2]*sizes[3];
	int n_c = sizes[4];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(POTTS_MEANPASS_CPU_GRADIENT_3D(b, sizes,
																	u + b*n_s*n_c,
																	g + b*n_s*n_c,
																	rx_cost + b*n_s*n_c,
																	ry_cost + b*n_s*n_c,
																	rz_cost + b*n_s*n_c,
																	g_data + b*n_s*n_c,
																	g_rx + b*n_s*n_c,
																	g_ry + b*n_s*n_c,
																	g_rz + b*n_s*n_c));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
  }
    
  int num_buffers_full(){ return 0; }
  int num_buffers_images(){ return 0; }
};
