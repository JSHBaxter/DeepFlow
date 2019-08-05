#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "potts_meanpass_cpu_solver.h"
#include "cpu_kernels.h"

class POTTS_MEANPASS_CPU_SOLVER_1D : public POTTS_MEANPASS_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* const rx;
	
protected:
    int min_iter_calc(){
		return n_x;
	}
    void init_vars(){}
    void calculate_regularization(){
		calculate_r_eff(r_eff, rx, u, n_x, n_c);
	}
    void clean_up(){}
	
public:
	POTTS_MEANPASS_CPU_SOLVER_1D(
        const int batch,
        const int sizes[3],
        const float* data_cost,
        const float* rx_cost,
        float* u 
	):
	POTTS_MEANPASS_CPU_SOLVER_BASE(batch, sizes[1], sizes[2], data_cost, u),
	n_x(sizes[1]),
	rx(rx_cost)
	{}
};

class POTTS_MEANPASS_CPU_GRADIENT_1D : public POTTS_MEANPASS_CPU_GRADIENT_BASE
{
private:
    const int n_x;
    const float* const rx;
    float* const g_rx;
	
protected:
    int min_iter_calc(){
		return n_x;
	}
    void init_vars(){
		clear(g_rx, n_c*n_s);
	}
	void get_reg_gradients_and_push(float tau){
		get_reg_gradients(d_y, u, g_rx, n_x, n_c, tau);
		get_gradient_for_u(d_y, rx, g_u, n_x, n_c, tau);
	}
    void clean_up(){}

public:
	POTTS_MEANPASS_CPU_GRADIENT_1D(
        const int batch,
        const int sizes[3],
        const float* u,
        const float* g,
        const float* rx_cost,
        float* g_d,
        float* g_rx
	) :
	POTTS_MEANPASS_CPU_GRADIENT_BASE(batch, sizes[1], sizes[2], u, g, g_d),
	n_x(sizes[1]),
	rx(rx_cost),
	g_rx(g_rx)
	{}
};

template <>
struct PottsMeanpass1dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[3],
      const float* data_cost,
      const float* rx_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      
    int n_batches = sizes[0];
	int n_s = sizes[1];
	int n_c = sizes[2];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(POTTS_MEANPASS_CPU_SOLVER_1D(b, sizes,
																  data_cost+ b*n_s*n_c,
																  rx_cost+ b*n_s*n_c,
																  u+ b*n_s*n_c));
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
struct PottsMeanpass1dGradFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[3],
      const float* data_cost,
      const float* rx_cost,
      const float* u,
      const float* g,
      float* g_data,
      float* g_rx,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/
  ){
	  
    int n_batches = sizes[0];
	int n_s = sizes[1];
	int n_c = sizes[2];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(POTTS_MEANPASS_CPU_GRADIENT_1D(b, sizes,
																	u+ b*n_s*n_c,
																	g+ b*n_s*n_c,
																	rx_cost+ b*n_s*n_c,
																	g_data+ b*n_s*n_c,
																	g_rx+ b*n_s*n_c));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
  }
    
  int num_buffers_full(){ return 0; }
  int num_buffers_images(){ return 0; }
};
