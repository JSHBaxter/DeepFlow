
#include <thread>
#include "potts_meanpass1d_cpu_solver.h"

template <>
struct PottsMeanpass1dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[3],
      const float* data_cost,
      const float* rx_cost,
      const float* init_u,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      
    int n_batches = sizes[0];
	int n_s = sizes[1];
	int n_c = sizes[2];
    int data_sizes[1] = {sizes[1]};
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(POTTS_MEANPASS_CPU_SOLVER_1D(false, b, n_c, data_sizes,
																  data_cost+ b*n_s*n_c,
																  rx_cost+ b*n_s*n_c,
                                                                  init_u + (init_u ? b*n_s*n_c : 0),
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
    int data_sizes[1] = {sizes[1]};
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(POTTS_MEANPASS_CPU_GRADIENT_1D(false, b, n_c, data_sizes,
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
