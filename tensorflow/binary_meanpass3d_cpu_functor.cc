#include <thread>

#include "binary_meanpass3d_cpu_solver.h"

template <>
struct BinaryMeanpass3dFunctor<CPUDevice> {
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
    int data_sizes[3] = {sizes[1],sizes[2],sizes[3]};
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(BINARY_MEANPASS_CPU_SOLVER_3D(false, b, n_c, data_sizes,
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
struct BinaryMeanpass3dGradFunctor<CPUDevice> {
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
    int data_sizes[3] = {sizes[1],sizes[2],sizes[3]};
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(BINARY_MEANPASS_CPU_GRADIENT_3D(false, b, n_c, data_sizes,
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

