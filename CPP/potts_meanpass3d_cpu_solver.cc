#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "cpu_kernels.h"

class PottsMeanpass3d_SolverBatchThreadChannelsLast
{
private:
    const int b;
    const int n_x;
    const int n_y;
    const int n_z;
    const int n_c;
    const int n_s;
    const float* data;
    const float* rx;
    const float* ry;
    const float* rz;
    float* u;
	float* r_eff;
    
	// optimization constants
	const float beta = 0.01f;
	const float epsilon = 0.01f;
	const float tau = 0.5f;
    
public:
    PottsMeanpass3d_SolverBatchThreadChannelsLast(
        const int batch,
        const int sizes[5],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* u ) :
    b(batch),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_z(sizes[3]),
    n_c(sizes[4]),
    n_s(sizes[1]*sizes[2]*sizes[3]),
    data(data_cost),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost),
    u(u)
    {}
    
	float run_block(int iter, int min_iter){
		float max_change = 0.0f;
		calculate_r_eff(r_eff, rx, ry, rz, u, n_x, n_y, n_z, n_c);
		for(int i = 0; i < n_s*n_c; i++)
			r_eff[i] = data[i]+r_eff[i];
		if( iter == min_iter - 1)
			max_change = softmax_with_convergence(r_eff, u, n_s, n_c, tau);
		else
			softmax_update(r_eff, u, n_s, n_c, tau);
		return max_change;
	}
	
    void operator()(){
        
        // allocate intermediate variables
        float max_change = 0.0f;
        r_eff = new float[n_s*n_c];

        //initialize variables
        softmax(data, u, n_s, n_c);

        // iterate in blocks
        int min_iter = 10;
        if (n_x > min_iter)
            min_iter = n_x;
        if (n_y > min_iter)
            min_iter = n_y;
        if (n_z > min_iter)
            min_iter = n_z;
        int max_loop = 200;
        for(int i = 0; i < max_loop; i++){    

            //run the solver a set block of iterations
            for (int iter = 0; iter < min_iter; iter++)
				max_change = run_block(iter, min_iter);

            //std::cout << "Thread #:" << b << "\tIter #: " << iter << " \tMax change: " << max_change << std::endl;
            if (max_change < tau*beta)
                break;
        }

        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++)
			run_block(iter,0);

        //calculate the effective regularization
        calculate_r_eff(r_eff, rx, ry, rz, u, n_x, n_y, n_z, n_c);
        
        //get final output
        for(int i = 0; i < n_s*n_c; i++)
            u[i] = data[i]+r_eff[i];
        
        //deallocate temporary buffers
        free(r_eff);
    
    }
};


class PottsMeanpass3d_GradientBatchThreadChannelsLast
{
private:
    const int batch;
    const int n_x;
    const int n_y;
    const int n_z;
    const int n_c;
    const int n_s;
    float* g_data;
    float* g_rx;
    float* g_ry;
    float* g_rz;
    const float* logits;
    const float* rx;
    const float* ry;
    const float* rz;
    const float* grad;
    
	const float epsilon = 0.00001;
	const float beta = 1e-20;
	const float tau = 0.5;
	
public:
    PottsMeanpass3d_GradientBatchThreadChannelsLast(
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
        float* g_rz) :
    batch(batch),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_z(sizes[3]),
    n_c(sizes[4]),
    n_s(sizes[1]*sizes[2]*sizes[3]),
    g_data(g_d),
    g_rx(g_rx),
    g_ry(g_ry),
    g_rz(g_rz),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost),
    logits(u),
    grad(g)
    {}
    
    void operator()(){
        
        int b = this->batch;
        
        int max_loops = n_x+n_y+n_z;
        const int min_iters = 10;
        
        //allocate temporary variables
        float* u = new float[n_s*n_c];
        float* dy = new float[n_s*n_c];
        float* g_u = new float[n_s*n_c];
        
        //transformat logits into labelling
        softmax(logits, u, n_s, n_c);

        //get initial gradient for the data and regularization terms
        copy(grad,g_data,n_s*n_c);
        clear(g_rx,g_ry,g_rz,n_s*n_c);
        get_reg_gradients(grad, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c, 1.0f);
         
        //psuh gradient back an iteration
        get_gradient_for_u(grad, rx, ry, rz, g_u, n_x, n_y, n_z, n_c, 1);
        
        for(int i = 0; i < max_loops; i++){
            for(int iter = 0; iter < min_iters; iter++){
                //untangle softmax
                untangle_softmax(g_u, u, dy, n_s, n_c);
                
                // populate data gradient
                inc(dy,g_data,tau,n_s*n_c);

                // populate effective regularization gradient
                get_reg_gradients(dy, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_c, tau);

                //push back gradient 
                get_gradient_for_u(dy, rx, ry, rz, g_u, n_x, n_y, n_z, n_c, tau);
            }
            
            //get max of gu and break if converged
            float gu_max = maxabs(g_u, n_s*n_c);
            if( gu_max < beta )
                break;
        }
        
        delete u;
        delete dy;
        delete g_u;
    }
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
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    int n_batches = sizes[0];
	int n_s = sizes[1]*sizes[2]*sizes[3];
	int n_c = sizes[4];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(PottsMeanpass3d_SolverBatchThreadChannelsLast(b, sizes, data_cost + b*n_s*n_c,
																																									  rx_cost + b*n_s*n_c,
																																									  ry_cost + b*n_s*n_c,
																																									  rz_cost + b*n_s*n_c,
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
      
      //std::cout << "On CPU" << std::endl;

    int n_batches = sizes[0];
	int n_s = sizes[1]*sizes[2]*sizes[3];
	int n_c = sizes[4];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(PottsMeanpass3d_GradientBatchThreadChannelsLast(b, sizes, u + b*n_s*n_c,
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
