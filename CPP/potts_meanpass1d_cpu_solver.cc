#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "cpu_kernels.h"

class PottsMeanpass1d_SolverBatchThreadChannelsLast
{
private:
    const int b;
    const int n_x;
    const int n_c;
    const int n_s;
    const float* data;
    const float* rx;
    float* u;
    
public:
    PottsMeanpass1d_SolverBatchThreadChannelsLast(
        const int batch,
        const int sizes[3],
        const float* data_cost,
        const float* rx_cost,
        float* u ) :
    b(batch),
    n_x(sizes[1]),
    n_c(sizes[2]),
    n_s(sizes[1]),
    data(data_cost),
    rx(rx_cost),
    u(u)
    {}
    
    void operator()(){}
    void operator[](int s){
            
        std::cout << "Inner part" << std::endl;
    
        // optimization constants
        const float beta = 0.01f;
        const float epsilon = 0.01f;
        const float tau = 0.5f;
        
        //buffers shifted to correct batch
        const float* data_b = data + b*n_s*n_c;
        const float* rx_b = rx + b*n_s*n_c;
        float* u_b = u + b*n_s*n_c;
        
        // allocate intermediate variables
        float max_change = 0.0f;
        float* r_eff = new float[n_s*n_c];

        //initialize variables
        softmax(data_b, u_b, n_s, n_c);

        // iterate in blocks
        int min_iter = 10;
        if (n_x > min_iter)
            min_iter = n_x;
        int max_loop = 200;
        for(int i = 0; i < max_loop; i++){    

            //run the solver a set block of iterations
            for (int iter = 0; iter < min_iter; iter++){
                calculate_r_eff(r_eff, rx_b, u_b, n_x, n_c);
                for(int i = 0; i < n_s*n_c; i++)
                    r_eff[i] = data_b[i]+r_eff[i];
                max_change = softmax_with_convergence(r_eff, u_b, n_s, n_c, tau);
            }

            //std::cout << "Thread #:" << b << "\tIter #: " << iter << " \tMax change: " << max_change << std::endl;
            if (max_change < tau*beta)
                break;
        }

        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++){
            calculate_r_eff(r_eff, rx_b, u_b, n_x, n_c);
            for(int i = 0; i < n_s*n_c; i++)
                r_eff[i] = data_b[i]+r_eff[i];
            max_change = softmax_with_convergence(r_eff, u_b, n_s, n_c, tau);
        }

        //calculate the effective regularization
        calculate_r_eff(r_eff, rx_b, u_b, n_x, n_c);
        
        //get final output
        for(int i = 0; i < n_s*n_c; i++)
            u_b[i] = data_b[i]+r_eff[i];
        
        //deallocate temporary buffers
        free(r_eff);
    
    }
};


class PottsMeanpass1d_GradientBatchThreadChannelsLast
{
private:
    const int batch;
    const int n_x;
    const int n_c;
    const int n_s;
    float* g_data;
    float* g_rx;
    float* g_ry;
    const float* logits;
    const float* rx_cost;
    const float* grad;
    
public:
    PottsMeanpass1d_GradientBatchThreadChannelsLast(
        const int batch,
        const int sizes[3],
        const float* u,
        const float* g,
        const float* rx_cost,
        float* g_d,
        float* g_rx) :
    batch(batch),
    n_x(sizes[1]),
    n_c(sizes[2]),
    n_s(sizes[1]),
    g_data(g_d),
    g_rx(g_rx),
    rx_cost(rx_cost),
    logits(u),
    grad(g)
    {}

    inline int idxc (const int s, const int c){
        return c + this->n_c*s;
    }
    inline int idx (const int b, const int x, const int y, const int c){
        return (b*this->n_s*this->n_c) + idxc(x,c);
    }
    
    void get_reg_gradients(const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau){
        for(int x = 0; x < n_x; x++)
        for(int c = 0; c < n_c; c++){

            //for x
            float up_contra = u[idxc(x+1,c)] * g[idxc(x,c)];
            float dn_contra = u[idxc(x,c)] * g[idxc(x+1,c)];
            float derivative = (x < n_x-1) ? up_contra + dn_contra: 0.0f;
            g_rx[idxc(x,c)] += 0.5f * tau * derivative;
        }

    }
    
    void get_gradient_for_u(const float* dy, const float* rx, float* du, const int n_x, const int n_c, const float tau){
        for(int x = 0; x < n_x; x++)
        for(int c = 0; c < n_c; c++){
            float grad_val = 0.0f;
            //x down
            float multiplier = dy[idxc(x-1,c)];
            float inc = 0.5f*multiplier*rx[idxc(x-1,c)];
            grad_val += (x > 0) ? inc: 0.0f;

            //x up
            multiplier = dy[idxc(x+1,c)];
            inc = 0.5f*multiplier*rx[idxc(x,c)];
            grad_val += (x < n_x-1) ? inc: 0.0f;

            du[idxc(x,c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,c)];
        }
    }
    
    void untangle_softmax(const float* g, const float* u, float* dy, const int n_s, const int n_c){
        for(int s = 0; s < n_s; s++)
            for (int c = 0; c < n_c; c++){
                float new_grad = 0.0f;
                float uc = u[idxc(s,c)];
                for(int a = 0; a < n_c; a++){
                    float da = g[idxc(s,a)];
                    if(c == a)
                        new_grad += da*(1.0f-uc);
                    else
                        new_grad -= da*u[idxc(s,a)];
                }
                dy[idxc(s,c)] = new_grad*uc;
        }
    }
    
    void operator()(){
        const float epsilon = 0.00001;
        const float beta = 1e-20;
        const float tau = 0.5;
        
        int b = this->batch;
        float* g_d_b = g_data + b*n_s*n_c;
        float* g_rx_b = g_rx + b*n_s*n_c;
        const float* rx_b = rx_cost + b*n_s*n_c;
        const float* g_b = grad + b*n_s*n_c;
        const float* l_b = logits + b*n_s*n_c;
        float* g_r_eff = g_d_b; 
        
        int max_loops = n_x;
        const int min_iters = 10;
        
        //allocate temporary variables
        float* u = new float[n_s*n_c];
        float* dy = new float[n_s*n_c];
        float* g_u = new float[n_s*n_c];
        
        //transformat logits into labelling
        softmax(l_b, u, n_s, n_c);

        //get initial gradient for the data and regularization terms
        copy(g_b,g_d_b,n_s*n_c);
        clear(g_rx_b,n_s*n_c);
        get_reg_gradients(g_b, u, g_rx_b, n_x, n_c, 1.0f);
         
        //psuh gradient back an iteration
        get_gradient_for_u(g_b, rx_b, g_u, n_x, n_c, 1);
        
        for(int i = 0; i < max_loops; i++){
            for(int iter = 0; iter < min_iters; iter++){
                //untangle softmax
                untangle_softmax(g_u, u, dy, n_s, n_c);
                
                // populate data gradient
                inc(dy,g_d_b,tau,n_s*n_c);

                // populate effective regularization gradient
                get_reg_gradients(g_b, u, g_rx_b, n_x, n_c, tau);

                //push back gradient 
                get_gradient_for_u(dy, rx_b, g_u, n_x, n_c, tau);
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
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(PottsMeanpass1d_SolverBatchThreadChannelsLast(b, sizes, data_cost, rx_cost, u));
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
      
      //std::cout << "On CPU" << std::endl;

    int n_batches = sizes[0];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(PottsMeanpass1d_GradientBatchThreadChannelsLast(b, sizes, u, g, rx_cost, g_data, g_rx));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
  }
    
  int num_buffers_full(){ return 0; }
  int num_buffers_images(){ return 0; }
};
