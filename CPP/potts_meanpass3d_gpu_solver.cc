#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"


#include "gpu_kernels.h"

#include <math.h> 	
#include <thread>

template <>
struct PottsMeanpass3dFunctor<GPUDevice>{
    
    void operator()(
            const GPUDevice& d,
            int sizes[5],
            const float* data_cost,
            const float* rx_cost,
            const float* ry_cost,
            const float* rz_cost,
            float* u,
            float** buffers_full,
            float** /*unused image buffers*/){

        //if we got channels last on GPU, send error message
        // TODO

        int n_bat = sizes[0];
        int n_c = sizes[1];
        int n_x = sizes[2];
        int n_y = sizes[3];
        int n_z = sizes[4];
        int n_s = n_x*n_y*n_z;

        float* temp = buffers_full[0];

        // optimization constants
        const float tau = 0.5f;
        const float beta = 0.01f;
        const float epsilon = 10e-5f;

        for(int b = 0; b < n_bat; b++){
            //std::cout << "Batch: " << b << std::endl;

            const float* data_b = data_cost + b*n_s*n_c;
            const float* rx_b = rx_cost + b*n_s*n_c;
            const float* ry_b = ry_cost + b*n_s*n_c;
            const float* rz_b = rz_cost + b*n_s*n_c;
            float* u_b = u + b*n_s*n_c;

            //initialize variables
            softmax(d, data_b, NULL, u_b, n_s, n_c);

            // iterate in blocks
            int min_iter = 10;
            if (n_x > min_iter)
                min_iter = n_x;
            if (n_y > min_iter)
                min_iter = n_y;
            if (n_z > min_iter)
                min_iter = n_z;
            int max_loop = 200;
            for(int i = 0; i < max_loop; i++) {

                //Each iteration consists of:
                // - calculating the effective regularization (store answer in temp)
                // - getting new probability estimates, and normalize (store answer in temp)
                // - updating probability estimates and get convergence criteria (stored in temp)
                for(int iter = 0; iter < min_iter; iter++){
                    get_effective_reg(d, temp, u_b, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_c);
                    softmax(d, data_b, temp, temp, n_s, n_c);
                    change_to_diff(d, u_b, temp, n_s*n_c, tau);
                }

                //get the max change
                float max_change = max_of_buffer(d, temp, n_c*n_s);

                //std::cout << "Iter #: " << iter << " Max change: " << max_change << std::endl;
                if (max_change < tau*beta)
                    break;

            }

            //extra block for good measure
            for(int iter = 0; iter < min_iter; iter++){
                get_effective_reg(d, temp, u_b, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_c);
                softmax(d, data_b, temp, temp, n_s, n_c);
                change_to_diff(d, u_b, temp, n_s*n_c, tau);
            }

            //one last pseudo-iteration
            get_effective_reg(d, temp, u_b, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_c);
            add_then_store(d, data_b, temp, u_b, n_s*n_c);
        }

    }    
    
    
    int num_buffers_full(){ return 1; }
    int num_buffers_images(){ return 0; }

};

template <>
struct PottsMeanpass3dGradFunctor<GPUDevice>{

    void operator()(
            const GPUDevice& d,
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
            float** buffers_full,
            float** /*unused image buffers*/
    ){

        //std::cout << "\t" << buffers_full << std::endl;

        const float epsilon = 0.00001;
        const float beta = 1e-20;
        const float tau = 0.5;
        int n_bat = sizes[0];
        int n_c = sizes[1];
        int n_x = sizes[2];
        int n_y = sizes[3];
        int n_z = sizes[4];
        int n_s = n_x*n_y*n_z;

        float* du_i = buffers_full[0];
        float* dy = buffers_full[1];
        float* u_tmp = buffers_full[2];

        //std::cout << du_i << std::endl;
        //std::cout << g_data << std::endl;

        //std::cout << "\t" << du_i << std::endl;
        //std::cout << "\t" << dy << std::endl;
        //std::cout << "\t" << u_tmp << std::endl;

        int max_loops = n_x+n_y+n_z;
        const int min_iters = 10;

        for (int b = 0; b < n_bat; b++){

            // create easier pointers
            const float* g_b = g+b*n_c*n_s;
            const float* u_b = u+b*n_c*n_s;
            const float* rx_b = rx_cost+b*n_c*n_s;
            const float* ry_b = ry_cost+b*n_c*n_s;
            const float* rz_b = rz_cost+b*n_c*n_s;
            float* g_d_b = g_data+b*n_c*n_s;
            float* g_rx_b = g_rx+b*n_c*n_s;
            float* g_ry_b = g_ry+b*n_c*n_s;
            float* g_rz_b = g_rz+b*n_c*n_s;

            //get initial gradient for the final logits
            softmax(d, u_b, NULL, u_tmp, n_s, n_c);
            copy_buffer(d, g_b, g_d_b, n_s*n_c);
            populate_reg_mean_gradients(d, g_b, u_tmp, g_rx_b, g_ry_b, g_rz_b, n_x, n_y, n_z, n_c);

            //push gradients back a number of iterations
            float max_change = 0.0f;
            get_gradient_for_u(d, g_b, du_i, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_c);
            
            for(int i = 0; i < max_loops; i++){
                for(int iter = 0; iter < min_iters; iter++){
                    process_grad_potts(d, du_i, u_tmp, dy, n_s, n_c, tau);
                    populate_reg_mean_gradients_and_add(d, dy, u_tmp, g_rx_b, g_ry_b, g_rz_b, n_x, n_y, n_z, n_c);
                    inc_buffer(d, dy, g_d_b, n_s*n_c);
                    get_gradient_for_u(d, dy, dy, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_c);
                    mult_buffer(d, 1.0f-tau, du_i, n_s*n_c);
                    inc_buffer(d, dy, du_i, n_s*n_c);
                }

                copy_buffer(d, du_i, dy, n_s*n_c);
                max_change = max_of_buffer(d, dy, n_c*n_s);
                //std::cout << "Batch " << b << " iteration " << min_iters*(i+1) << "du max " << max_change <<  ": \t" << max_change << std::endl;
                if(max_change < beta)
                    break;
            }

            //std::cout << "Batch " << b << " iteration " << max_loops*i << ": \t" << max_change << std::endl;

        }
    }

    int num_buffers_full(){ return 3; }
    int num_buffers_images(){ return 0; }
    
};

#endif // GOOGLE_CUDA
                           