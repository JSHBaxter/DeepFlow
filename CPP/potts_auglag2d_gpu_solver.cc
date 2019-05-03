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
struct PottsAuglag2dFunctor<GPUDevice>{
    
    void operator()(
            const GPUDevice& d,
            int sizes[4],
            const float* data_cost,
            const float* rx_cost,
            const float* ry_cost,
            float* u,
            float** buffers_full,
            float** buffers_img){

        //if we got channels last on GPU, send error message
        // TODO

        int n_bat = sizes[0];
        int n_c = sizes[1];
        int n_x = sizes[2];
        int n_y = sizes[3];
        int n_s = n_x*n_y;

        float* ps = buffers_img[0];
        float* pt = buffers_full[0];
        float* g = buffers_full[1];
        float* div = buffers_full[2];
        float* px = buffers_full[3];
        float* py = buffers_full[4];

        // optimization constants
        const float beta = 0.001f;
        const float tau = 0.1f;
        const float cc = 0.125;
        const float icc = 1.0f / cc;

        for(int b = 0; b < n_bat; b++){
            //std::cout << "Batch: " << b << std::endl;

            const float* data_b = data_cost + b*n_s*n_c;
            const float* rx_b = rx_cost + b*n_s*n_c;
            const float* ry_b = ry_cost + b*n_s*n_c;
            float* u_b = u + b*n_s*n_c;

            //initialize variables
            //softmax(d, data_b, NULL, u_b, n_s, n_c);
            clear_buffer(d, u_b, n_s*n_c);
            clear_buffer(d, px, n_s*n_c);
            clear_buffer(d, py, n_s*n_c);
            clear_buffer(d, div, n_s*n_c);
            clear_buffer(d, g, n_s*n_c);
            //clear_buffer(d, ps, n_s);
            //clear_buffer(d, pt, n_s*n_c);
            find_min_constraint(d, ps, data_b, n_c, n_s);
            rep_buffer(d, ps, pt, n_c, n_s);

            // iterate in blocks
            int min_iter = 10;
            if (n_x > min_iter)
                min_iter = n_x;
            if (n_y > min_iter)
                min_iter = n_y;
            int max_loop = 200;
            for(int i = 0; i < max_loop; i++) {

                //Each iteration consists of:
                // - updating the spatial flows and recalculating the divergence
                // - updating the source flow and sink flow and constraining them
                // - updating the multipliers, saving the change in a seperate buffer
                for(int iter = 0; iter < min_iter; iter++){
                    calc_capacity_potts(d, g, div, ps, pt, u_b, n_s, n_c, icc, tau);
                    update_spatial_flows(d, g, div, px, py, rx_b, ry_b, n_x, n_y, n_s*n_c);
                    update_source_sink_multiplier_potts(d, ps, pt, div, u_b, g, data_b, cc, icc, n_c, n_s);
                    
                    //get the max change
                    float max_change = max_of_buffer(d, g, n_c*n_s);

                    std::cout << "Iter # : " << iter << " Max change: " << max_change << std::endl;

                }

                //get the max change
                float max_change = max_of_buffer(d, g, n_c*n_s);
                if (max_change < tau*beta)
                    break;

            }

            //extra block for good measure
            for(int iter = 0; iter < min_iter; iter++){
                calc_capacity_potts(d, g, div, ps, pt, u_b, n_s, n_c, icc, tau);
                update_spatial_flows(d, g, div, px, py, rx_b, ry_b, n_x, n_y, n_s*n_c);
                update_source_sink_multiplier_potts(d, ps, pt, div, u_b, g, data_b, cc, icc, n_c, n_s);
            }

            //do logarithm on u
            log_buffer(d,u_b,u_b,n_c*n_s);
        }

    }    
    
    
    int num_buffers_full(){ return 5; }
    int num_buffers_images(){ return 1; }

};

#endif // GOOGLE_CUDA
                           