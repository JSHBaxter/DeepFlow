#include <iostream>
#include "../CPP/binary_auglag3d_gpu_solver.h"
#include "../CPP/binary_meanpass3d_gpu_solver.h"
#include "../CPP/gpu_kernels.h"

#define DEBUG true

void print_gpu_buffer(cudaStream_t& dev, const float* const buffer, int size){
    if(!DEBUG)
        return;
    float* buffer_c = new float[size];
    get_from_gpu(dev,buffer,buffer_c,size*sizeof(float));
    for(int i = 0; i < size; i++)
        std::cout << buffer_c[i] << " ";
    std::cout << std::endl;
    delete[] buffer_c;
}

int main(int argc, char** argv){
    std::cout << "Starting" << std::endl;
	cudaSetDevice(0);
    std::cout << "Got device" << std::endl;
    cudaStream_t dev; cudaStreamCreate (&dev);
    std::cout << "Made stream" << std::endl;
        
	//get tensor sizing information
	int n_b = 1;
	int n_c = 5;
	int n_x = 4;
	int n_y = 4;
	int n_z = 4;
	int n_s = n_c*n_x*n_y*n_z;
	
    std::cout << "Starting to make input tensors" << std::endl;
    
	//get input buffers
    float * data_buf = 0; cudaMalloc(&data_buf, n_b*n_s*sizeof(float));
	float * rx_buf = 0; cudaMalloc(&rx_buf, n_b*n_s*sizeof(float));
	float * ry_buf = 0; cudaMalloc(&ry_buf, n_b*n_s*sizeof(float));
	float * rz_buf = 0; cudaMalloc(&rz_buf, n_b*n_s*sizeof(float));
	float * u_buf = 0; cudaMalloc(&u_buf, n_b*n_s*sizeof(float));
	float * out_buf = 0; cudaMalloc(&out_buf, n_b*n_s*sizeof(float));
	float * g_buf = 0; cudaMalloc(&g_buf, n_b*n_s*sizeof(float));
    set_buffer(dev, g_buf, 1.0, n_b*n_s);
    clear_buffer(dev, data_buf, n_b*n_s);
    clear_buffer(dev, rx_buf, n_b*n_s);
    clear_buffer(dev, ry_buf, n_b*n_s);
    clear_buffer(dev, rz_buf, n_b*n_s);
    set_buffer(dev, rx_buf, 0.1, n_b*n_s);
    set_buffer(dev, ry_buf, 0.1, n_b*n_s);
    set_buffer(dev, rz_buf, 0.1, n_b*n_s);
    for(int b = 0; b < n_b; b++)
	    set_buffer(dev, data_buf+b*n_s, 1.0, n_s/n_c);
    std::cout << "Made input tensors" << std::endl;
    

	//get the temporary buffers
	int num_buffers_full = std::max(BINARY_AUGLAG_GPU_SOLVER_3D::num_buffers_full(),BINARY_MEANPASS_GPU_SOLVER_3D::num_buffers_full());
	int num_buffers_img = BINARY_AUGLAG_GPU_SOLVER_3D::num_buffers_images();
	float* buffer = 0;
	int success = cudaMalloc( &buffer, (n_s+num_buffers_full*n_s+num_buffers_img*(n_s/n_c))*sizeof(float));
    std::cout << "cudaMalloc code : " << success <<std::endl;
	float* u_init_buf = buffer;
	float* buffer_ptr = buffer+n_s;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}
	for(int b = 0; b < num_buffers_img; b++)
	{
		buffers_img[b] = buffer_ptr;
		buffer_ptr += n_s/n_c;
	}
	
    
	//create and run the solver
    std::cout << "Run forward solvers" << std::endl;
	int data_sizes [3] = {n_x,n_y,n_z};
	for(int b = 0; b < n_b; b++){
		auto solver_auglag = BINARY_AUGLAG_GPU_SOLVER_3D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, u_buf+b*n_s, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_buf+b*n_s,u_buf+b*n_s,n_s);
        
        std::cout << "u" << std::endl;
        print_gpu_buffer(dev,u_buf+b*n_s,n_s);
        std::cout << "\npt" << std::endl;
        print_gpu_buffer(dev,buffers_full[0],n_s);
        std::cout << "\nps" << std::endl;
        print_gpu_buffer(dev,buffers_full[3],n_s);
        std::cout << "\ndiv" << std::endl;
        print_gpu_buffer(dev,buffers_full[1],n_s);
        std::cout << "\ng" << std::endl;
        print_gpu_buffer(dev,buffers_full[2],n_s);
        std::cout << "\npx" << std::endl;
        print_gpu_buffer(dev,buffers_full[4],n_s);
        std::cout << "\npy" << std::endl;
        print_gpu_buffer(dev,buffers_full[5],n_s);
        std::cout << "\npz" << std::endl;
        print_gpu_buffer(dev,buffers_full[6],n_s);
		
		auto solver_meanpass = BINARY_MEANPASS_GPU_SOLVER_3D(dev, b, n_c, data_sizes, data_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, u_buf+b*n_s, out_buf+b*n_s, buffers_full);
		solver_meanpass();
        
        std::cout << "out_buf" << std::endl;
        print_gpu_buffer(dev,out_buf+b*n_s,n_s);
        std::cout << "r_eff" << std::endl;
        print_gpu_buffer(dev,buffers_full[0],n_s);
	}
    
	//free temporary memory
	cudaFree(buffer);
	delete [] buffers_full;
	delete [] buffers_img;
    
	//get the temporary buffers
	num_buffers_full = BINARY_MEANPASS_GPU_GRADIENT_3D::num_buffers_full();
	buffer = 0;
	success = cudaMalloc(&buffer, num_buffers_full*n_s*sizeof(float));
    std::cout << "cudaMalloc code : " << success <<std::endl;
	buffer_ptr = buffer;
	buffers_full = new float* [num_buffers_full];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_s;
	}
	
	//make output tensor  
	float* g_data_buf = 0; cudaMalloc(&g_data_buf, n_b*n_s*sizeof(float));
	float* g_rx_buf = 0; cudaMalloc(&g_rx_buf, n_b*n_s*sizeof(float));
	float* g_ry_buf = 0; cudaMalloc(&g_ry_buf, n_b*n_s*sizeof(float));
    float* g_rz_buf = 0; cudaMalloc(&g_rz_buf, n_b*n_s*sizeof(float));

	//create and run the solver
    std::cout << "Run backward solvers" << std::endl;
	for(int b = 0; b < n_b; b++){
		auto grad_meanpass = BINARY_MEANPASS_GPU_GRADIENT_3D(dev, b, n_c, data_sizes, out_buf+b*n_s, g_buf+b*n_s, rx_buf+b*n_s, ry_buf+b*n_s, rz_buf+b*n_s, g_data_buf+b*n_s, g_rx_buf+b*n_s, g_ry_buf+b*n_s, g_rz_buf+b*n_s, buffers_full);
		grad_meanpass();
        
        std::cout << "g_data" << std::endl;
        print_gpu_buffer(dev,g_data_buf+b*n_s,n_s);
        std::cout << "g_rx" << std::endl;
        print_gpu_buffer(dev,g_rx_buf+b*n_s,n_s);
        std::cout << "g_ry" << std::endl;
        print_gpu_buffer(dev,g_ry_buf+b*n_s,n_s);
        std::cout << "g_rz" << std::endl;
        print_gpu_buffer(dev,g_rz_buf+b*n_s,n_s);
	}
    
	//free temporary memory
	cudaFree(buffer);
	delete [] buffers_full;

    
    
    //free 'images'
    cudaFree(rx_buf);
    cudaFree(ry_buf);
    cudaFree(rz_buf);
    cudaFree(g_rx_buf);
    cudaFree(g_ry_buf);
    cudaFree(g_rz_buf);
    cudaFree(u_buf);
    cudaFree(g_buf);
    cudaFree(g_data_buf);
    
    return 0;   
}