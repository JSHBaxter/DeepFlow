#include <iostream>
#include "../CPP/hmf_auglag3d_gpu_solver.h"
#include "../CPP/hmf_meanpass3d_gpu_solver.h"
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
    int l = 2;
	int n_c = 1 << l;
	int n_r = (2 << l) - 2;
	int n_x = 2;
	int n_y = 2;
	int n_z = 2;
	int n_s = n_c*n_x*n_y*n_z;
	int n_sr = n_r*n_x*n_y*n_z;
    
    std::cout << n_c << " " << n_r << std::endl;
	
    std::cout << "Starting to make input tensors" << std::endl;
    
	//get input buffers
    float * data_buf = 0; cudaMalloc(&data_buf, n_b*n_s*sizeof(float));
	float * rx_buf = 0; cudaMalloc(&rx_buf, n_b*n_sr*sizeof(float));
	float * ry_buf = 0; cudaMalloc(&ry_buf, n_b*n_sr*sizeof(float));
	float * rz_buf = 0; cudaMalloc(&rz_buf, n_b*n_sr*sizeof(float));
	float * u_buf = 0; cudaMalloc(&u_buf, n_b*n_s*sizeof(float));
	float * out_buf = 0; cudaMalloc(&out_buf, n_b*n_s*sizeof(float));
	float * g_buf = 0; cudaMalloc(&g_buf, n_b*n_s*sizeof(float));
    set_buffer(dev, g_buf, 0.0, n_b*n_s);
    for(int b = 0; b < n_b; b++)
	    set_buffer(dev, g_buf+b*n_s+(n_s-1), 1.0, 1);
    clear_buffer(dev, data_buf, n_b*n_s);
    for(int b = 0; b < n_b; b++)
	    set_buffer(dev, data_buf+b*n_s+(3*n_s/4-1), 10, 1);
    clear_buffer(dev, rx_buf, n_b*n_sr);
    clear_buffer(dev, ry_buf, n_b*n_sr);
    clear_buffer(dev, rz_buf, n_b*n_sr);
    set_buffer(dev, rx_buf, 1.0f / (14.0f+0.0f), n_b*n_sr);
    set_buffer(dev, ry_buf, 1.0f / (14.0f+0.0f), n_b*n_sr);
    set_buffer(dev, rz_buf, 1.0f / (14.0f+0.0f), n_b*n_sr);
    std::cout << "Made input tensors" << std::endl;
    

    //build the tree
    int* parentage = new int[n_r];
    for(int i = 0; i < n_r; i++)
        parentage[i] = (i/2)-1;
	TreeNode* node = NULL;
	TreeNode** children = NULL;
	TreeNode** bottom_up_list = NULL;
	TreeNode** top_down_list = NULL;
	TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, n_r, n_c);
    
	//get the temporary buffers
	int num_buffers_full = std::max(HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_full(),HMF_MEANPASS_GPU_SOLVER_3D::num_buffers_full());
	int num_buffers_img = HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_images();
	float* buffer = 0;
	int success = cudaMalloc( &buffer, (n_sr+num_buffers_full*n_sr+num_buffers_img*(n_s/n_c))*sizeof(float));
    std::cout << "cudaMalloc code : " << success <<std::endl;
	float* u_init_buf = buffer;
	float* buffer_ptr = buffer+n_sr;
	float** buffers_full = new float* [num_buffers_full];
	float** buffers_img = new float* [num_buffers_img];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_sr;
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
		auto solver_auglag = HMF_AUGLAG_GPU_SOLVER_3D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, u_buf+b*n_s, buffers_full, buffers_img);
		solver_auglag();
		exp(dev,u_buf+b*n_s,u_buf+b*n_s,n_s);
        
        std::cout << "data_buf" << std::endl;
        print_gpu_buffer(dev,data_buf+b*n_s,n_s);
        std::cout << "\nrx " << rx_buf << std::endl;
        print_gpu_buffer(dev,rx_buf+b*n_sr,n_sr);
        std::cout << "\nry " << ry_buf << std::endl;
        print_gpu_buffer(dev,ry_buf+b*n_sr,n_sr);
        std::cout << "\nrz " << rz_buf << std::endl;
        print_gpu_buffer(dev,rz_buf+b*n_s,n_s);
        std::cout << "\nu" << std::endl;
        print_gpu_buffer(dev,u_buf+b*n_s,n_s);
        std::cout << "\npt" << std::endl;
        print_gpu_buffer(dev,buffers_full[0],n_s);
        std::cout << "\nps" << std::endl;
        print_gpu_buffer(dev,buffers_img[0],n_s/n_c);
        std::cout << "\nu_tmp" << std::endl;
        print_gpu_buffer(dev,buffers_full[1],n_s);
        std::cout << "\ndiv" << std::endl;
        print_gpu_buffer(dev,buffers_full[2],n_s);
        std::cout << "\ng" << std::endl;
        print_gpu_buffer(dev,buffers_full[3],n_s);
        std::cout << "\npx" << std::endl;
        print_gpu_buffer(dev,buffers_full[4],n_s);
        std::cout << "\npy" << std::endl;
        print_gpu_buffer(dev,buffers_full[5],n_s);
        std::cout << "\npz" << std::endl;
        print_gpu_buffer(dev,buffers_full[6],n_s);
		
		auto solver_meanpass = HMF_MEANPASS_GPU_SOLVER_3D(dev, bottom_up_list, b, n_c, n_r, data_sizes, data_buf+b*n_s, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, u_buf+b*n_s, out_buf+b*n_s, buffers_full, buffers_img);
		solver_meanpass();
        
        std::cout << "\nout_buf" << std::endl;
        print_gpu_buffer(dev,out_buf+b*n_s,n_s);
        std::cout << "\nr_eff" << std::endl;
        print_gpu_buffer(dev,buffers_full[0],n_sr);
        std::cout << "\nu_tmp" << std::endl;
        print_gpu_buffer(dev,buffers_full[1],n_sr);
	}
    
	//free temporary memory
	cudaFree(buffer);
	delete [] buffers_full;
	delete [] buffers_img;
    
	//get the temporary buffers
	num_buffers_full = HMF_MEANPASS_GPU_GRADIENT_3D::num_buffers_full();
	buffer = 0;
	success = cudaMalloc(&buffer, num_buffers_full*n_sr*sizeof(float));
    std::cout << "cudaMalloc code : " << success << "\t" << num_buffers_full << std::endl;
	buffer_ptr = buffer;
	buffers_full = new float* [num_buffers_full];
	for(int b = 0; b < num_buffers_full; b++)
	{
		buffers_full[b] = buffer_ptr;
		buffer_ptr += n_sr;
	}
	
	//make output tensor  
	float* g_data_buf = 0; cudaMalloc(&g_data_buf, n_b*n_s*sizeof(float));
	float* g_rx_buf = 0; cudaMalloc(&g_rx_buf, n_b*n_sr*sizeof(float));
	float* g_ry_buf = 0; cudaMalloc(&g_ry_buf, n_b*n_sr*sizeof(float));
    float* g_rz_buf = 0; cudaMalloc(&g_rz_buf, n_b*n_sr*sizeof(float));

	//create and run the solver
    std::cout << "Run backward solvers" << std::endl;
	for(int b = 0; b < n_b; b++){
        
        std::cout << "\nu " << buffers_full[0] << std::endl;
        print_gpu_buffer(dev,buffers_full[0],n_sr);
        std::cout << "\ndy " << buffers_full[1] << std::endl;
        print_gpu_buffer(dev,buffers_full[1],n_sr);
        std::cout << "\ndu " << buffers_full[2] << std::endl;
        print_gpu_buffer(dev,buffers_full[2],n_sr);
        std::cout << "\ntmp " << buffers_full[3] << std::endl;
        print_gpu_buffer(dev,buffers_full[3],n_sr);
        
        
        std::cout << rx_buf+b*n_sr << " " << ry_buf+b*n_sr << " " << rz_buf+b*n_sr << std::endl;
		auto grad_meanpass = HMF_MEANPASS_GPU_GRADIENT_3D(dev, bottom_up_list, b, n_c, n_r, data_sizes, rx_buf+b*n_sr, ry_buf+b*n_sr, rz_buf+b*n_sr, out_buf+b*n_s, g_buf+b*n_s,  g_data_buf+b*n_s, g_rx_buf+b*n_sr, g_ry_buf+b*n_sr, g_rz_buf+b*n_sr, buffers_full);
		grad_meanpass();
        
        std::cout << "\ng_data " << g_data_buf << std::endl;
        print_gpu_buffer(dev,g_data_buf+b*n_s,n_s);
        std::cout << "\ng_rx " << g_rx_buf << std::endl;
        print_gpu_buffer(dev,g_rx_buf+b*n_sr,n_sr);
        std::cout << "\ng_ry " << g_ry_buf << std::endl;
        print_gpu_buffer(dev,g_ry_buf+b*n_sr,n_sr);
        std::cout << "\ng_rz " << g_rz_buf << std::endl;
        print_gpu_buffer(dev,g_rz_buf+b*n_sr,n_sr);
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
    delete[] parentage;
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
    
    return 0;   
}