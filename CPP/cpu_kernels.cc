#include "cpu_kernels.h"
#include <limits>
#include <iostream>

#define epsilon 0.00001f

#include <cmath>
void clear(float* buffer, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer[i] = 0.0f;
}
void clear(float* buffer1, float* buffer2, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer1[i] = buffer2[i] = 0.0f;
}
void clear(float* buffer1, float* buffer2, float* buffer3, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer1[i] = buffer2[i] = buffer3[i] = 0.0f;
}

void print_buffer(float* buffer, const int n_s){
	for(int i = 0; i < n_s; i++)
		std::cout << buffer[i] << " ";
	std::cout << std::endl;
}
void set(float* buffer, const float number, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer[i] = number;
}

void copy(const float* bufferin, float* bufferout, const int n_s){
    for(int i = 0; i < n_s; i++)
        bufferout[i] = bufferin[i];
}

void inc(const float* inc, float* acc, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] += inc[i];
}

void ninc(const float* inc, float* acc, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] -= inc[i];
}

void inc(const float* inc, float* acc, const float alpha, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] += alpha*inc[i];
}

void log_buffer(float* buffer, const int n_s){
    for(int i = 0; i < n_s; i++)
		if( buffer[i] < epsilon )
			buffer[i] = log(epsilon);
		else
			buffer[i] = log(buffer[i]);
}

void div_buffer(float* buffer, const float number, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer[i] /= number;
}
void mult_buffer(float* buffer, const float number, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer[i] *= number;
}

void constrain(float* buffer, const float* constraint, const int n_s){
    for(int i = 0; i < n_s; i++)
        if(buffer[i] > constraint[i])
            buffer[i] = constraint[i];
}

float maxabs(const float* buffer, const int n_s){
    float maxabs = 0.0f;
    for(int i = 0; i < n_s; i++){
        if( buffer[i] < -maxabs )
            maxabs = -buffer[i];
        if( buffer[i] > maxabs )
            maxabs = buffer[i];
    }
    return maxabs;
}

float max_diff(const float* buffer, const int n_c, const int n_s){
    float avg_all_diff = 0.0;
    for(int s = 0; s < n_s; s++){
        float max_diff = 0.0;
        for(int c1 = 0; c1 < n_c; c1++)
        for(int c2 = 0; c2 < n_c; c2++)
            if( buffer[s*n_c+c1]-buffer[s*n_c+c2] > max_diff )
                max_diff = buffer[s*n_c+c1]-buffer[s*n_c+c2];
        avg_all_diff += max_diff;
    }
    avg_all_diff /= (float) n_s;
    return avg_all_diff;
}

void unfold_buffer(float* buffer, const int n_s, const int n_c, const int n_r){
	for(int s = n_s-1; s >= 0; s--){
		for(int c = n_c-1; c >= 0; c--)
			buffer[n_r*s+c] = buffer[n_c*s+c];
		for(int c = n_c; c < n_r; c++)
			buffer[n_r*s+c] = 0.0f;
	}
}

void refold_buffer(float* buffer, const int n_s, const int n_c, const int n_r){
	for(int s = 0; s < n_s; s++)
	for(int c = 0; c < n_c; c++)
		buffer[n_c*s+c] = buffer[n_r*s+c];
}

inline int idx(const int x, const int n_x, const int y, const int n_y){
    return y + n_y*x;
}

inline int idx(const int x, const int n_x, const int y, const int n_y, const int z, const int n_z){
    return z + n_z*idx(x,n_x,y,n_y);
}

inline int idxc(const int s, const int n_s, const int c, const int n_c){
    return c + n_c*s;
}

inline int idxc(const int x, const int n_x, const int y, const int n_y, const int c, const int n_c){
    return c + n_c*idx(x,n_x,y,n_y);
}

inline int idxc(const int x, const int n_x, const int y, const int n_y, const int z, const int n_z, const int c, const int n_c){
    return c + n_c*idx(x,n_x,y,n_y,z,n_z);
}

void softmax_channels_first(const float* bufferin, float* bufferout, const int n_s, const int n_c){
    for(int s = 0; s < n_s; s++) {
        float max_cost = bufferin[s];
        for(int c = 1; c < n_c; c++)
            if(bufferin[n_s* c + s] > max_cost)
                max_cost = bufferin[n_s* c + s];
        float accum = 0.0f;
        for(int c = 0; c < n_c; c++){
            bufferout[n_s* c + s] = std::exp(bufferin[n_s* c + s]-max_cost);
            accum += bufferout[n_s* c + s];
        }
        for(int c = 0; c < n_c; c++)
            bufferout[n_s* c + s] /= accum;
    }
}

void softmax(const float* bufferin, float* bufferout, const int n_s, const int n_c){
    for(int s = 0; s < n_s; s++) {
        float max_cost = bufferin[n_c*s];
        for(int c = 1; c < n_c; c++)
            if(bufferin[c + n_c*s] > max_cost)
                max_cost = bufferin[c + n_c*s];
        float accum = 0.0f;
        for(int c = 0; c < n_c; c++){
            bufferout[c + n_c*s] = std::exp(bufferin[c + n_c*s]-max_cost);
            accum += bufferout[c + n_c*s];
        }
        for(int c = 0; c < n_c; c++)
            bufferout[c + n_c*s] /= accum;
    }
}

void softmax_update(const float* bufferin, float* bufferout, const int n_s, const int n_c, const float alpha){
    float* new_u = new float[n_c];
    for(int s = 0; s < n_s; s++) {
        float max_cost = bufferin[n_c*s];
        for(int c = 1; c < n_c; c++)
            if(bufferin[c + n_c*s] > max_cost)
                max_cost = bufferin[c + n_c*s];
        float accum = 0.0f;
        for(int c = 0; c < n_c; c++){
            new_u[c] = std::exp(bufferin[c + n_c*s]-max_cost);
            accum += new_u[c];
        }
        for(int c = 0; c < n_c; c++){
            new_u[c] /= accum;
            float diff = alpha * (new_u[c]-bufferout[c + n_c*s]);
            bufferout[c + n_c*s] += diff;
            
        }
    }
    delete new_u;
}
    
float softmax_with_convergence(const float* bufferin, float* bufferout, const int n_s, const int n_c, const float alpha){
    float* new_u = new float[n_c];
    float max_change = 0.0f;
    for(int s = 0; s < n_s; s++) {
        float max_cost = bufferin[n_c*s];
        for(int c = 1; c < n_c; c++)
            if(bufferin[c + n_c*s] > max_cost)
                max_cost = bufferin[c + n_c*s];
        float accum = 0.0f;
        for(int c = 0; c < n_c; c++){
            new_u[c] = std::exp(bufferin[c + n_c*s]-max_cost);
            accum += new_u[c];
        }
        for(int c = 0; c < n_c; c++){
            new_u[c] /= accum;
            float diff = alpha * (new_u[c]-bufferout[c + n_c*s]);
                
            if( diff < -max_change )
                max_change = -diff;
            if( diff > max_change )
                max_change = diff;
            
            bufferout[c + n_c*s] += diff;
            
        }
    }
    delete new_u;
    return max_change;
}

void sigmoid(const float* bufferin, float* bufferout, const int n_s){
    for(int s = 0; s < n_s; s++) {
		float cost = bufferin[s];
        bufferout[s] = 1.0f / (1.0f + std::exp(-cost));
    }
}

void update(float* buffer, const float* update, const int n_s, const float alpha){
	for(int s = 0; s < n_s; s++) {
		float diff = alpha * (update[s]-buffer[s]);
		buffer[s] += diff;
	}
}
    
float update_with_convergence(float* buffer, const float* update, const int n_s, const float alpha){
    float max_change = 0.0f;
	for(int s = 0; s < n_s; s++) {
		float diff = alpha * (update[s]-buffer[s]);
		buffer[s] += diff;
		if( diff > max_change )
			max_change = diff;
		else if( -diff > max_change )
			max_change = -diff;
	}
    return max_change;
}

void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) 
    for (int z = 0; z < n_z; z++)
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] = 0.0f;

        //in z+
        if(z < n_z-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rz[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)]-1.0);

        //in z-
        if(z > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rz[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)] * (2.0*u[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)]-1.0);

        //in y+
        if(y < n_y-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += ry[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)]-1.0);

        //in y-
        if(y > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += ry[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)]-1.0);

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rx[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)]-1.0);

    }
}

void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* u, const int n_x, const int n_y, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) 
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,y,n_y,c,n_c)] = 0.0f;
        
        //in y+
        if(y < n_y-1)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += ry[idxc(x,n_x,y,n_y,c,n_c)] * (2.0*u[idxc(x,n_x,y+1,n_y,c,n_c)]-1.0);

        //in y-
        if(y > 0)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += ry[idxc(x,n_x,y-1,n_y,c,n_c)] * (2.0*u[idxc(x,n_x,y-1,n_y,c,n_c)]-1.0);

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += rx[idxc(x,n_x,y,n_y,c,n_c)] * (2.0*u[idxc(x+1,n_x,y,n_y,c,n_c)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += rx[idxc(x-1,n_x,y,n_y,c,n_c)] * (2.0*u[idxc(x-1,n_x,y,n_y,c,n_c)]-1.0);

    }
}

void calculate_r_eff(float* r_eff, const float* rx, const float* u, const int n_x, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,c,n_c)] = 0.0f;

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,c,n_c)] += rx[idxc(x,n_x,c,n_c)] * (2.0*u[idxc(x+1,n_x,c,n_c)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,c,n_c)] += rx[idxc(x-1,n_x,c,n_c)] * (2.0*u[idxc(x-1,n_x,c,n_c)]-1.0);

    }
}


void aggregate_bottom_up(float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list){
    for (int s = 0; s < n_s; s++)
        for (int l = 0; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            if(n->d == -1){
                for(int c = 0; c < n->c; c++)
                    buffer[idxc(s,n_s,n->r,n_r)] += buffer[idxc(s,n_s,n->children[c]->r,n_r)];
            }else{
                buffer[idxc(s,n_s,n->r,n_r)] = buffer[idxc(s,n_s,n->d,n_r)];
            }
        }
}

void aggregate_bottom_up(const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list){
    for (int s = 0; s < n_s; s++)
        for (int l = 0; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            if(n->c > 0){
                bufferout[idxc(s,n_s,n->r,n_r)] = 0.0f;
                for(int c = 0; c < n->c; c++)
                    bufferout[idxc(s,n_s,n->r,n_r)] += bufferout[idxc(s,n_s,n->children[c]->r,n_r)];
            }else{
                bufferout[idxc(s,n_s,n->r,n_r)] = bufferin[idxc(s,n_s,n->d,n_c)];
            }
        }
}

void aggregate_top_down(float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list){
	for(int s = 0; s < n_s; s++)
        for (int l = n_r-1; l >= 0; l--) {
            const TreeNode* n = bottom_up_list[l];
			for(int c = 0; c < n->c; c++)
				buffer[idxc(s,n_s,n->children[c]->r,n_r)] += buffer[idxc(s,n_s,n->r,n_r)];
		}
}

void untangle_softmax(const float* g, const float* u, float* dy, const int n_s, const int n_c){
	for(int s = 0; s < n_s; s++)
		for (int c = 0; c < n_c; c++){
			float new_grad = 0.0f;
			float uc = u[idxc(s,n_s,c,n_c)];
			for(int a = 0; a < n_c; a++){
				float da = g[idxc(s,n_s,a,n_c)];
				if(c == a)
					new_grad += da*(1.0f-uc);
				else
					new_grad -= da*u[idxc(s,n_s,a,n_c)];
			}
			dy[idxc(s,n_s,c,n_c)] = new_grad*uc;
		}
}

void untangle_sigmoid(const float* g, const float* u, float* dy, const int n_s){
	for(int s = 0; s < n_s; s++)
		dy[s] = g[s]*u[s]*(1-u[s]);
}

void get_gradient_for_u(const float* dy, const float* rx, const float* ry, const float* rz, float* du, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++)
	for(int c = 0; c < n_c; c++){
		float grad_val = 0.0f;

		//z down
		if( z > 0 )
			grad_val += 2.0f*dy[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)]*rz[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)];

		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)]*rz[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)]*ry[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)];

		//z up
		if( z < n_z - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)]*ry[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)]*rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)]*rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		du[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
	}
}

void get_gradient_for_u(const float* dy, const float* rx, const float* ry, float* du, const int n_x, const int n_y, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int c = 0; c < n_c; c++){
		float grad_val = 0.0f;
		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idxc(x,n_x,y-1,n_y,c,n_c)]*ry[idxc(x,n_x,y-1,n_y,c,n_c)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idxc(x-1,n_x,y,n_y,c,n_c)]*ry[idxc(x-1,n_x,y,n_y,c,n_c)];
		
		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y+1,n_y,c,n_c)]*rx[idxc(x,n_x,y,n_y,c,n_c)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idxc(x+1,n_x,y,n_y,c,n_c)]*rx[idxc(x,n_x,y,n_y,c,n_c)];

		du[idxc(x,n_x,y,n_y,c,n_c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,n_x,y,n_y,c,n_c)];
	}
}

void get_gradient_for_u(const float* dy, const float* rx, float* du, const int n_x, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int c = 0; c < n_c; c++){
		float grad_val = 0.0f;
		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idxc(x-1,n_x,c,n_c)]*rx[idxc(x-1,n_x,c,n_c)];
		
		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idxc(x+1,n_x,c,n_c)]*rx[idxc(x,n_x,c,n_c)];

		du[idxc(x,n_x,c,n_c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,n_x,c,n_c)];
	}
}

void get_reg_gradients(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++)
	for(int c = 0; c < n_c; c++){
		
		//for z
		if( z < n_z - 1 ){
			float up_contra = (2.0f*u[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)];
			g_rz[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += tau * (up_contra + dn_contra);
		}

		//for y
		if( y < n_y - 1 ){
			float up_contra = (2.0f*u[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)];
			g_ry[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += tau * (up_contra + dn_contra);
		}

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)];
			g_rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += tau * (up_contra + dn_contra);
		}
	}

}

void get_reg_gradients(const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int c = 0; c < n_c; c++){

		//for y
		if( y < n_y - 1 ){
			float up_contra = (2.0f*u[idxc(x,n_x,y+1,n_y,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,c,n_c)]-1.0f) * g[idxc(x,n_x,y+1,n_y,c,n_c)];
			g_ry[idxc(x,n_x,y,n_y,c,n_c)] += tau * (up_contra + dn_contra);
		}

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idxc(x+1,n_x,y,n_y,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,c,n_c)]-1.0f) * g[idxc(x+1,n_x,y,n_y,c,n_c)];
			g_rx[idxc(x,n_x,y,n_y,c,n_c)] += tau * (up_contra + dn_contra);
		}
	}

}

void get_reg_gradients(const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int c = 0; c < n_c; c++){

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idxc(x+1,n_x,c,n_c)]-1.0f) * g[idxc(x,n_x,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,c,n_c)]-1.0f) * g[idxc(x+1,n_x,c,n_c)];
			g_rx[idxc(x,n_x,c,n_c)] += tau * (up_contra + dn_contra);
		}
	}

}
	

void compute_source_flow( const float* u, float* ps, const float* pt, const float* div, const float icc, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++){
        ps[s] = icc;
        for(int c = 0; c < n_c; c++)
            ps[s] += pt[idxc(s,n_s,c,n_c)] + div[idxc(s,n_s,c,n_c)] - u[idxc(s,n_s,c,n_c)] * icc;
        ps[s] /= n_c;
    }
}

void compute_sink_flow( const float* u, const float* ps, float* pt, const float* div, const float* d, const float icc, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++)
        for (int c = 0; c < n_c; c++) {
            int i = idxc(s,n_s,c,n_c);
            pt[i] = ps[s] - div[i] + u[i] * icc;
            if( pt[i] > -d[i] )
                pt[i] = -d[i];
        }
}
    
void compute_multipliers( float* erru, float* u, const float* ps, const float* pt, const float* div, const float cc, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++)
        for (int c = 0; c < n_c; c++) {
            int i = idxc(s,n_s,c,n_c);
            erru[i] = cc * (ps[s] - div[i] - pt[i]);
            u[i] += erru[i];
        }
}

void compute_source_sink_multipliers( float* erru, float* u, float* ps, float* pt, const float* div, const float* d, const float cc, const float icc, const int n_c, const int n_s){

    for(int s = 0; s < n_s; s++){
        ps[s] = icc;
        for(int c = 0; c < n_c; c++)
            ps[s] += pt[idxc(s,n_s,c,n_c)] + div[idxc(s,n_s,c,n_c)] - u[idxc(s,n_s,c,n_c)] * icc;
        ps[s] /= n_c;
        
        for (int c = 0; c < n_c; c++) {
            int i = idxc(s,n_s,c,n_c);
            pt[i] = ps[s] - div[i] + u[i] * icc;
            if( pt[i] > -d[i] )
                pt[i] = -d[i];
            
            erru[i] = cc * (ps[s] - div[i] - pt[i]);
            u[i] += erru[i];
            if(erru[i] < 0.0f)
                erru[i] = -erru[i];
        }
        
    }
    
}

void compute_source_sink_multipliers_binary( float* erru, float* u, float* ps, float* pt, const float* div, const float* d, const float cc, const float icc, const int n_s){
    for(int s = 0; s < n_s; s++){
        ps[s] = icc + pt[s] + div[s] - u[s] * icc;
		if( d[s] < 0.0f )
			ps[s] = 0.0f;
		else if( ps[s] > d[s])
			ps[s] = d[s];
		
        pt[s] = ps[s] - div[s] + u[s] * icc;
		if( d[s] > 0.0f )
			pt[s] = 0.0f;
		else if( pt[s] > -d[s])
			pt[s] = -d[s];
            
		erru[s] = cc * (ps[s] - div[s] - pt[s]);
		u[s] += erru[s];
		if(erru[s] < 0.0f)
			erru[s] = -erru[s];
    }
}

void compute_capacity_potts(float* g, const float* u, const float* ps, const float* pt, const float* div, const int n_s, const int n_c, const float tau, const float icc){
    for(int s = 0, cs = 0; s < n_s; s++)
        for(int c = 0; c < n_c; c++, cs++)
            g[cs] = tau * (div[cs] + pt[cs] - ps[s] - u[cs] * icc);
}

void compute_capacity_binary(float* g, const float* u, const float* ps, const float* pt, const float* div, const int n_s, const float tau, const float icc){
    for(int s = 0; s < n_s; s++)
		g[s] = tau * (div[s] + pt[s] - ps[s] - u[s] * icc);
}

void compute_flows(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const int n_c, const int n_x, const int n_y, const int n_z){
    const int n_s = n_x*n_y*n_z;
    
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            if (x != 0) {
                int sxm = idxc(x-1,n_x,y,n_y,z,n_z,c,n_c);
                px[cs] += g[cs] - g[sxm];
                if (px[cs] > rx[cs])
                    px[cs] = rx[cs];
                if (px[cs] < -rx[cs])
                    px[cs] = -rx[cs];
            }
            if (y != 0){
                int sym = idxc(x,n_x,y-1,n_y,z,n_z,c,n_c);
                py[cs] += g[cs] - g[sym];
                if (py[cs] > ry[cs])
                    py[cs] = ry[cs];
                if (py[cs] < -ry[cs])
                    py[cs] = -ry[cs];
            }
            if (z != 0){
                int szm = idxc(x,n_x,y,n_y,z-1,n_z,c,n_c);
                pz[cs] += g[cs] - g[szm];
                if (pz[cs] > rz[cs])
                    pz[cs] = rz[cs];
                if (pz[cs] < -rz[cs])
                    pz[cs] = -rz[cs];
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            div[cs] = -px[cs]-py[cs]-pz[cs];
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,z,n_z,c,n_c);
                div[cs] += px[sxm];
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,z,n_z,c,n_c);
                div[cs] += py[sym];
            }
            if (z < n_z-1){
                int szm = idxc(x,n_x,y,n_y,z+1,n_z,c,n_c);
                div[cs] += pz[szm];
            }
        }
    }
            
}


void compute_flows(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_c, const int n_x, const int n_y){
    const int n_s = n_x*n_y;
    
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            if (x != 0) {
                int sxm = idxc(x-1,n_x,y,n_y,c,n_c);
                px[cs] += g[cs] - g[sxm];
                if (px[cs] > rx[cs])
                    px[cs] = rx[cs];
                if (px[cs] < -rx[cs])
                    px[cs] = -rx[cs];
            }
            if (y != 0){
                int sym = idxc(x,n_x,y-1,n_y,c,n_c);
                py[cs] += g[cs] - g[sym];
                if (py[cs] > ry[cs])
                    py[cs] = ry[cs];
                if (py[cs] < -ry[cs])
                    py[cs] = -ry[cs];
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            div[cs] = -px[cs]-py[cs];
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,c,n_c);
                div[cs] += px[sxm];
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,c,n_c);
                div[cs] += py[sym];
            }
        }
    }
            
}

void compute_flows(const float* g, float* div, float* px, const float* rx, const int n_c, const int n_x){
    const int n_s = n_x;
    
    for(int x = 0, s = 0; x < n_x; x++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            if (x != 0) {
                int sxm = idxc(x-1,n_x,c,n_c);
                px[cs] += g[cs] - g[sxm];
                if (px[cs] > rx[cs])
                    px[cs] = rx[cs];
                if (px[cs] < -rx[cs])
                    px[cs] = -rx[cs];
            }
        }
    }
    
    for(int x = 0; x < n_x; x++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            div[cs] = -px[cs];
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,c,n_c);
                div[cs] += px[sxm];
            }
        }
    }
            
}

void compute_flows_channels_first(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const int n_c, const int n_x, const int n_y, const int n_z){
    const int n_s = n_x*n_y*n_z;
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, cs++){
        if (x != 0) {
            int sxm = cs - n_y*n_z;
            px[cs] += g[cs] - g[sxm];
            if (px[cs] > rx[cs])
                px[cs] = rx[cs];
            if (px[cs] < -rx[cs])
                px[cs] = -rx[cs];
        }
        if (y != 0){
            int sym = cs - n_z;
            py[cs] += g[cs] - g[sym];
            if (py[cs] > ry[cs])
                py[cs] = ry[cs];
            if (py[cs] < -ry[cs])
                py[cs] = -ry[cs];
        }
        if (z != 0){
            int szm = cs - 1;
            pz[cs] += g[cs] - g[szm];
            if (pz[cs] > rz[cs])
                pz[cs] = rz[cs];
            if (pz[cs] < -rz[cs])
                pz[cs] = -rz[cs];
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++,cs++){
        div[cs] = -px[cs]-py[cs]-pz[cs];
        if (x < n_x-1) {
            int sxm = cs + n_y*n_z;
            div[cs] += px[sxm];
        }
        if (y < n_y-1){
            int sym = cs + n_z;;
            div[cs] += py[sym];
        }
        if (z < n_z-1){
            int szm = cs + 1;
            div[cs] += pz[szm];
        }
    }
            
}


void compute_flows_channels_first(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_c, const int n_x, const int n_y){
    const int n_s = n_x*n_y;
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        if (x != 0) {
            int sxm = cs - n_y;
            px[cs] += g[cs] - g[sxm];
            if (px[cs] > rx[cs])
                px[cs] = rx[cs];
            if (px[cs] < -rx[cs])
                px[cs] = -rx[cs];
        }
        if (y != 0){
            int sym = cs - 1;
            py[cs] += g[cs] - g[sym];
            if (py[cs] > ry[cs])
                py[cs] = ry[cs];
            if (py[cs] < -ry[cs])
                py[cs] = -ry[cs];
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        div[cs] = -px[cs]-py[cs];
        if (x < n_x-1) {
            int sxm = cs + n_y;
            div[cs] += px[sxm];
        }
        if (y < n_y-1){
            int sym = cs + 1;
            div[cs] += py[sym];
        }
    }
            
}

void compute_flows_channels_first(const float* g, float* div, float* px, const float* rx, const int n_c, const int n_x){
    const int n_s = n_x;
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++,cs++){
        if (x != 0) {
            int sxm = cs - 1;
            px[cs] += g[cs] - g[sxm];
            if (px[cs] > rx[cs])
                px[cs] = rx[cs];
            if (px[cs] < -rx[cs])
                px[cs] = -rx[cs];
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++,cs++){
        div[cs] = -px[cs];
        if (x < n_x-1) {
            int sxm = cs + 1;
            div[cs] += px[sxm];
        }
    }
            
}

void init_flows(const float* d, float* ps, float* pt, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++){
        float max_d = -std::numeric_limits<float>::infinity();
        for(int c = 0; c < n_c; c++){
            int cs = idxc(s,n_s,c,n_c);
            if( max_d < d[cs] )
                max_d = d[cs];
        }
        //cs -= n_c;
        ps[s] = -max_d;
        for(int c = 0; c < n_c; c++){
            int cs = idxc(s,n_s,c,n_c);
            pt[cs] = -max_d;
        }
    }
            
}
void init_flows(const float* d, float* ps, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++){
        float max_d = -std::numeric_limits<float>::infinity();
        for(int c = 0; c < n_c; c++){
            int cs = idxc(s,n_s,c,n_c);
            if( max_d < d[cs] )
                max_d = d[cs];
        }
        //cs -= n_c;
        ps[s] = -max_d;
    }
            
}
void init_flows_channels_first(const float* d, float* ps, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++)
        ps[s] = std::numeric_limits<float>::infinity();
    for(int c = 0, cs = 0; c < n_c; c++)
        for(int s = 0; s < n_s; s++, cs++)
            if( ps[s] > d[cs] )
                ps[s] = d[cs];
}