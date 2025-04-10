#include "cpu_kernels_meanpass.h"
#include "cpu_kernels.h"
#include <limits>
#include <iostream>
#define epsilon 0.00001f

void change_to_diff(const CPU_DEVICE & dev, float* buffer, float* update, const int n_s, const float alpha){
	for(int s = 0; s < n_s; s++) {
		float diff = alpha * (update[s]-buffer[s]);
		buffer[s] += diff;
        update[s] = diff;
	}
}

void calculate_r_eff_channels_last(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c) {
    
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

void calculate_r_eff_channels_last(float* r_eff, const float* rx, const float* ry, const float* u, const int n_x, const int n_y, const int n_c) {
    
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

void calculate_r_eff_channels_last(float* r_eff, const float* rx, const float* u, const int n_x, const int n_c) {
    
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

void calculate_r_eff_channels_first(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c) {
    
    for (int c = 0; c < n_c; c++)
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) 
    for (int z = 0; z < n_z; z++) {
        r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] = 0.0f;

        //in z+
        if(z < n_z-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += rz[idx(c,n_c,x,n_x,y,n_y,z  ,n_z)] * (2.0*u[idx(c,n_c,x,n_x,y,n_y,z+1,n_z)]-1.0);

        //in z-
        if(z > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += rz[idx(c,n_c,x,n_x,y,n_y,z-1,n_z)] * (2.0*u[idx(c,n_c,x,n_x,y,n_y,z-1,n_z)]-1.0);

        //in y+
        if(y < n_y-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += ry[idx(c,n_c,x,n_x,y  ,n_y,z,n_z)] * (2.0*u[idx(c,n_c,x,n_x,y+1,n_y,z,n_z)]-1.0);

        //in y-
        if(y > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += ry[idx(c,n_c,x,n_x,y-1,n_y,z,n_z)] * (2.0*u[idx(c,n_c,x,n_x,y-1,n_y,z,n_z)]-1.0);

        //in x+
        if(x < n_x-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += rx[idx(c,n_c,x  ,n_x,y,n_y,z,n_z)] * (2.0*u[idx(c,n_c,x+1,n_x,y,n_y,z,n_z)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += rx[idx(c,n_c,x-1,n_x,y,n_y,z,n_z)] * (2.0*u[idx(c,n_c,x-1,n_x,y,n_y,z,n_z)]-1.0);

    }
}

void calculate_r_eff_channels_first(float* r_eff, const float* rx, const float* ry, const float* u, const int n_x, const int n_y, const int n_c) {
    
    for (int c = 0; c < n_c; c++) 
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) {
        r_eff[idx(c,n_c,x,n_x,y,n_y)] = 0.0f;
        
        //in y+
        if(y < n_y-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y)] += ry[idx(c,n_c,x,n_x,y  ,n_y)] * (2.0*u[idx(c,n_c,x,n_x,y+1,n_y)]-1.0);

        //in y-
        if(y > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y)] += ry[idx(c,n_c,x,n_x,y-1,n_y)] * (2.0*u[idx(c,n_c,x,n_x,y-1,n_y)]-1.0);

        //in x+
        if(x < n_x-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y)] += rx[idx(c,n_c,x  ,n_x,y,n_y)] * (2.0*u[idx(c,n_c,x+1,n_x,y,n_y)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y)] += rx[idx(c,n_c,x-1,n_x,y,n_y)] * (2.0*u[idx(c,n_c,x-1,n_x,y,n_y)]-1.0);

    }
}

void calculate_r_eff_channels_first(float* r_eff, const float* rx, const float* u, const int n_x, const int n_c) {
    
    for (int c = 0; c < n_c; c++) 
    for (int x = 0; x < n_x; x++) {
        r_eff[idx(c,n_c,x,n_x)] = 0.0f;

        //in x+
        if(x < n_x-1)
            r_eff[idx(c,n_c,x,n_x)] += rx[idx(c,n_c,x,n_x)] * (2.0*u[idx(c,n_c,x+1,n_x)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idx(c,n_c,x,n_x)] += rx[idx(c,n_c,x-1,n_x)] * (2.0*u[idx(c,n_c,x-1,n_x)]-1.0);

    }
}

void get_effective_reg(const CPU_DEVICE & dev, float* const r_eff, const float* const u, const float *const *const r, const int dim, const int* const n, const int n_c){
    if(dev.channels_first)
        switch(dim){
            case 1:
                calculate_r_eff_channels_first(r_eff, r[0], u, n[0], n_c);
                break;
            case 2:
                calculate_r_eff_channels_first(r_eff, r[0], r[1], u, n[0], n[1], n_c);
                break;
            case 3:
                calculate_r_eff_channels_first(r_eff, r[0], r[1], r[2], u, n[0], n[1], n[2], n_c);
                break;
        }
    else
        switch(dim){
            case 1:
                calculate_r_eff_channels_last(r_eff, r[0], u, n[0], n_c);
                break;
            case 2:
                calculate_r_eff_channels_last(r_eff, r[0], r[1], u, n[0], n[1], n_c);
                break;
            case 3:
                calculate_r_eff_channels_last(r_eff, r[0], r[1], r[2], u, n[0], n[1], n[2], n_c);
                break;
        }
}


void untangle_softmax_channels_first(const float* g, const float* u, float* dy, const int n_s, const int n_c){
	for(int s = 0; s < n_s; s++)
		for (int c = 0; c < n_c; c++){
			float new_grad = 0.0f;
			float uc = u[idx(c,n_c,s,n_s)];
			for(int a = 0; a < n_c; a++){
				float da = g[idx(a,n_c,s,n_s)];
				if(c == a)
					new_grad += da*(1.0f-uc);
				else
					new_grad -= da*u[idx(a,n_c,s,n_s)];
			}
			dy[idx(c,n_c,s,n_s)] = new_grad*uc;
		}
}

void untangle_softmax_channels_last(const float* g, const float* u, float* dy, const int n_s, const int n_c){
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

void untangle_softmax(const CPU_DEVICE & dev, const float* g, const float* u, float* dy, const int n_s, const int n_c){
    if(dev.channels_first)
        untangle_softmax_channels_first(g,u,dy,n_s,n_c);
    else
        untangle_softmax_channels_last(g,u,dy,n_s,n_c);
}

void untangle_sigmoid(const CPU_DEVICE & dev, const float* g, const float* u, float* dy, const int n_s){
	for(int s = 0; s < n_s; s++)
		dy[s] = g[s]*u[s]*(1-u[s]);
}

void get_gradient_for_u_channels_last(const float* dy, const float* rx, const float* ry, const float* rz, float* du, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
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
			grad_val += 2.0f*dy[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)]*ry[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)]*rx[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)];

		//z up
		if( z < n_z - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)]*rz[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)]*ry[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)]*rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		du[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
	}
}

void get_gradient_for_u_channels_last(const float* dy, const float* rx, const float* ry, float* du, const int n_x, const int n_y, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int c = 0; c < n_c; c++){
		float grad_val = 0.0f;
		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idxc(x,n_x,y-1,n_y,c,n_c)]*ry[idxc(x,n_x,y-1,n_y,c,n_c)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idxc(x-1,n_x,y,n_y,c,n_c)]*rx[idxc(x-1,n_x,y,n_y,c,n_c)];
		
		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y+1,n_y,c,n_c)]*ry[idxc(x,n_x,y,n_y,c,n_c)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idxc(x+1,n_x,y,n_y,c,n_c)]*rx[idxc(x,n_x,y,n_y,c,n_c)];

		du[idxc(x,n_x,y,n_y,c,n_c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,n_x,y,n_y,c,n_c)];
	}
}

void get_gradient_for_u_channels_last(const float* dy, const float* rx, float* du, const int n_x, const int n_c, const float tau){
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

void get_gradient_for_u_channels_first(const float* dy, const float* rx, const float* ry, const float* rz, float* du, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++){
		float grad_val = 0.0f;

		//z down
		if( z > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y,n_y,z-1,n_z)]*rz[idx(c,n_c,x,n_x,y,n_y,z-1,n_z)];

		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y-1,n_y,z,n_z)]*ry[idx(c,n_c,x,n_x,y-1,n_y,z,n_z)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x-1,n_x,y,n_y,z,n_z)]*rx[idx(c,n_c,x-1,n_x,y,n_y,z,n_z)];

		//z up
		if( z < n_z - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y,n_y,z+1,n_z)]*rz[idx(c,n_c,x,n_x,y,n_y,z,n_z)];

		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y+1,n_y,z,n_z)]*ry[idx(c,n_c,x,n_x,y,n_y,z,n_z)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x+1,n_x,y,n_y,z,n_z)]*rx[idx(c,n_c,x,n_x,y,n_y,z,n_z)];

		du[idx(c,n_c,x,n_x,y,n_y,z,n_z)] = tau*grad_val + (1.0f-tau)*du[idx(c,n_c,x,n_x,y,n_y,z,n_z)];
	}
}

void get_gradient_for_u_channels_first(const float* dy, const float* rx, const float* ry, float* du, const int n_x, const int n_y, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++){
		float grad_val = 0.0f;
		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y-1,n_y)]*ry[idx(c,n_c,x,n_x,y-1,n_y)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x-1,n_x,y,n_y)]*rx[idx(c,n_c,x-1,n_x,y,n_y)];
		
		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y+1,n_y)]*ry[idx(c,n_c,x,n_x,y,n_y)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x+1,n_x,y,n_y)]*rx[idx(c,n_c,x,n_x,y,n_y)];

		du[idx(c,n_c,x,n_x,y,n_y)] = tau*grad_val + (1.0f-tau)*du[idx(c,n_c,x,n_x,y,n_y)];
	}
}

void get_gradient_for_u_channels_first(const float* dy, const float* rx, float* du, const int n_x, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++){
		float grad_val = 0.0f;
		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x-1,n_x)]*rx[idx(c,n_c,x-1,n_x)];
		
		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x+1,n_x)]*rx[idx(c,n_c,x,n_x)];

		du[idx(c,n_c,x,n_x)] = tau*grad_val + (1.0f-tau)*du[idx(c,n_c,x,n_x)];
	}
}


void get_gradient_for_u(const CPU_DEVICE & dev, const float* dy, const float *const *const r, float* const du, const int dim, const int* const n, const int n_c, const float tau){
    if(dev.channels_first)
        switch(dim){
            case 1:
                get_gradient_for_u_channels_first(dy, r[0], du, n[0], n_c, tau);
                break;
            case 2:
                get_gradient_for_u_channels_first(dy, r[0], r[1], du, n[0], n[1], n_c, tau);
                break;
            case 3:
                get_gradient_for_u_channels_first(dy, r[0], r[1], r[2], du, n[0], n[1], n[2], n_c, tau);
                break;
        }
    else
        switch(dim){
            case 1:
                get_gradient_for_u_channels_last(dy, r[0], du, n[0], n_c, tau);
                break;
            case 2:
                get_gradient_for_u_channels_last(dy, r[0], r[1], du, n[0], n[1], n_c, tau);
                break;
            case 3:
                get_gradient_for_u_channels_last(dy, r[0], r[1], r[2], du, n[0], n[1], n[2], n_c, tau);
                break;
        }
        
}

void get_reg_gradients_channels_last(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
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

void get_reg_gradients_channels_last(const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau){
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

void get_reg_gradients_channels_last(const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau){
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

void get_reg_gradients_channels_first(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++){
		
		//for z
		if( z < n_z - 1 ){
			float up_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y,z+1,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y,z,n_z)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y,z+1,n_z)];
			g_rz[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += tau * (up_contra + dn_contra);
		}

		//for y
		if( y < n_y - 1 ){
			float up_contra = (2.0f*u[idx(c,n_c,x,n_x,y+1,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y,z,n_z)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y+1,n_y,z,n_z)];
			g_ry[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += tau * (up_contra + dn_contra);
		}

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idx(c,n_c,x+1,n_x,y,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y,z,n_z)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x+1,n_x,y,n_y,z,n_z)];
			g_rx[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += tau * (up_contra + dn_contra);
		}
	}
}

void get_reg_gradients_channels_first(const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++){

		//for y
		if( y < n_y - 1 ){
			float up_contra = (2.0f*u[idx(c,n_c,x,n_x,y+1,n_y)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y)]-1.0f) * g[idx(c,n_c,x,n_x,y+1,n_y)];
			g_ry[idx(c,n_c,x,n_x,y,n_y)] += tau * (up_contra + dn_contra);
		}

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idx(c,n_c,x+1,n_x,y,n_y)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y)]-1.0f) * g[idx(c,n_c,x+1,n_x,y,n_y)];
			g_rx[idx(c,n_c,x,n_x,y,n_y)] += tau * (up_contra + dn_contra);
		}
	}
}

void get_reg_gradients_channels_first(const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++){

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idx(c,n_c,x+1,n_x)]-1.0f) * g[idx(c,n_c,x,n_x)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x)]-1.0f) * g[idx(c,n_c,x+1,n_x)];
			g_rx[idx(c,n_c,x,n_x)] += tau * (up_contra + dn_contra);
		}
	}
}

void populate_reg_mean_gradients_and_add(const CPU_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau){
    if(dev.channels_first)
        switch(dim){
            case 1:
                get_reg_gradients_channels_first(g, u, g_r[0], n[0], n_c, tau);
                break;
            case 2:
                get_reg_gradients_channels_first(g, u, g_r[0], g_r[1], n[0], n[1], n_c, tau);
                break;
            case 3:
                get_reg_gradients_channels_first(g, u, g_r[0], g_r[1], g_r[2], n[0], n[1], n[2], n_c, tau);
                break;
        }
    else
        switch(dim){
            case 1:
                get_reg_gradients_channels_last(g, u, g_r[0], n[0], n_c, tau);
                break;
            case 2:
                get_reg_gradients_channels_last(g, u, g_r[0], g_r[1], n[0], n[1], n_c, tau);
                break;
            case 3:
                get_reg_gradients_channels_last(g, u, g_r[0], g_r[1], g_r[2], n[0], n[1], n[2], n_c, tau);
                break;
        }
}
	