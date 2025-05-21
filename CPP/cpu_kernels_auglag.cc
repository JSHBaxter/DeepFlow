#include "cpu_kernels_auglag.h"
#include "cpu_kernels.h"
#include <limits>
#include <iostream>
#define epsilon 0.00001f


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

void update_source_sink_multiplier_potts_channels_last( float* erru, float* u, float* ps, float* pt, const float* div, const float* d, const float cc, const float icc, const int n_c, const int n_s){

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

void update_source_sink_multiplier_potts_channels_first( float* erru, float* u, float* ps, float* pt, const float* div, const float* d, const float cc, const float icc, const int n_c, const int n_s){

    for(int s = 0; s < n_s; s++){
        ps[s] = icc;
        for(int c = 0; c < n_c; c++){
            ps[s] += pt[idx(c,n_c,s,n_s)];
            ps[s] += div[idx(c,n_c,s,n_s)];
            ps[s] -= u[idx(c,n_c,s,n_s)] * icc;
        }
        ps[s] /= n_c;
        
        for (int c = 0; c < n_c; c++) {
            int i = idx(c,n_c,s,n_s);
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

void update_source_sink_multiplier_potts(const CPU_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s){
    if(dev.channels_first)
        update_source_sink_multiplier_potts_channels_first(erru, u, ps, pt, div, d, cc, icc, n_c, n_s);
    else
        update_source_sink_multiplier_potts_channels_last(erru, u, ps, pt, div, d, cc, icc, n_c, n_s);
}

void update_source_sink_multiplier_binary(const CPU_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s){
    for(int s = 0; s < n_s; s++){
        float d_tmp = d[s];
        float pt_tmp = pt[s];
        float div_tmp = div[s];
        float u_tmp = u[s];
        
        float ps_tmp = icc + pt_tmp + div_tmp - u_tmp * icc;
		if( d_tmp < 0.0f )
			ps_tmp = 0.0f;
		else if( ps_tmp > d_tmp)
			ps_tmp = d_tmp;
        ps[s] = ps_tmp;
		
        pt_tmp = ps_tmp - div_tmp + u_tmp * icc;
		if( d_tmp > 0.0f )
			pt_tmp = 0.0f;
		else if( pt_tmp > -d_tmp)
			pt_tmp = -d_tmp;
        pt[s] = pt_tmp;
        
		float erru_tmp = cc * (ps_tmp - div_tmp - pt_tmp);
		u[s] += erru_tmp;
        erru[s] = (erru_tmp > 0.0f) ? erru_tmp : -erru_tmp;
    }
}

void calc_capacity_potts_channels_last(float* g, const float* u, const float* ps, const float* pt, const float* div, const int n_s, const int n_c, const float tau, const float icc){
    for(int s = 0, cs = 0; s < n_s; s++)
    for(int c = 0; c < n_c; c++, cs++)
        g[cs] = tau * (div[cs] + pt[cs] - ps[s] - u[cs] * icc);
}

void calc_capacity_potts_channels_first(float* g, const float* u, const float* ps, const float* pt, const float* div, const int n_s, const int n_c, const float tau, const float icc){
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int s = 0; s < n_s; s++, cs++)
        g[cs] = tau*(div[cs] + pt[cs] - ps[s] - u[cs] * icc);
}

void calc_capacity_potts(const CPU_DEVICE & dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    if(dev.channels_first)
        calc_capacity_potts_channels_first(g,u,ps,pt,div,n_s,n_c,tau,icc);
    else
        calc_capacity_potts_channels_last(g,u,ps,pt,div,n_s,n_c,tau,icc);
}

void calc_capacity_binary(const CPU_DEVICE & dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau){
    for(int s = 0; s < n_s; s++)
		g[s] = tau * (div[s] + pt[s] - ps[s] - u[s] * icc);
}

void update_spatial_flows__update_specific_flow(const int cs, const int csn, const float* g, float* p, const float* r){
    p[cs] += g[cs] - g[csn];
    if (p[cs] > r[cs])
        p[cs] = r[cs];
    else if (p[cs] < -r[cs])
        p[cs] = -r[cs];
}

void update_spatial_flows__update_specific_star_flow(const int cs, const int csn, const float* g, float* p, const float* r, const float* l){
    p[cs] += g[cs] - g[csn];
    if( p[cs]*l[cs] <= 0.0f){
        if (p[cs] > r[cs])
            p[cs] = r[cs];
        else if (p[cs] < -r[cs])
            p[cs] = -r[cs];
    }
}

void update_spatial_flows_channels_last(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const int n_c, const int n_x, const int n_y, const int n_z){
	
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,z,n_z,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,z,n_z,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sym,g,py,ry);
            }
            if (z < n_z-1){
                int szm = idxc(x,n_x,y,n_y,z+1,n_z,c,n_c);
                update_spatial_flows__update_specific_flow(cs,szm,g,pz,rz);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            div[cs] = -px[cs]-py[cs]-pz[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,y,n_y,z,n_z,c,n_c);
                div[cs] += px[sxm];
            }
            if (y > 0){
                int sym = idxc(x,n_x,y-1,n_y,z,n_z,c,n_c);
                div[cs] += py[sym];
            }
            if (z > 0){
                int szm = idxc(x,n_x,y,n_y,z-1,n_z,c,n_c);
                div[cs] += pz[szm];
            }
        }
    }
            
}

void update_spatial_star_flows_channels_last(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const float* lx, const float* ly, const float * lz, const int n_c, const int n_x, const int n_y, const int n_z){
	
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,z,n_z,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,z,n_z,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sym,g,py,ry,ly);
            }
            if (z < n_z-1){
                int szm = idxc(x,n_x,y,n_y,z+1,n_z,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,szm,g,pz,rz,lz);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            div[cs] = -px[cs]-py[cs]-pz[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,y,n_y,z,n_z,c,n_c);
                div[cs] += px[sxm];
            }
            if (y > 0){
                int sym = idxc(x,n_x,y-1,n_y,z,n_z,c,n_c);
                div[cs] += py[sym];
            }
            if (z > 0){
                int szm = idxc(x,n_x,y,n_y,z-1,n_z,c,n_c);
                div[cs] += pz[szm];
            }
        }
    }
            
}

void update_spatial_flows_channels_last(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_c, const int n_x, const int n_y){
	
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sym,g,py,ry);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            div[cs] = -px[cs]-py[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,y,n_y,c,n_c);
                div[cs] += px[sxm];
            }
            if (y > 0){
                int sym = idxc(x,n_x,y-1,n_y,c,n_c);
                div[cs] += py[sym];
            }
        }
    }
            
}

void update_spatial_star_flows_channels_last(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const float* lx, const float* ly, const int n_c, const int n_x, const int n_y){
	
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sym,g,py,ry,ly);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            div[cs] = -px[cs]-py[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,y,n_y,c,n_c);
                div[cs] += px[sxm];
            }
            if (y > 0){
                int sym = idxc(x,n_x,y-1,n_y,c,n_c);
                div[cs] += py[sym];
            }
        }
    }
            
}

void update_spatial_flows_channels_last(const float* g, float* div, float* px, const float* rx, const int n_c, const int n_x){
    
    for(int x = 0, s = 0; x < n_x; x++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            if (x != n_x-1) {
                int sxm = idxc(x+1,n_x,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            div[cs] = -px[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,c,n_c);
                div[cs] += px[sxm];
            }
        }
    }
            
}
void update_spatial_star_flows_channels_last(const float* g, float* div, float* px, const float* rx, const float* lx, const int n_c, const int n_x){
    
    for(int x = 0, s = 0; x < n_x; x++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            if (x != n_x-1) {
                int sxm = idxc(x+1,n_x,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            div[cs] = -px[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,c,n_c);
                div[cs] += px[sxm];
            }
        }
    }
            
}

void update_spatial_flows_channels_first(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const int n_c, const int n_x, const int n_y, const int n_z){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, cs++){
        if (x < n_x-1) {
            int sxm = cs + n_y*n_z;
            update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
        }
        if (y < n_y-1){
            int sym = cs + n_z;
            update_spatial_flows__update_specific_flow(cs,sym,g,py,ry);
        }
        if (z < n_z-1){
            int szm = cs + 1;
            update_spatial_flows__update_specific_flow(cs,szm,g,pz,rz);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++,cs++){
        div[cs] = -px[cs]-py[cs]-pz[cs];
        if (x > 0) {
            int sxm = cs - n_y*n_z;
            div[cs] += px[sxm];
        }
        if (y > 0){
            int sym = cs - n_z;;
            div[cs] += py[sym];
        }
        if (z > 0){
            int szm = cs - 1;
            div[cs] += pz[szm];
        }
    }
            
}

void update_spatial_star_flows_channels_first(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const float* lx, const float* ly, const float * lz, const int n_c, const int n_x, const int n_y, const int n_z){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, cs++){
        if (x < n_x-1) {
            int sxm = cs + n_y*n_z;
            update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
        }
        if (y < n_y-1){
            int sym = cs + n_z;
            update_spatial_flows__update_specific_star_flow(cs,sym,g,py,ry,ly);
        }
        if (z < n_z-1){
            int szm = cs + 1;
            update_spatial_flows__update_specific_star_flow(cs,szm,g,pz,rz,lz);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++,cs++){
        div[cs] = -px[cs]-py[cs]-pz[cs];
        if (x > 0) {
            int sxm = cs - n_y*n_z;
            div[cs] += px[sxm];
        }
        if (y > 0){
            int sym = cs - n_z;;
            div[cs] += py[sym];
        }
        if (z > 0){
            int szm = cs - 1;
            div[cs] += pz[szm];
        }
    }
            
}

void update_spatial_flows_channels_first(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_c, const int n_x, const int n_y){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        if (x < n_x-1) {
            int sxm = cs + n_y;
            update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
        }
        if (y < n_y-1){
            int sym = cs + 1;
            update_spatial_flows__update_specific_flow(cs,sym,g,py,ry);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        div[cs] = -px[cs]-py[cs];
        if (x > 0) {
            int sxm = cs - n_y;
            div[cs] += px[sxm];
        }
        if (y > 0){
            int sym = cs - 1;
            div[cs] += py[sym];
        }
    }
            
}

void update_spatial_star_flows_channels_first(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const float* lx, const float* ly, const int n_c, const int n_x, const int n_y){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        if (x < n_x-1) {
            int sxm = cs + n_y;
            update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
        }
        if (y < n_y-1){
            int sym = cs + 1;
            update_spatial_flows__update_specific_star_flow(cs,sym,g,py,ry,ly);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        div[cs] = -px[cs]-py[cs];
        if (x > 0) {
            int sxm = cs - n_y;
            div[cs] += px[sxm];
        }
        if (y > 0){
            int sym = cs - 1;
            div[cs] += py[sym];
        }
    }
            
}


void update_spatial_flows_channels_first(const float* g, float* div, float* px, const float* rx, const int n_c, const int n_x){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++,cs++){
        if (x < n_x-1) {
            int sxm = cs + 1;
            update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++, cs++){
        div[cs] = -px[cs];
        if (x > 0) {
            int sxm = cs - 1;
            div[cs] += px[sxm];
        }
    }

}

void update_spatial_star_flows_channels_first(const float* g, float* div, float* px, const float* rx, const float* lx, const int n_c, const int n_x){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++,cs++){
        if (x < n_x-1) {
            int sxm = cs + 1;
            update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++, cs++){
        div[cs] = -px[cs];
        if (x > 0) {
            int sxm = cs - 1;
            div[cs] += px[sxm];
        }
    }

}
    
void update_spatial_flows(const CPU_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const int dim, const int* const n, const int n_c){
    if(dev.channels_first)
        switch(dim){
            case 1:
                update_spatial_flows_channels_first(g, div, p[0], r[0], n_c, n[0]);
                break;
            case 2:
                update_spatial_flows_channels_first(g, div, p[0], p[1], r[0], r[1], n_c, n[0], n[1]);
                break;
            case 3:
                update_spatial_flows_channels_first(g, div, p[0], p[1], p[2], r[0], r[1], r[2], n_c, n[0], n[1], n[2]);
                break;
        }
    else
        switch(dim){
            case 1:
                update_spatial_flows_channels_last(g, div, p[0], r[0], n_c, n[0]);
                break;
            case 2:
                update_spatial_flows_channels_last(g, div, p[0], p[1], r[0], r[1], n_c, n[0], n[1]);
                break;
            case 3:
                update_spatial_flows_channels_last(g, div, p[0], p[1], p[2], r[0], r[1], r[2], n_c, n[0], n[1], n[2]);
                break;
        }
}
    
void update_spatial_star_flows(const CPU_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const float *const *const l, const int dim, const int* const n, const int n_c){
    if(dev.channels_first)
        switch(dim){
            case 1:
                update_spatial_star_flows_channels_first(g, div, p[0], r[0], l[0], n_c, n[0]);
                break;
            case 2:
                update_spatial_star_flows_channels_first(g, div, p[0], p[1], r[0], r[1], l[0], l[1], n_c, n[0], n[1]);
                break;
            case 3:
                update_spatial_star_flows_channels_first(g, div, p[0], p[1], p[2], r[0], r[1], r[2], l[0], l[1], l[2], n_c, n[0], n[1], n[2]);
                break;
        }
    else
        switch(dim){
            case 1:
                update_spatial_star_flows_channels_last(g, div, p[0], r[0], l[0], n_c, n[0]);
                break;
            case 2:
                update_spatial_star_flows_channels_last(g, div, p[0], p[1], r[0], r[1], l[0], l[1], n_c, n[0], n[1]);
                break;
            case 3:
                update_spatial_star_flows_channels_last(g, div, p[0], p[1], p[2], r[0], r[1], r[2], l[0], l[1], l[2], n_c, n[0], n[1], n[2]);
                break;
        }
}

void init_flows_binary(const CPU_DEVICE & dev, const float* d, float* ps, float* pt, float* u, const int n_s){
    for(int s = 0; s < n_s; s++){
		float d_val = d[s];
		if (d_val > 0.0f){
			ps[s] = d_val;
            pt[s] = 0.0f;
            u[s] = 1.0;
        }else{
			pt[s] = -d_val;
            ps[s] = 0.0f;
            u[s] = 0.0;
        }
	}
}

void init_flows_potts_channels_last(const float* d, float* ps, float* pt, float* u, const int n_c, const int n_s){
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
            if (d[cs] == max_d)
                u[cs] = 1.0f;
            else
                u[cs] = 0.0f;
        }
    }    
}

void init_flows_potts_channels_first(const float* d, float* ps, float* pt, float* u, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++){
        float max_d = -std::numeric_limits<float>::infinity();
        for(int c = 0; c < n_c; c++){
            int cs = idx(c,n_c,s,n_s);
            if( max_d < d[cs] )
                max_d = d[cs];
        }
        //cs -= n_c;
        ps[s] = -max_d;
        for(int c = 0; c < n_c; c++){
            int cs = idx(c,n_c,s,n_s);
            pt[cs] = -max_d;
            if (d[cs] == max_d)
                u[cs] = 1.0f;
            else
                u[cs] = 0.0f;
        }
    }     
}

void init_flows_potts(const CPU_DEVICE & dev,const float* d, float* ps, float* pt, float* u, const int n_s, const int n_c){
    if(dev.channels_first)
        init_flows_potts_channels_first(d, ps, pt, u, n_c, n_s);
    else
        init_flows_potts_channels_last(d, ps, pt, u, n_c, n_s);
}

void init_flows_channels_last(const float* d, float* ps, const int n_c, const int n_s){
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

void init_flows(const CPU_DEVICE & dev, const float* d, float* ps, const int n_c, const int n_s){
    if(dev.channels_first)
        init_flows_channels_first(d, ps, n_c, n_s);
    else
        init_flows_channels_last(d, ps, n_c, n_s);
}