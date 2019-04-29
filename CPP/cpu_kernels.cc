#include "cpu_kernels.h"

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

void copy(const float* bufferin, float* bufferout, const int n_s){
    for(int i = 0; i < n_s; i++)
        bufferout[i] = bufferin[i];
}

void inc(const float* inc, float* acc, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] += inc[i];
}

void inc(const float* inc, float* acc, const float alpha, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] += alpha*inc[i];
}

void log_buffer(float* buffer, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer[i] = log(buffer[i]+0.000001f);
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

void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) 
    for (int z = 0; z < n_z; z++)
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] = 0.0f;

        //in z+
        if(z < n_z-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rz[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * u[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)];

        //in z-
        if(z > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rz[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)] * u[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)];

        //in y+
        if(y < n_y-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += ry[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * u[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)];

        //in y-
        if(y > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += ry[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)] * u[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)];

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * u[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)];

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rx[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)] * u[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)];

        r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] *= 0.5f;
    }
}

void calculate_r_eff(float* r_eff, const float* rx, const float* ry, const float* u, const int n_x, const int n_y, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) 
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,y,n_y,c,n_c)] = 0.0f;
        
        //in y+
        if(y < n_y-1)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += ry[idxc(x,n_x,y,n_y,c,n_c)] * u[idxc(x,n_x,y+1,n_y,c,n_c)];

        //in y-
        if(y > 0)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += ry[idxc(x,n_x,y-1,n_y,c,n_c)] * u[idxc(x,n_x,y-1,n_y,c,n_c)];

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += rx[idxc(x,n_x,y,n_y,c,n_c)] * u[idxc(x+1,n_x,y,n_y,c,n_c)];

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += rx[idxc(x-1,n_x,y,n_y,c,n_c)] * u[idxc(x-1,n_x,y,n_y,c,n_c)];

        r_eff[idxc(x,n_x,y,n_y,c,n_c)] *= 0.5f;
    }
}

void calculate_r_eff(float* r_eff, const float* rx, const float* u, const int n_x, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,c,n_c)] = 0.0f;

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,c,n_c)] += rx[idxc(x,n_x,c,n_c)] * u[idxc(x+1,n_x,c,n_c)];

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,c,n_c)] += rx[idxc(x-1,n_x,c,n_c)] * u[idxc(x-1,n_x,c,n_c)];

        r_eff[idxc(x,n_x,c,n_c)] *= 0.5f;
    }
}


void aggregate_bottom_up(const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list){
    for (int s = 0; s < n_s; s++)
        for (int l = 0; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            if(n->d == -1){
                bufferout[idxc(s,n_s,n->r,n_r)] = 0.0f;
                for(int c = 0; c < n->c; c++)
                    bufferout[idxc(s,n_s,n->r,n_r)] += bufferout[idxc(s,n_s,n->children[c]->r,n_r)];
            }else{
                bufferout[idxc(s,n_s,n->r,n_r)] = bufferin[idxc(s,n_s,n->d,n_c)];
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


void compute_flows( float* g, const float* u, const float* ps, const float* pt, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const float tau, const float icc, const int n_c, const int n_x, const int n_y, const int n_z){
    const int n_s = n_x*n_y*n_z;
    
    for(int x = 0, s = 0, cs = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, s++){
        for(int c = 0; c < n_c; c++, cs++){
            g[cs] = tau * (div[cs] + pt[cs] - ps[s] - u[cs] * icc);
            if (x != 0) {
                int sxm = idxc(x-1,n_x,y,n_y,z,n_z,c,n_c);
                px[cs] += g[cs] - g[sxm];
                if (px[cs] > rx[sxm])
                    px[cs] = rx[sxm];
                if (px[cs] < -rx[sxm])
                    px[cs] = -rx[sxm];
            }
            if (y != 0){
                int sym = idxc(x,n_x,y-1,n_y,z,n_z,c,n_c);
                py[cs] += g[cs] - g[sym];
                if (py[cs] > ry[sym])
                    py[cs] = ry[sym];
                if (py[cs] < -ry[sym])
                    py[cs] = -ry[sym];
            }
            if (z != 0){
                int szm = idxc(x,n_x,y,n_y,z-1,n_z,c,n_c);
                pz[cs] += g[cs] - g[szm];
                if (pz[cs] > rz[szm])
                    pz[cs] = rz[szm];
                if (pz[cs] < -rz[szm])
                    pz[cs] = -rz[szm];
            }
        }
    }
    
    for(int x = 0, s = 0, cs = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, s++){
        for(int c = 0; c < n_c; c++, cs++){
            div[cs] = -px[cs]-py[cs]-pz[cs];
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,z,n_z,c,n_c);
                div[cs] += px[sxm];
            }
            if (y != 0){
                int sym = idxc(x,n_x,y+1,n_y,z,n_z,c,n_c);
                div[cs] += py[sym];
            }
            if (z != 0){
                int szm = idxc(x,n_x,y,n_y,z+1,n_z,c,n_c);
                div[cs] += pz[szm];
            }
        }
    }
            
}