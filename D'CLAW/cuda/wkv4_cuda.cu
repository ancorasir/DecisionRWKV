// This includes the standard input/output library
#include <stdio.h>
// This includes the assert library for debugging
#include <assert.h>

// Defines a constant for minimum value
#define MIN_VALUE (-1e38)



template <typename F>
__global__ void kernel_forward(const int B,
                                const int T,
                                const int C,
                               const F *__restrict__ const _w,
                               const F *__restrict__ const _u,
                               const F *__restrict__ const _k,
                               const F *__restrict__ const _v,
                               F *__restrict__ const _y) {
   
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    // blockDim.x : B ; threadID.x : C 
    
    const int _b = idx / C;  
    
    const int _c = idx % C;   
    
  
    const int _offset = _b * T * C + _c;
    
    F u = _u[_c];
   
    F w = _w[_c];
    

    // Calculate the pointers for k, v, and y
    const F *__restrict__ const k = _k + _offset;  
    const F *__restrict__ const v = _v + _offset;  
    F *__restrict__ const y = _y + _offset;        

    // Initialize p, q, and o
    F p = 0, q = 0, o = MIN_VALUE;   // init a'_1 = v_0, b'_1 = 1, p_0 = k_0
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < T; i++) { // iterate with time and the degree of parallelism would be ctxLen. Goodjob, bro!
        // the i-th position of the start works
        const int ii = i * C;

        // Calculate intermediate values and update y
        F no = max(o, u + k[ii]);                           // equation (25) calculate q from p_{t-1} and u + k_t
        F A = exp(o - no);                                  // equation (26) first term, overall equal!
        F B = exp(u + k[ii] - no);                          // equation (27) second term, overall equal!
        y[ii] = (A * p + B * v[ii]) / (A * q + B);          // equation (28) overall equal! get wkv_t

        // Calculate intermediate values and update p, q, and o
        no = max(w + o, k[ii]);                             // equation (29) calculate q from p_{t-1} - 2 and k_t
        A = exp(w + o - no);                                // prepare first term
        B = exp(k[ii] - no);                                // prepare second term
        p = A * p + B * v[ii];                              // equation (30)
        q = A * q + B;                                      // equation (31)
        o = no;                                             // equation (32)  p_t = q
    }
}

// CUDA Kernel for the backward pass of a computation
template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C,
                                const F *__restrict__ const _w, const F *__restrict__ const _u, 
                                const F *__restrict__ const _k, const F *__restrict__ const _v,
                                 const F *__restrict__ const _gy,
                                F *__restrict__ const _gw, F *__restrict__ const _gu,
                                 F *__restrict__ const _gk, F *__restrict__ const _gv) {
    // Calculate the index for each thread
    // T:1024; C:512
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    // Calculate the batch and channel indices
    const int _b = idx / C; 
    const int _c = idx % C; 
    // Calculate the offset for indexing into the data
    const int _offset = _b * T * C + _c; 

    F u = _u[_c];   
    F w = _w[_c];

    // Calculate the pointers for k, v, and gy
    const F *__restrict__ const k = _k + _offset;  
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const gy = _gy + _offset;

    // Calculate the pointers for gk and gv
    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    // Initialize arrays for y, z, and zexp
    F y[Tmax], z[Tmax], zexp[Tmax];

    // Initialize variables for gw, gu, p, q, dpdw, dqdw, and o
    F gw = 0, gu = 0;
    F p = 0, q = 0;
    F dpdw = 0, dqdw = 0;
    F o = MIN_VALUE;

    // Perform the forward pass for the backward computation
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        F no = max(o, k[ii] + u);
        F A = exp(o - no);
        F B = exp(k[ii] + u - no);

        F num = A * p + B * v[ii];
        F iden = 1 / (A * q + B);

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = k[ii] + u - no;

        gw += gy[ii] * (dpdw - dqdw * y[i]) * iden * A;  
        gu += gy[ii] * (v[ii] - y[i]) * B * iden;   

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }

    // Initialize variables for gp, gq, and o
    F gp = 0, gq = 0;
    o = MIN_VALUE;

    // Perform the backward pass for the backward computation
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        F A = gy[ii] * z[i] * exp(zexp[i]);
        F B = exp(k[ii] + o);
        gk[ii] = A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
        gv[ii] = A + B * gp;

        F no = max(w + o, zexp[i] - k[ii] - u);
        A = exp(w + o - no);
        B = gy[ii] * z[i] * exp(zexp[i] - k[ii] - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[i];
        o = no;
    }

    // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even though it's not in the forward pass
    // w = -torch.exp(w.float().contiguous())
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] += gw * _w[_c];
    _gu[_offsetBC] += gu;
}

// Function to launch the forward pass kernel
void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y) {
    // Determine the number of threads per block and number of blocks
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    // Ensure that the total number of threads is divisible by the number of threads per block
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);  
    // Launch the forward pass kernel
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

// Function to launch the backward pass kernel
void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv) {
    // Determine the number of threads per block and number of blocks
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    // Ensure that the total number of threads is divisible by the number of threads per block
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    // Launch the backward pass kernel
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
}
