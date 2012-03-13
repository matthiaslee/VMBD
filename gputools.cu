#include <stdint.h>
#define uint uint32_t
#define IDX(i,j,ld) (((i)*(ld))+(j))
#define IDX3(i,j,k,rows,cols) ((k)*(rows)*(cols)+(i)*(cols)+j)
#include<cufft.h>
#include<cuda.h>

//__global__ void copy_Kernel( real *out, const int *outm, const int *outn, real *in, const int ink, const int inm, const int inn, const int om, const int on, const int startRow){
//  int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
//  int j = blockIdx.y*BLOCK_SIZE + threadIdx.y;
// 
//  dy[IDX( om+i, on+j, outn )] += dys[IDX( startRow+i, j, inn )];
//}


__global__ void chop_mod_pad_Kernel(float *dx, int xm, int xn, int xr, float *dy, int ym, int yn, float *dw, int wm, int wn, int om, int on, int cm, int cn, int hopm, int hopn){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( i<(xm*xr) && j<xn ){
    
    int k = i/xm;
    int m = k/cn;
    int n = k-(m*cn);

    if( k < xr ){
      if( ((k*xm)+om-1)<i && i<((k*xm)+wm+om) && (on-1)<j && j<(wn+on) ){
	dx[IDX( i, j, xn)] = dw[IDX( (k*wm)-(k*xm)+i-om, j-on, wn)] * dy[IDX( (m*hopm)-(k*xm)+i-om, (n*hopn)+j-on, yn )];
      }
      else{
	dx[IDX( i, j, xn )] = 0.0f;
      }
    }
  }
}

__global__ void chop_mod_pad_ComplexKernel(cufftComplex *dx, int xm, int xn, int xr, float *dy, int ym, int yn, float *dw, int wm, int wn, int om, int on, int cm, int cn, int hopm, int hopn){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( i<(xm*xr) && j<xn ){
    
    int k = i/xm;
    int m = k/cn;
    int n = k-(m*cn);

    if( k < xr ){
      if( ((k*xm)+om-1)<i && i<((k*xm)+wm+om) && (on-1)<j && j<(wn+on) ){
	dx[IDX( i, j, xn)].x = dw[IDX( (k*wm)-(k*xm)+i-om, j-on, wn)] * dy[IDX( (m*hopm)-(k*xm)+i-om, (n*hopn)+j-on, yn )];
	dx[IDX( i, j, xn)].y = 0.0f;
      }
      else{
	dx[IDX( i, j, xn )].x = 0.0f;
	dx[IDX( i, j, xn )].y = 0.0f;
      }
    }
  }
}

__global__ void chop_pad_Kernel(float *dx, int xm, int xn, int xr, float *dy, int ym, int yn, int wm, int wn, int om, int on, int cm, int cn, int hopm, int hopn){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( i<(xm*xr) && j<xn ){
    
    int k = i/xm;
    int m = k/cn;
    int n = k-(m*cn);

    if( k < xr ){
      if( ((k*xm)+om-1)<i && i<((k*xm)+wm+om) && (on-1)<j && j<(wn+on) ){
	dx[IDX( i, j, xn)] = dy[IDX( (m*hopm)-(k*xm)+i-om, (n*hopn)+j-on, yn )];
      }
      else{
	dx[IDX( i, j, xn )] = 0.0f;
      }
    }
  }
}

__global__ void chop_pad_Kernel_test(float *out, const uint xm, const uint xn,
                                     const uint xr, float *in,
                                     const uint ym, const uint yn,
                                     const uint wm, const uint wn,
                                     const uint om, const uint on,
                                     const uint cm, const uint cn,
                                     const uint hopm, const uint hopn){
  const uint i = (__umul24(blockIdx.x, blockDim.x) + threadIdx.x) / xn;
  const uint j = (__umul24(blockIdx.x, blockDim.x) + threadIdx.x) % xn;
  const uint k = (__umul24(blockIdx.y, blockDim.y) + threadIdx.y);
  const uint i_patch = k / cn;
  const uint j_patch = k % cn;
  if (i >= xm || j >= xn || k >= xr )
    return;

  if (i >= om && j >= on && i < om+wm && j < on+wn) {
    out[IDX3(i, j, k, xm, xn)] = in[IDX(i-om+__umul24(i_patch, hopm),
                                        j-on+__umul24(j_patch, hopn), yn)];
  }
  else {
    out[IDX3(i, j, k, xm, xn)] = 0.f;
  }
}

__global__ void chop_pad_ComplexKernel(cufftComplex *dx, int xm, int xn, int xr, float *dy, int ym, int yn, int wm, int wn, int om, int on, int cm, int cn, int hopm, int hopn){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( i<(xm*xr) && j<xn ){
    
    int k = i/xm;
    int m = k/cn;
    int n = k-(m*cn);

    if( k < xr){
      if( ((k*xm)+om-1)<i && i<((k*xm)+wm+om) && (on-1)<j && j<(wn+on) ){
	dx[IDX( i, j, xn)].x = dy[IDX( (m*hopm)-(k*xm)+i-om, (n*hopn)+j-on, yn )];
	dx[IDX( i, j, xn)].y = 0.0f;
      }
      else{
	dx[IDX( i, j, xn )].x = 0.0f;
	dx[IDX( i, j, xn )].y = 0.0f;
      }
    }
  }
}

__global__ void chop_pad_ComplexKernel_test(cufftComplex *out, const uint xm,
                                     const uint xn, const uint xr, float *in,
                                     const uint ym, const uint yn,
                                     const uint wm, const uint wn,
                                     const uint om, const uint on,
                                     const uint cm, const uint cn,
                                     const uint hopm, const uint hopn){
  const uint i = (__umul24(blockIdx.x, blockDim.x) + threadIdx.x) / xn;
  const uint j = (__umul24(blockIdx.x, blockDim.x) + threadIdx.x) % xn;
  const uint k = (__umul24(blockIdx.y, blockDim.y) + threadIdx.y);
  const uint i_patch = k / cn;
  const uint j_patch = k % cn;
  if (i >= xm || j >= xn || k >= xr )
    return;

  if (i >= om && j >= on && i < om+wm && j < on+wn) {
    out[IDX3(i, j, k, xm, xn)].x = in[IDX(i-om+__umul24(i_patch, hopm),
                                          j-on+__umul24(j_patch, hopn), yn)];
    out[IDX3(i, j, k, xm, xn)].y = 0.f;
  }
  else {
    out[IDX3(i, j, k, xm, xn)].x = 0.f;
    out[IDX3(i, j, k, xm, xn)].y = 0.f;
  }
}

__global__ void zeroPadKernel(float *dx, int m, int n, float *dy, int p, int q, int s, int t){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( i<m && j<n ){
    if( (s-1)<i && i<(p+s) && (t-1)<j && j<(q+t) ){
      dx[IDX( i, j, n )] = dy[IDX( i-s, j-t, q )];
    }
    else{
      dx[IDX( i, j, n )] = 0.0f;
    }
  }
}

__global__ void zeroPadComplexKernel(cufftComplex *dx, int m, int n, float *dy, int p, int q, int s, int t){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( i<m && j<n ){
    if( (s-1)<i && i<(p+s) && (t-1)<j && j<(q+t) ){
      dx[IDX( i, j, n )].x = dy[IDX( i-s, j-t, q )];
      dx[IDX( i, j, n )].y = 0.0f;
    }
    else{
    dx[IDX( i, j, n )].x = 0.0f;
    dx[IDX( i, j, n )].y = 0.0f;
    }
  }
}

__global__ void pad_stack_Kernel(float *dx, int xk, int xm, int xn, float *dxp, int pm, int pn, int om, int on){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( i<(pm*xk) && j<(pn) ){
    
    int k = i/pm;

    if( k < xk ){
      if( ((k*pm)+om-1)<i && i<((k*pm)+xm+om) && (on-1)<j && j<(xn+on) ){
	dxp[IDX( i, j, pn) ] = dx[IDX( i-om+(k*(xm-pm)), j-on, xn )];
      }
      else{
	dxp[IDX( i, j, pn )] = 0.0f;
      }
    }
  }
}

__global__ void pad_stack_ComplexKernel(float *dx, int xk, int xm, int xn, cufftComplex *dxp, int pm, int pn, int om, int on){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( i<(pm*xk) && j<(pn) ){
    
    int k = i/pm;

    if( k < xk ){
      if( ((k*pm)+om-1)<i && i<((k*pm)+xm+om) && (on-1)<j && j<(xn+on) ){
	dxp[IDX( i, j, pn) ].x = dx[IDX( i-om+(k*(xm-pm)), j-on, xn )];
	dxp[IDX( i, j, pn) ].y = 0.0f;
      }
      else{
	dxp[IDX( i, j, pn )].x = 0.0f;
	dxp[IDX( i, j, pn )].y = 0.0f;
      }
    }
  }
}


__global__ void crop_stack_Kernel(float *dx, int xk, int xm, int xn, float *dxp, int pm, int pn, int om, int on){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  int k = i/xm;

  if( k < xk ){
    if( ((k*xm)-1)<i && i<((k+1)*xm) && j<xn){
      dx[IDX( i, j, xn) ] = dxp[IDX( i+om+(k*(pm-xm)), j+on, pn )];
    }
  }
}


__global__ void crop_stack_ComplexKernel(cufftComplex *dx, int xk, int xm, int xn, cufftComplex *dxp, int pm, int pn, int om, int on){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  int k = i/xm;

  if( k < xk){
    if( ((k*xm)-1)<i && i<((k+1)*xm) && j<xn){
      dx[IDX( i, j, xn) ].x = dxp[IDX( i+om+(k*(pm-xm)), j+on, pn )].x;
      dx[IDX( i, j, xn) ].y = dxp[IDX( i+om+(k*(pm-xm)), j+on, pn )].y;
    }
  }
}


__global__ void crop_Kernel(float *dx, int m, int n, float *dy, int p, int q, int s, int t){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( (s-1)<i && i<m+s ){
    if( (t-1)<j && j<n+t ){
      dx[IDX( i-s, j-t, n )] = dy[IDX( i, j, q )];
    }
  }
}

__global__ void crop_ComplexKernel(float *dx, int m, int n, cufftComplex *dy, int p, int q, int s, int t){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( (s-1)<i && i<m+s ){
    if( (t-1)<j && j<n+t ){
      dx[IDX( i-s, j-t, n )] = dy[IDX( i, j, q )].x;
    }
  }
}

__global__ void ola_ComplexKernel_test(float *out, cufftComplex *in, const uint outm, const uint outn, const uint inm, const uint inn, const uint wm, const uint wn, const uint om, const uint on, const uint cm,const uint cn, const uint hopm, const uint hopn) {
  const int i = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  const int j = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (0<=i && i<outm && 0<=j && j<outn){
    const int ci = min(i / hopm, cm - 1);
    const int cj = min(j / hopn, cn - 1);

    float result = 0.f;
    for (int d_i = 0; i<wm+hopm*(ci-d_i) && ci-d_i>=0; d_i++) {
      for (int d_j = 0; j<wn+hopn*(cj-d_j) && cj-d_j>=0; d_j++) {
	result += in[IDX3(om+i+(d_i-ci)*hopm,
			  on+j+(d_j-cj)*hopn,
			  IDX(ci-d_i,cj-d_j,cn),inm,inn)].x;
      }
    }
    out[IDX(i,j,outn)] = result;
  }
}

__global__ void ola_Kernel_test(float *out, float *in, const uint outm, const uint outn, const uint inm, const uint inn, const uint wm, const uint wn, const uint om, const uint on, const uint cm,const uint cn, const uint hopm, const uint hopn) {
  const int i = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  const int j = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (0<=i && i<outm && 0<=j && j<outn){
    const int ci = min(i / hopm, cm - 1);
    const int cj = min(j / hopn, cn - 1);

    float result = 0.f;
    for (int d_i = 0; i<wm+hopm*(ci-d_i) && ci-d_i>=0; d_i++) {
      for (int d_j = 0; j<wn+hopn*(cj-d_j) && cj-d_j>=0; d_j++) {
	result += in[IDX3(om+i+(d_i-ci)*hopm,
			  on+j+(d_j-cj)*hopn,
			  IDX(ci-d_i,cj-d_j,cn),inm,inn)];
      }
    }
    out[IDX(i,j,outn)] = result;
  }
}

__global__ void comp_ola_deconv_Kernel(cufftComplex *out, const int outk, const int outm, const int outn, cufftComplex *f, cufftComplex *y, cufftComplex *L, const float alpha, const float beta) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  const int k = i / outm;
  
  if(k < outk){
    if(((k*outm)-1)<i && i<((k+1)*outm) && j<(outn)){

      const float numx = f[IDX(i,j,outn)].x * y[IDX(i,j,outn)].x + 
	                    f[ IDX(i,j,outn)].y * y[IDX(i,j,outn)].y;
      const float numy = -f[IDX(i,j,outn)].y * y[IDX(i,j,outn)].x +
                            f[IDX(i,j,outn)].x * y[IDX(i,j,outn)].y;
	
      const float denomx = f[IDX(i,j,outn)].x * f[IDX(i,j,outn)].x + 
	                    f[IDX(i,j,outn)].y * f[IDX(i,j,outn)].y + 
	                    alpha * L[IDX(i-(k*outm),j,outn)].x + beta;
      const float denomy =  alpha * L[IDX(i-(k*outm),j,outn)].y;

      out[IDX(i,j,outn)].x = (numx * denomx + numy * denomy) /
				 (denomx * denomx + denomy * denomy);
      out[IDX(i,j,outn)].y = (numy * denomx - numx * denomy) / 
				 (denomx * denomx + denomy * denomy);
    }
  }
}

__global__ void comp_ola_gdeconv_Kernel(cufftComplex *out, const int outk, const int outm, const int outn, cufftComplex *xx, cufftComplex *xy, cufftComplex *yx, cufftComplex *yy, cufftComplex *L, const float alpha, const float beta) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  const int k = i/outm;
    
  if( k < outk ){
    if( ((k*outm)-1)<i && i<((k+1)*outm)  && j<(outn) ){

      const float   numx =  xx[ IDX(i, j, outn) ].x * yx[ IDX(i, j, outn) ].x + 
	                    xx[ IDX(i, j, outn) ].y * yx[ IDX(i, j, outn) ].y + 
	                    xy[ IDX(i, j, outn) ].x * yy[ IDX(i, j, outn) ].x + 
	                    xy[ IDX(i, j, outn) ].y * yy[ IDX(i, j, outn) ].y;
      const float   numy = -xx[ IDX(i, j, outn) ].y * yx[ IDX(i, j, outn) ].x +
                            xx[ IDX(i, j, outn) ].x * yx[ IDX(i, j, outn) ].y 
	                   -xy[ IDX(i, j, outn) ].y * yy[ IDX(i, j, outn) ].x +
                       	    xy[ IDX(i, j, outn) ].x * yy[ IDX(i, j, outn) ].y;

      const float denomx =  xx[ IDX(i, j, outn) ].x * xx[ IDX(i, j, outn) ].x + 
	                    xx[ IDX(i, j, outn) ].y * xx[ IDX(i, j, outn) ].y + 
	                    xy[ IDX(i, j, outn) ].x * xy[ IDX(i, j, outn) ].x + 
	                    xy[ IDX(i, j, outn) ].y * xy[ IDX(i, j, outn) ].y + 
                            alpha * L[ IDX(i-(k*outm), j, outn) ].x + beta;
      const float denomy =  alpha * L[ IDX(i-(k*outm), j, outn) ].y;

      out[IDX( i, j, outn) ].x = (  numx * denomx +   numy * denomy)/
				 (denomx * denomx + denomy * denomy);
      out[IDX( i, j, outn) ].y = (  numy * denomx -   numx * denomy)/ 
			         (denomx * denomx + denomy * denomy);
    }
  }
}

__global__ void comp_ola_sdeconv_Kernel(cufftComplex *out, const int outk, const int outm, const int outn, cufftComplex *gx, cufftComplex *gy, cufftComplex *xx, cufftComplex *xy, cufftComplex *Ftpy, cufftComplex *f, cufftComplex *L, const float alpha, const float beta, const float gamma) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  const int k = i/outm;
    
  if( k < outk ){
    if( ((k*outm)-1)<i && i<((k+1)*outm)  && j<(outn) ){

      const float   numx =  gx[ IDX(i-(k*outm), j, outn) ].x * xx[ IDX(i, j, outn) ].x + 
	                    gx[ IDX(i-(k*outm), j, outn) ].y * xx[ IDX(i, j, outn) ].y + 
	                    gy[ IDX(i-(k*outm), j, outn) ].x * xy[ IDX(i, j, outn) ].x + 
	                    gy[ IDX(i-(k*outm), j, outn) ].y * xy[ IDX(i, j, outn) ].y +
	                    Ftpy[ IDX(i, j, outn) ].x / (alpha * beta);
      const float   numy = -gx[ IDX(i-(k*outm), j, outn) ].y * xx[ IDX(i, j, outn) ].x +
                            gx[ IDX(i-(k*outm), j, outn) ].x * xx[ IDX(i, j, outn) ].y 
	                   -gy[ IDX(i-(k*outm), j, outn) ].y * xy[ IDX(i, j, outn) ].x +
                       	    gy[ IDX(i-(k*outm), j, outn) ].x * xy[ IDX(i, j, outn) ].y +
	                    Ftpy[ IDX(i, j, outn) ].y / (alpha * beta);
      const float denomx =  (f[ IDX(i, j, outn) ].x * f[ IDX(i, j, outn) ].x + 
			     f[ IDX(i, j, outn) ].y * f[ IDX(i, j, outn) ].y ) / (alpha * beta) + 
                            L[ IDX(i-(k*outm), j, outn) ].x + gamma;
      const float denomy =  L[ IDX(i-(k*outm), j, outn) ].y;

      out[IDX( i, j, outn) ].x = (  numx * denomx +   numy * denomy)/
				 (denomx * denomx + denomy * denomy);
      out[IDX( i, j, outn) ].y = (  numy * denomx -   numx * denomy)/ 
			         (denomx * denomx + denomy * denomy);
    }
  }
}
