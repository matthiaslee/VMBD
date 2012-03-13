#include <stdint.h>
#define uint uint32_t
#define IDX(i,j,ld) (((i)*(ld))+(j))
#define IDX3(i,j,k,rows,cols) ((k)*(rows)*(cols)+(i)*(cols)+j)
#define SQRT2H 0.70710678118654752440f
#define SQRT3 1.7320508075688772f
#define TOL 0.0000001f

__global__ void shock(real *a, const float dt, const float h)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real sdata[];
  for (int idx=IDX(threadIdx.y,threadIdx.x,blockDim.x);
       idx < ((blockDim.x+2)*(blockDim.y+2));
       idx += blockDim.x*blockDim.y) {
    const int i_block = (idx / (2+blockDim.x)) - 1;
    const int j_block = (idx % (2+blockDim.x)) - 1;
    int i_abs = i_block + blockIdx.y*blockDim.y;
    int j_abs = j_block + blockIdx.x*blockDim.x;
    if (i_abs < 0) i_abs = 0;
    else if (i_abs >= ROWS) i_abs = ROWS;
    if (j_abs < 0) j_abs = 0;
    else if (j_abs >= COLS) j_abs = COLS;
    if (i_abs >= 0 && i_abs < ROWS &&
        j_abs >= 0 && j_abs < COLS)
      sdata[idx] = a[IDX(i_abs,j_abs,COLS)];
    else
      sdata[idx] = 0.f;
  }
  __syncthreads();
  if (i > ROWS-1 || j > COLS-1)
    return;

  // estimate derivates and minmod x
  real temp1;
  real temp2;
  temp1 = - sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)] +
    sdata[IDX(threadIdx.y+1,threadIdx.x+2,blockDim.x+2)];
  temp2 = + sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)] -
    sdata[IDX(threadIdx.y+1,threadIdx.x,blockDim.x+2)];
  const real Ix = (temp1+temp2)/2.f;
  real temp3;
  if (temp1*temp2 < 0.f)
    temp3 = 0.f;
  else
    temp3 = fminf(fabsf(temp1), fabsf(temp2));

  // estimate derivates and minmod y
  temp1 = - sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)] +
    sdata[IDX(threadIdx.y+2,threadIdx.x+1,blockDim.x+2)];
  temp2 = + sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)] -
    sdata[IDX(threadIdx.y,threadIdx.x+1,blockDim.x+2)];
  const real Iy = (temp1+temp2)/2.f;
  real flow;
  if (temp1*temp2 < 0.f)
    flow = 0.f;
  else
    flow = fminf(fabsf(temp1), fabsf(temp2));

  // compute flow
  flow = sqrtf(temp3*temp3+flow*flow);

  // compute second derivatives
  temp1 = sdata[IDX(threadIdx.y+1,threadIdx.x+2,blockDim.x+2)] +
    sdata[IDX(threadIdx.y+1,threadIdx.x,blockDim.x+2)] -
    2.*sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)];
  temp2 = sdata[IDX(threadIdx.y+2,threadIdx.x+1,blockDim.x+2)] +
    sdata[IDX(threadIdx.y,threadIdx.x+1,blockDim.x+2)] -
    2.*sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)];
  temp3 = (sdata[IDX(threadIdx.y+2,threadIdx.x+2,blockDim.x+2)] -
           sdata[IDX(threadIdx.y+2,threadIdx.x,blockDim.x+2)] -
           sdata[IDX(threadIdx.y,threadIdx.x+2,blockDim.x+2)] +
           sdata[IDX(threadIdx.y,threadIdx.x,blockDim.x+2)]) / 4.f;

  real Inn = (temp2*Iy*Iy + 2.f*temp3*Iy*Ix + temp1*Ix*Ix) / 
    (Ix*Ix + Iy*Iy + 0.00000001f);

  if ((Ix==0)&&(Iy==0))
    Inn = temp1;
 
  temp1 = sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)] - 
    dt * copysignf(flow, Inn) / h;

  a[IDX(i,j,COLS)] = temp1;
    
}

__global__ void bfilter(real *a, const int w, const float sigma_d,
                        const float sigma_r)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real sdata[];
  for (int idx=IDX(threadIdx.y,threadIdx.x,blockDim.x);
       idx < ((blockDim.x+2*w)*(blockDim.y+2*w));
       idx += blockDim.x*blockDim.y) {
    int i_block = (idx / (2*w+blockDim.x));
    const int j_block = (idx - i_block * (2*w+blockDim.x)) - w;
    i_block -= w;
    const int i_abs = i_block + blockIdx.y*blockDim.y;
    const int j_abs = j_block + blockIdx.x*blockDim.x;
    if (i_abs >= 0 && i_abs < ROWS &&
        j_abs >= 0 && j_abs < COLS)
      sdata[idx] = a[IDX(i_abs,j_abs,COLS)];
    else
      sdata[idx] = 0.f;
  }
  __syncthreads();
  if (i > ROWS-1 || j > COLS-1)
    return;

  real sum_weights = 0.f;
  real result = 0.f;
  for (int x = -w; x < (w+1); x++) {
    for (int y = -w; y < (w+1); y++) {
      real weight = sdata[IDX(w+threadIdx.y+y,w+threadIdx.x+x,blockDim.x+2*w)] -
        sdata[IDX(w+threadIdx.y,w+threadIdx.x,blockDim.x+2*w)];
      weight *= weight / (2.*sigma_r*sigma_r);
      weight += ((real) (x*x+y*y)) / (2.*sigma_d*sigma_d);
      weight = expf(-weight);
      if (i+y >= 0 && i+y < ROWS && j+x >= 0 && j+x < COLS) {
        sum_weights += weight;
        result += weight *
          sdata[IDX(w+threadIdx.y+y,w+threadIdx.x+x,blockDim.x+2*w)];
      }
    }
  }
  result /= sum_weights;

  //a[IDX(i,j,COLS)] = sdata[IDX(w+threadIdx.y,w+threadIdx.x,blockDim.x+2*w)];
  a[IDX(i,j,COLS)] = result;

}

__global__ void gamma_compress(real *out)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int k = threadIdx.z;
  if (i > ROWS-1 || j > COLS-1)
    return;

  real result = out[IDX3(i,j,k,ROWS,COLS)];
  if (result <= 0.0031308f && result > 0.f)
    result *= 12.92f;
  else if (result > 0.0031308f && result < 1.f)
    result = 1.055f * powf(result, 0.416666667f) - 0.055f;
  else if (result <= 0.f)
    result = 0.f;
  else
    result = 1.f;
    
  out[IDX3(i,j,k,ROWS,COLS)] = result;
}

__global__ void gamma_uncompress(real *out)   
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int k = threadIdx.z;
  if (i > ROWS-1 || j > COLS-1)
    return;

  real result = out[IDX3(i,j,k,ROWS,COLS)]; 
  if (result <= 0.04045f && result > 0.f)
    result *= 0.07739938080495356037f; 
  else if (result > 0.04045f && result < 1.f)
    result = powf((result + 0.055f) * 0.94786729857819905213f, 2.4f);
  else if (result <= 0.f)
    result = 0.f;
  else
    result = 1.f;

  out[IDX3(i,j,k,ROWS,COLS)] = result; 
}    

__global__ void gradient_valid(real *gradx, real *grady, real *y)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real sdata[];
  for (int idx=IDX(threadIdx.y,threadIdx.x,blockDim.x);
       idx < ((blockDim.x+1)*(blockDim.y+1));
       idx += blockDim.x*blockDim.y) {
    const int i_block = idx / (1+blockDim.x);
    const int j_block = idx % (1+blockDim.x);
    const int i_abs = i_block + blockIdx.y*blockDim.y;
    const int j_abs = j_block + blockIdx.x*blockDim.x;
    if ((i_abs<ROWS) && (j_abs<COLS))
      sdata[idx] = y[IDX(i_abs,j_abs,COLS)];
  }
  __syncthreads();
  if (i > ROWS-1 || j > COLS-1)
    return;

  if (i < ROWS-1)
    grady[IDX(i,j,COLS)] = SQRT2H *
      (sdata[IDX(threadIdx.y,threadIdx.x,blockDim.x+1)] -
       sdata[IDX(threadIdx.y+1,threadIdx.x,blockDim.x+1)]);
  if (j < COLS-1)
    gradx[IDX(i,j,COLS-1)] = SQRT2H *
      (sdata[IDX(threadIdx.y,threadIdx.x,blockDim.x+1)] -
       sdata[IDX(threadIdx.y,threadIdx.x+1,blockDim.x+1)]);
}

__global__ void gradient_same(real *gradx, real *grady, real *y)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real sdata[];
  for (int idx=IDX(threadIdx.y,threadIdx.x,blockDim.x);
       idx < ((blockDim.x+1)*(blockDim.y+1));
       idx += blockDim.x*blockDim.y) {
    const int i_block = idx / (1+blockDim.x);
    const int j_block = idx % (1+blockDim.x);
    const int i_abs = i_block + blockIdx.y*blockDim.y;
    const int j_abs = j_block + blockIdx.x*blockDim.x;
    if ((i_abs<ROWS) && (j_abs<COLS))
      sdata[idx] = y[IDX(i_abs,j_abs,COLS)];
  }
  __syncthreads();
  if (i > ROWS-1 || j > COLS-1)
    return;
  
  if (i < ROWS-1)
    grady[IDX(i,j,COLS)] = SQRT2H *
      (sdata[IDX(threadIdx.y,threadIdx.x,blockDim.x+1)] -
       sdata[IDX(threadIdx.y+1,threadIdx.x,blockDim.x+1)]);
  else if (i == ROWS-1)
    grady[IDX(i,j,COLS)] = 0.f;
  if (j < COLS-1)
    gradx[IDX(i,j,COLS)] = SQRT2H *
      (sdata[IDX(threadIdx.y,threadIdx.x,blockDim.x+1)] -
       sdata[IDX(threadIdx.y,threadIdx.x+1,blockDim.x+1)]);
  else if (j == COLS-1)
    gradx[IDX(i,j,COLS)] = 0.f;
}

__global__ void impad(real *out, real *im, const uint left, const uint top,
                      const uint original_rows, const uint original_cols) {
  const uint i = blockIdx.y*blockDim.y + threadIdx.y;
  const uint j = blockIdx.x*blockDim.x + threadIdx.x;
  if (i > ROWS-1 || j > COLS-1)
    return;

  uint i_source, j_source;
  if (i < top)
    i_source = 0;
  else if (i-top >= original_rows)
    i_source = original_rows - 1;
  else
    i_source = i - top;
  if (j < left)
    j_source = 0;
  else if (j-left >= original_cols)
    j_source = original_cols - 1;
  else
    j_source = j - left;

  out[IDX(i,j,COLS)] = im[IDX(i_source,j_source,original_cols)];
}

__global__ void laplace_valid(real *laplace, real *im)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real sdata[];
  for (int idx=IDX(threadIdx.y,threadIdx.x,blockDim.x);
       idx < ((blockDim.x+2)*(blockDim.y+2));
       idx += blockDim.x*blockDim.y) {
    const int i_block = idx / (2+blockDim.x);
    const int j_block = idx % (2+blockDim.x);
    const int i_abs = i_block + blockIdx.y*blockDim.y;
    const int j_abs = j_block + blockIdx.x*blockDim.x;
    if ((i_abs<ROWS) && (j_abs<COLS))
      sdata[idx] = im[IDX(i_abs,j_abs,COLS)];
  }
  __syncthreads();
  if (i > ROWS-1 || j > COLS-1)
    return;

  if (i < ROWS-2 && j < COLS-2)
    laplace[IDX(i,j,COLS-2)] =
      -0.22360679774997896964f *
      (sdata[IDX(threadIdx.y+2,threadIdx.x+1,blockDim.x+2)] +
       sdata[IDX(threadIdx.y,threadIdx.x+1,blockDim.x+2)] +
       sdata[IDX(threadIdx.y+1,threadIdx.x+2,blockDim.x+2)] +
       sdata[IDX(threadIdx.y+1,threadIdx.x,blockDim.x+2)]) +
      0.89442719099991587856f *
      sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)];
}

__global__ void laplace_same(real *laplace, real *im)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ real sdata[];
  for (int idx=IDX(threadIdx.y,threadIdx.x,blockDim.x);
       idx < ((blockDim.x+2)*(blockDim.y+2));
       idx += blockDim.x*blockDim.y) {
    const int i_block = idx / (2+blockDim.x) - 1;
    const int j_block = idx % (2+blockDim.x) - 1;
    const int i_abs = i_block + blockIdx.y*blockDim.y;
    const int j_abs = j_block + blockIdx.x*blockDim.x;
    if ((i_abs<ROWS) && (j_abs<COLS) &&
        (i_abs>=0) && (j_abs>=0))
      sdata[idx] = im[IDX(i_abs,j_abs,COLS)];
    else
      sdata[idx] = 0.f;
  }
  __syncthreads();
  if (i > ROWS-1 || j > COLS-1)
    return;
  
  if (i>0 && i<ROWS-1 && j>0 && j<COLS-1)
    laplace[IDX(i,j,COLS)] =
      -0.22360679774997896964f *
      (sdata[IDX(threadIdx.y+2,threadIdx.x+1,blockDim.x+2)] +
       sdata[IDX(threadIdx.y,threadIdx.x+1,blockDim.x+2)] +
       sdata[IDX(threadIdx.y+1,threadIdx.x+2,blockDim.x+2)] +
       sdata[IDX(threadIdx.y+1,threadIdx.x,blockDim.x+2)]) +
      0.89442719099991587856f *
      sdata[IDX(threadIdx.y+1,threadIdx.x+1,blockDim.x+2)];
  else
    laplace[IDX(i,j,COLS)] = 0.f;
}

__global__ void laplace_stack_same(real *laplace, real *im)
{
  const int i = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  const int j = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int k = threadIdx.z;

  extern __shared__ real sdata[];
  for (int idx=IDX3(threadIdx.y,threadIdx.x,k,
                    blockDim.y,blockDim.x);
       idx < ((blockDim.x+2)*(blockDim.y+2)*(blockDim.z+2));
       idx += blockDim.x*blockDim.y*blockDim.z) {
    int k_new = idx / ((2+blockDim.x)*(2+blockDim.y));
    int i_new = (idx - k_new*(2+blockDim.x)*(2+blockDim.y))
      / (2+blockDim.x);
    int j_new = idx - k_new*(2+blockDim.x)*(2+blockDim.y) -
      i_new*(2+blockDim.x) - 1;
    k_new--;
    i_new += blockIdx.y*blockDim.y - 1;
    j_new += blockIdx.x*blockDim.x;
    if ((i_new<ROWS) && (j_new<COLS) && (k_new<blockDim.z) &&
        (i_new>=0) && (j_new>=0) && (k_new>=0))
      sdata[idx] = im[IDX3(i_new,j_new,k_new,ROWS,COLS)];
    else
      sdata[idx] = 0.f;
  }
  __syncthreads();
  if (i > ROWS-1 || j > COLS-1)
    return;
  
  if (i>0 && i<ROWS-1 && j>0 && j<COLS-1 && k>0 && k<blockDim.z-1)
    laplace[IDX3(i,j,k,ROWS,COLS)] =
      -0.22360679774997896964f *
      (sdata[IDX3(threadIdx.y+2,threadIdx.x+1,k+1,blockDim.y+2,blockDim.x+2)] +
       sdata[IDX3(threadIdx.y,threadIdx.x+1,k+1,blockDim.y+2,blockDim.x+2)] +
       sdata[IDX3(threadIdx.y+1,threadIdx.x+2,k+1,blockDim.y+2,blockDim.x+2)] +
       sdata[IDX3(threadIdx.y+1,threadIdx.x,k+1,blockDim.y+2,blockDim.x+2)]) +
      0.89442719099991587856f *
      sdata[IDX3(threadIdx.y+1,threadIdx.x+1,k+1,blockDim.y+2,blockDim.x+2)];
  else
    laplace[IDX3(i,j,k,ROWS,COLS)] = 0.f;
}

__global__ void laplace3d_same(real *laplace, real *im)
{
  const int i = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  const int j = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  const int k = threadIdx.z;

  extern __shared__ real sdata[];
  for (int idx=IDX3(threadIdx.y,threadIdx.x,k,
                    blockDim.y,blockDim.x);
       idx < ((blockDim.x+2)*(blockDim.y+2)*(blockDim.z+2));
       idx += blockDim.x*blockDim.y*blockDim.z) {
    int k_new = idx / ((2+blockDim.x)*(2+blockDim.y));
    int i_new = (idx - k_new*(2+blockDim.x)*(2+blockDim.y))
      / (2+blockDim.x);
    int j_new = idx - k_new*(2+blockDim.x)*(2+blockDim.y) -
      i_new*(2+blockDim.x) - 1;
    k_new--;
    i_new += blockIdx.y*blockDim.y - 1;
    j_new += blockIdx.x*blockDim.x;
    if ((i_new<ROWS) && (j_new<COLS) && (k_new<blockDim.z) &&
        (i_new>=0) && (j_new>=0) && (k_new>=0))
      sdata[idx] = im[IDX3(i_new,j_new,k_new,ROWS,COLS)];
    else
      sdata[idx] = 0.f;
  }
  __syncthreads();
  if (i > ROWS-1 || j > COLS-1)
    return;
  
  if (i>0 && i<ROWS-1 && j>0 && j<COLS-1 && k>0 && k<blockDim.z-1)
    laplace[IDX3(i,j,k,ROWS,COLS)] =
     -(sdata[IDX3(threadIdx.y+2,threadIdx.x+1,k+1,blockDim.y+2,blockDim.x+2)] +
       sdata[IDX3(threadIdx.y,threadIdx.x+1,k+1,blockDim.y+2,blockDim.x+2)] +
       sdata[IDX3(threadIdx.y+1,threadIdx.x+2,k+1,blockDim.y+2,blockDim.x+2)] +
       sdata[IDX3(threadIdx.y+1,threadIdx.x,k+1,blockDim.y+2,blockDim.x+2)] +
       sdata[IDX3(threadIdx.y+1,threadIdx.x+1,k+2,blockDim.y+2,blockDim.x+2)] +
       sdata[IDX3(threadIdx.y+1,threadIdx.x+1,k,blockDim.y+2,blockDim.x+2)]) +
      6.f * 
      sdata[IDX3(threadIdx.y+1,threadIdx.x+1,k+1,blockDim.y+2,blockDim.x+2)];
  else
    laplace[IDX3(i,j,k,ROWS,COLS)] = 0.f;
}

__global__ void modify_alpha23_ana(real *out, const float beta)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  if (i > ROWS-1 || j > COLS-1)
    return;

  real result = 0.f;

  real v = out[IDX(i,j,COLS)];
  const real p2d8 = 1.125f * v*v;
  const real qd2 =  1.5f * v*v;
  const real a = 3.f * (v*v - p2d8);
  const real b = -3.f * v * (p2d8 - qd2) - v*v*v;
  const real d = p2d8 * (qd2 - .75f * p2d8) - .75f * v*v*v*v +
    8.f/(27.f*beta*beta*beta);

  real cc0 = 2.f * a;
  real cc1 = a*a - 4.f * d;
  real cc2 = - b*b;

  real shift = -cc0 / 3.f;
  const real a2 = cc0*cc0;
  const real p = cc1 - a2 / 3.f;
  const real q = cc0 * (2.f * a2 / 9.f - cc1) / 3.f + cc2;
  real D = p*p*p / 27.f + q*q * 0.25f;
  
  real alpha2;
  if (p < TOL && p > -TOL) {
    alpha2 = q < 0.f ? cbrtf(-q) : -cbrtf(q);
    alpha2 += shift;
  }
  else if (q < TOL && q > -TOL) {
    if (p < 0.f) {
      alpha2 = sqrtf(-p);
      alpha2 = fmaxf(alpha2 + shift, -alpha2 + shift);
    }
    else 
      alpha2 = shift;
  }
  else if (D < TOL && D > -TOL) {
    alpha2 = q > 0.f ? -cbrtf(q * 0.5f) : cbrtf(-q * 0.5f);
    alpha2 = fmaxf(2.f * alpha2 + shift, -alpha2 + shift);
  }
  else if (D > 0.f) {
    alpha2 = sqrtf(D) - q * 0.5f;
    alpha2 = alpha2 < 0.f ? -cbrtf(-alpha2) : cbrtf(alpha2);
    alpha2 += -p / (alpha2 * 3.f) + shift;
  }
  else {
    const real smp_3 = sqrtf(-p/3.f);
    const real argalpha2 = acosf(1.5f * q / (p * smp_3)) / 3.f;
    real x1 = cosf(argalpha2);
    real x2 = SQRT3 * sqrtf(1.f - x1*x1);
    x1 *= smp_3;
    x2 *= smp_3;
    alpha2 = x2 - x1 + shift;
    alpha2 = fmaxf(alpha2, alpha2 - 2.f * x2);
    alpha2 = fmaxf(alpha2, 2.f * x1 + shift);
  }

  cc1 = sqrtf(alpha2);
  const real rho = -b / cc1;
  cc2 = (a + alpha2 + rho) * 0.5f;
  real comp;
  shift = 0.75f * v;

  D = cc1*cc1 - 4.f*cc2;
  if (D >= 0.f) {
    D = sqrtf(D);
    comp = (-D - cc1) * 0.5f;
    comp += shift;
    comp *= (v>0.f?1.f:-1.f);
    if ((comp > fabsf(v) * 0.5f) && (comp < fabsf(v))) {
      result = fmaxf(result, comp);
    }
    comp = (+D - cc1) * 0.5f;
    comp += shift;
    comp *= (v>0.f?1.f:-1.f);
    if ((comp > fabsf(v) * 0.5f) && (comp < fabsf(v))) {
      result = fmaxf(result, comp);
    }
  }
  
  cc1 = -cc1;
  cc2 -= rho;

  D = cc1*cc1 - 4.f*cc2;
  if (D >= 0.f) {
    D = sqrtf(D);
    comp = (-D - cc1) * 0.5f;
    comp += shift;
    comp *= (v>0.f?1.f:-1.f);
    if ((comp > fabsf(v) * 0.5f) && (comp < fabsf(v))) {
      result = fmaxf(result, comp);
    }
    comp = (+D - cc1) * 0.5f;
    comp += shift;
    comp *= (v>0.f?1.f:-1.f);
    if ((comp > fabsf(v) * 0.5f) && (comp < fabsf(v))) {
      result = fmaxf(result, comp);
    }
  }

  result *= (v>0.f?1.f:-1.f);

  out[IDX(i,j,COLS)] = result;
}

__global__ void modify_alpha(real *out, const float beta, const float alpha,
                             const uint numel) {
  const uint i = blockIdx.y*blockDim.y + threadIdx.y;
  const uint j = blockIdx.x*blockDim.x + threadIdx.x;
  if (IDX(i,j,COLS) >= numel)
    return;

  const real v = out[IDX(i,j,COLS)];
  real x = v;
  for (int k = 0; k < 4; k++) {
    const real fd = alpha*(x>0.f?1.f:-1.f) * powf(fabsf(x), alpha-1.f) +
      beta * (x - v);
    const real fdd = alpha*(alpha-1.f) * powf(fabsf(x), alpha-2.f) + beta;
    x -= fd / fdd;
  }

  if (!isfinite(x))
    x = 0.f;

  if (powf(fabs(x), alpha) + beta * 0.5f * (x - v)*(x - v) >
      beta * 0.5f * v*v)
    x = 0.f;

  out[IDX(i,j,COLS)] = x;
}

__global__ void modify_alpha23(real *out, const float beta, const uint numel) {
  const uint i = blockIdx.y*blockDim.y + threadIdx.y;
  const uint j = blockIdx.x*blockDim.x + threadIdx.x;
  if (IDX(i,j,COLS) >= numel)
    return;

  real v = out[IDX(i,j,COLS)];
  real x = v;
  for (int k = 0; k < 4; k++) {
    const real temp1 = rcbrt(fabsf(x)) * (2.f / 3.f);
    x -= (copysignf(temp1, x) + beta * (x-v)) / (temp1 / (-3.f * fabsf(x))
                                                 + beta);
  }

  if (!isfinite(x))
    x = 0.f;

  real temp = cbrt(fabsf(x));
  temp *= temp;
  if (temp + beta * 0.5f * (x - v)*(x - v) > beta * 0.5f * v*v)
    x = 0.f;

  out[IDX(i,j,COLS)] = x;
}



