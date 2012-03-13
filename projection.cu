#define IDX(i,j,ld) (((i)*(ld))+(j))

__global__ void projection(real *out, real *fs, real *basis,
                           const uint basis_length) {
  const uint i = threadIdx.x;
  const uint j = threadIdx.y;
  const uint cols = blockDim.y;
  __shared__ real sdata[BLOCK_SIZE];
  sdata[IDX(i,j,cols)] = 0.f;
  
  if (blockIdx.y*cols + j >= basis_length)
    return;

  for (uint k = i; k < ROWS*COLS; k += blockDim.x) {
    sdata[IDX(i,j,cols)] += fs[k] * basis[ROWS*COLS*j + k];
  }
  
  __syncthreads();
  
  if (i == 0) {
    real result = 0.f;
    for (uint k = 0; k < blockDim.x; ++k) {
      result += sdata[IDX(k,j,cols)];
    }
    out[blockIdx.y*cols + j] = result;
  }
}

