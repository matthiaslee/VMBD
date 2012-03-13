#define IDX(i,j,ld) (((i)*(ld))+(j))

texture<real, 2> in_tex;

__global__ void resize(real *out, const uint out_rows, const uint out_cols)
{
  const uint i = blockIdx.y*blockDim.y + threadIdx.y;
  const uint j = blockIdx.x*blockDim.x + threadIdx.x;
  if (i > out_rows-1 || j > out_cols-1)
    return;

  out[IDX(i,j,out_cols)] = tex2D(in_tex,
                                 ((float)j) / ((float)out_cols),
                                 ((float)i) / ((float)out_rows));
}
