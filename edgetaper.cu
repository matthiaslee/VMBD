#define IDX(i,j,ld) (((i)*(ld))+(j))

__constant__ real wx[wx_size];
__constant__ real wy[wy_size];

__global__ void edgetaper(real *im, const int hsfx, const int hsfy)
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  if (i > ROWS-1 || j > COLS-1)
    return;

  real value;

  if (i<hsfy || i>ROWS-1-hsfy || j<hsfx || j>COLS-1-hsfx)
    value = im[IDX(i,j,COLS)];
  else {
    return;
  }

  if (i<hsfy)
    value *= wy[i];
  else if (i>ROWS-1-hsfy)
    value *= wy[ROWS-1-i];
  if (j<hsfx)
    value *= wx[j];
  else if (j>COLS-1-hsfx)
    value *= wx[COLS-1-j];

  im[IDX(i,j,COLS)] = value;
}
