#ifndef RASTERIZER_CUDA_H
#define RASTERIZER_CUDA_H

#include "primitive_cuda.cuh"
#include <stdio.h>

__global__ void rasterization(Vert_cuda *verts, int verts_size, unsigned char *image, float *z_buffer, int width, int height, int bytespp);

__device__ Vec3f_cuda baryCentric(Vert_cuda &vert0, Vert_cuda &vert1, Vert_cuda &vert2, float x, float y);

__device__ Fragment_cuda perspInterpolate(Vert_cuda &vert0, Vert_cuda &vert1, Vert_cuda &vert2, Vec3f_cuda &lambda, int x, int y);

__device__ void setPixel(int x, int y, Color_cuda &c, unsigned char* image, int width, int bytespp);

#endif

