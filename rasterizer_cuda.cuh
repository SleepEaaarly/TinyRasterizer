#ifndef RASTERIZER_CUDA_H
#define RASTERIZER_CUDA_H

#include "primitive_cuda.cuh"
#include <stdio.h>

__global__ void rasterization(Vert_cuda *verts, int verts_size, unsigned char *image, float *z_buffer, int width, int height, int bytespp, 
    unsigned char *texture, int tex_width, int tex_height, int tex_bytespp, Vec3f_cuda *light_color, Vec3f_cuda *light_dir);

__device__ Vec3f_cuda baryCentric(Vert_cuda &vert0, Vert_cuda &vert1, Vert_cuda &vert2, float x, float y);

__device__ Fragment_cuda perspInterpolate(Vert_cuda &vert0, Vert_cuda &vert1, Vert_cuda &vert2, Vec3f_cuda &lambda, int x, int y);

__device__ void setPixel(int x, int y, Color_cuda &c, unsigned char* image, int width, int bytespp);

__device__ Color_cuda getTextureColor(unsigned char* texture, int tex_width, int tex_height, int tex_bytespp, float u, float v);

__device__ Color_cuda BlinnPhongShade(Fragment_cuda &frag, unsigned char* texture, int tex_width, int tex_height, int tex_bytespp, Vec3f_cuda* light_color, Vec3f_cuda* light_dir);

__device__ Vec4f_cuda CudaUchar2Float(Color_cuda c);

__device__ Color_cuda CudaFloat2Uchar(Vec4f_cuda f);

__device__ Vec3f_cuda vec3f_add(Vec3f_cuda &v0, Vec3f_cuda &v1);

#endif

