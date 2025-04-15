#ifndef TRANSFORM_CUDA_H
#define TRANSFORM_CUDA_H

#include <cuda_runtime.h>
#include "geometry.h"
#include "primitive.h"
#include "primitive_cuda.cuh"
#include <stdio.h>

__global__ void test_cuda(Vert_cuda* ptr);

__global__ void transformObjectToClipKernel(Vert_cuda* verts, float* model_view, float* model_view_inv_trans, float* model_view_persp, Vert_cuda* verts_rst, int input_num);
__global__ void transformObjectToScreenKernal(Vert_cuda* verts, float* model_view, float* model_view_inv_trans, float* model_view_persp, float* vp, Vert_cuda* verts_rst, bool* verts_rst_bool, int input_num);

__device__ void transformObjectToClip(Vert_cuda* verts, float* model_view, float* model_view_inv_trans, float* model_view_persp, Vert_cuda* verts_rst, int i, int j);
__device__ void sutherland_hodgeman(Vert_cuda* verts, bool* marks);
__device__ void transformClipToScreen(Vert_cuda* vert_rst, float* vp, int i);
__device__ bool inside_way_plane(Vert_cuda *v, Vec4f_cuda *plane);
__device__ float my_abs(float n);
__device__ Vert_cuda intersect(Vert_cuda *v0, Vert_cuda *v1, Vec4f_cuda *plane);
__device__ float vec4_dot(Vec4f_cuda *v0, Vec4f_cuda *v1);
__device__ Vec2f_cuda vec2_mul(Vec2f_cuda *v0, float t);
__device__ Vec3f_cuda vec3_mul(Vec3f_cuda *v0, float t);
__device__ Vec4f_cuda vec4_mul(Vec4f_cuda *v0, float t);
__device__ Vec2f_cuda vec2_add(Vec2f_cuda *v0, Vec2f_cuda *v1);
__device__ Vec3f_cuda vec3_add(Vec3f_cuda *v0, Vec3f_cuda *v1);
__device__ Vec4f_cuda vec4_add(Vec4f_cuda *v0, Vec4f_cuda *v1);


#endif