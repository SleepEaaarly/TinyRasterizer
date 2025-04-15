#ifndef PRIMITIVE_CUDA_CUH
#define PRIMITIVE_CUDA_CUH

struct Vec4f_cuda {
    float x, y, z, w;
    __device__ Vec4f_cuda() {}
    __device__ Vec4f_cuda(float x_in, float y_in, float z_in, float w_in) { x=x_in; y=y_in; z=z_in; w=w_in; }
};

struct Vec3f_cuda {
    float x, y, z;
    __device__ Vec3f_cuda() {}
    __device__ Vec3f_cuda(float x_in, float y_in, float z_in) { x=x_in; y=y_in; z=z_in; }
};

struct Vec2f_cuda {
    float x, y;
    __device__ Vec2f_cuda() {}
    __device__ Vec2f_cuda(float x_in, float y_in) { x=x_in; y=y_in; }
};

struct Vert_cuda {
    Vec4f_cuda pos;
    Vec3f_cuda norm;
    Vec2f_cuda tex;
    Vec3f_cuda pos_view;
    __device__ Vert_cuda() {}
};

struct Triangle_cuda {
    Vec3f_cuda poses[3];
    Vec2f_cuda texs[3];
    Vec3f_cuda norms[3];
    Vec3f_cuda poses_view[3];
    __device__ Triangle_cuda() {}
    
};

#endif