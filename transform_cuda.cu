#include "transform_cuda.h"
#include "primitive_cuda.cuh"

__global__ void test_cuda(Vert_cuda* ptr) {
    printf("%f\n", ptr->pos.x);
    printf("%f\n", ptr[1].pos.w);

}

__global__ void transformObjectToClipKernel(Vert_cuda* verts, float* model_view, float* model_view_inv_trans, float* model_view_persp, Vert_cuda* verts_rst, int input_num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input_num) return;

    transformObjectToClip(verts, model_view, model_view_inv_trans, model_view_persp, verts_rst, i, i);                                 
}

__global__ void transformObjectToScreenKernal(Vert_cuda* verts, float* model_view, float* model_view_inv_trans, float* model_view_persp, float* vp, Vert_cuda* verts_rst, bool* verts_rst_bool, int input_num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input_num) return;
    // int j = i / 3 * 5 + i % 3;
    int j = i;

    transformObjectToClip(verts, model_view, model_view_inv_trans, model_view_persp, verts_rst, i, j);
    __syncthreads();

    // if (i * 3 < input_num) {
    //     int verts_group = i * 5;
    //     sutherland_hodgeman(verts_rst+verts_group, verts_rst_bool+verts_group);
    // }
    // __syncthreads();

    transformClipToScreen(verts_rst, vp, i);

}

__device__ void transformObjectToClip(Vert_cuda* verts, float* model_view, float* model_view_inv_trans, float* model_view_persp, Vert_cuda* verts_rst, int i, int j) {
    verts_rst[j].tex = verts[i].tex;
    verts_rst[j].pos.x = verts[i].pos.x * model_view_persp[0] +
                         verts[i].pos.y * model_view_persp[1] +
                         verts[i].pos.z * model_view_persp[2] + 
                         verts[i].pos.w * model_view_persp[3];
    verts_rst[j].pos.y = verts[i].pos.x * model_view_persp[4] +
                         verts[i].pos.y * model_view_persp[5] +
                         verts[i].pos.z * model_view_persp[6] + 
                         verts[i].pos.w * model_view_persp[7];
    verts_rst[j].pos.z = verts[i].pos.x * model_view_persp[8] +
                         verts[i].pos.y * model_view_persp[9] +
                         verts[i].pos.z * model_view_persp[10] + 
                         verts[i].pos.w * model_view_persp[11];
    verts_rst[j].pos.w = verts[i].pos.x * model_view_persp[12] +
                         verts[i].pos.y * model_view_persp[13] +
                         verts[i].pos.z * model_view_persp[14] + 
                         verts[i].pos.w * model_view_persp[15];
    verts_rst[j].pos_view.x = verts[i].pos.x * model_view[0] + 
                              verts[i].pos.y * model_view[1] + 
                              verts[i].pos.z * model_view[2] + 
                              verts[i].pos.w * model_view[3];
    verts_rst[j].pos_view.y = verts[i].pos.x * model_view[4] + 
                              verts[i].pos.y * model_view[5] + 
                              verts[i].pos.z * model_view[6] + 
                              verts[i].pos.w * model_view[7];
    verts_rst[j].pos_view.z = verts[i].pos.x * model_view[8] + 
                              verts[i].pos.y * model_view[9] + 
                              verts[i].pos.z * model_view[10] + 
                              verts[i].pos.w * model_view[11];
    verts_rst[j].norm.x = verts[i].norm.x * model_view_inv_trans[0] + 
                          verts[i].norm.y * model_view_inv_trans[1] +
                          verts[i].norm.z * model_view_inv_trans[2];
    verts_rst[j].norm.y = verts[i].norm.x * model_view_inv_trans[4] + 
                          verts[i].norm.y * model_view_inv_trans[5] +
                          verts[i].norm.z * model_view_inv_trans[6];
    verts_rst[j].norm.z = verts[i].norm.x * model_view_inv_trans[8] + 
                          verts[i].norm.y * model_view_inv_trans[9] +
                          verts[i].norm.z * model_view_inv_trans[10];      
}

__device__ void sutherland_hodgeman(Vert_cuda* verts, bool* marks) {
    Vec4f_cuda clip_planes[2] = {
        Vec4f_cuda(0.f, 0.f, 1.f, 1.f),
		Vec4f_cuda(0.f, 0.f, -1.f, 1.f)
    };
    int output_size = 3;
    Vert_cuda vert_output[5];
    for (int i = 0; i < output_size; ++i) {
        vert_output[i] = verts[i];
    }
    Vert_cuda vert_input[5];
    for (int i = 0; i < 2; ++i) {
        // input init
        int input_size = 0;
        for (int j = 0; j < output_size; ++j) {
            vert_input[j] = vert_output[i];
            input_size = input_size + 1;
        }
        output_size = 0;
        for (int j = 0; j < input_size; ++j) {
            Vert_cuda* cur = verts + j;
            Vert_cuda* last = verts + (j+input_size-1)%input_size;
            if (inside_way_plane(cur, clip_planes+i)) {
                if (!inside_way_plane(last, clip_planes+i)) {
                    Vert_cuda intersect_point = intersect(last, cur, clip_planes+i);
                    vert_output[output_size] = intersect_point;
                    output_size = output_size + 1;
                }
                vert_output[output_size] = *cur;
                output_size = output_size + 1;
            } else if (inside_way_plane(last, clip_planes+i)) {
                Vert_cuda intersect_point = intersect(last, cur, clip_planes+i);
                vert_output[output_size] = intersect_point;
                output_size = output_size + 1;
            }
        }
    }
    for (int i = 0; i < output_size; ++i) {
        verts[i] = vert_output[i];
        marks[i] = true;
    }
    for (int i = output_size; i < 5; ++i) {
        marks[i] = false;
    }
}

__device__ void transformClipToScreen(Vert_cuda* vert_rst, float* vp, int i) {
    float w = vert_rst[i].pos.w;
    if (w < 1e-6f && w > 0.f) {
        w = 1e-6f;
    } else if (w > -1e-6f && w < 0.f) {
        w = -1e-6f;
    }
    // clip
    vert_rst[i].pos.x = vert_rst[i].pos.x / w;
    vert_rst[i].pos.y = vert_rst[i].pos.y / w;
    vert_rst[i].pos.z = vert_rst[i].pos.z / w;
    vert_rst[i].pos.w = 1.f;
    // viewport transform
    float x = vert_rst[i].pos.x;
    float y = vert_rst[i].pos.y;
    float z = vert_rst[i].pos.z;
    vert_rst[i].pos.x = x * vp[0] + 
                        y * vp[1] +
                        z * vp[2] + 
                        vp[3];
    vert_rst[i].pos.y = x * vp[4] + 
                        y * vp[5] +
                        z * vp[6] + 
                        vp[7];
    vert_rst[i].pos.z = x * vp[8] + 
                        y * vp[9] +
                        z * vp[10] + 
                        vp[11];
    
}

__device__ bool inside_way_plane(Vert_cuda *v, Vec4f_cuda *plane) {
    if (vec4_dot(&(v->pos), plane) >= 0.f) {
        return true;
    }
    return false;
}

__device__ float my_abs(float n) {
    if (n < 0.f) return -n;
    return n;
}

__device__ Vert_cuda intersect(Vert_cuda *v0, Vert_cuda *v1, Vec4f_cuda *plane) {
    float dist0 = my_abs(vec4_dot(&v0->pos, plane));
    float dist1 = my_abs(vec4_dot(&v1->pos, plane));

    float t = dist0 / (dist0 + dist1);
	Vert_cuda lerp_vert;
    lerp_vert.norm = vec3_add(&vec3_mul(&v0->norm, 1-t), &vec3_mul(&v1->norm, t));
	lerp_vert.pos = vec4_add(&vec4_mul(&v0->pos, 1-t), &vec4_mul(&v1->pos, t));
    lerp_vert.pos_view = vec3_add(&vec3_mul(&v0->pos_view, 1-t), &vec3_mul(&v1->pos_view, t));
    lerp_vert.tex = vec2_add(&vec2_mul(&v0->tex, 1-t), &vec2_mul(&v1->tex, t));
    return lerp_vert;
}

__device__ float vec4_dot(Vec4f_cuda *v0, Vec4f_cuda *v1) {
    return v0->x*v1->x + v0->y*v1->y + v0->z*v1->z + v0->w*v1->w;
}

__device__ Vec2f_cuda vec2_mul(Vec2f_cuda *v0, float t) {
    Vec2f_cuda rst;
    rst.x = v0->x * t;
    rst.y = v0->y * t;
    return rst;
}

__device__ Vec3f_cuda vec3_mul(Vec3f_cuda *v0, float t) {
    Vec3f_cuda rst;
    rst.x = v0->x * t;
    rst.y = v0->y * t;
    rst.z = v0->z * t;
    return rst;
}

__device__ Vec4f_cuda vec4_mul(Vec4f_cuda *v0, float t) {
    Vec4f_cuda rst;
    rst.x = v0->x * t;
    rst.y = v0->y * t;
    rst.z = v0->z * t;
    rst.w = v0->w * t;
    return rst;
}

__device__ Vec2f_cuda vec2_add(Vec2f_cuda *v0, Vec2f_cuda *v1) {
    Vec2f_cuda rst;
    rst.x = v0->x + v1->x;
    rst.y = v0->y + v1->y;
    return rst;
}

__device__ Vec3f_cuda vec3_add(Vec3f_cuda *v0, Vec3f_cuda *v1) {
    Vec3f_cuda rst;
    rst.x = v0->x + v1->x;
    rst.y = v0->y + v1->y;
    rst.z = v0->z + v1->z;
    return rst;
}

__device__ Vec4f_cuda vec4_add(Vec4f_cuda *v0, Vec4f_cuda *v1) {

    Vec4f_cuda rst;
    rst.x = v0->x + v1->x;
    rst.y = v0->y + v1->y;
    rst.z = v0->z + v1->z;
    rst.w = v0->w + v1->w;
    return rst;
}

