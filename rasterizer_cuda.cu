#include "rasterizer_cuda.cuh"

__global__ void rasterization(Vert_cuda *verts, int verts_size, unsigned char *image, float *z_buffer, int width, int height, int bytespp) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) return;

    // clear Image and Zbuffer
    int idx = row * width + col;
    for (int i = 0; i < bytespp; ++i) {
        image[bytespp*idx + i] = 0;
    }
    z_buffer[idx] = INFINITY;

    // rasterize all verts, 3 in groups
    for (int i = 0; i < verts_size / 3; ++i) {
        Vert_cuda vert0 = verts[3*i];
        Vert_cuda vert1 = verts[3*i+1];
        Vert_cuda vert2 = verts[3*i+2];

        // bbox judgement
        int bboxmin_x = max(0, (int)(fminf(vert0.pos.x, fminf(vert1.pos.x, vert2.pos.x))));
        int bboxmax_x = min(width-1, (int)(fmaxf(vert0.pos.x, fmaxf(vert1.pos.x, vert2.pos.x))));
        int bboxmin_y = max(0, (int)(fminf(vert0.pos.y, fminf(vert1.pos.y, vert2.pos.y))));
        int bboxmax_y = min(height-1, (int)(fmaxf(vert0.pos.y, fmaxf(vert1.pos.y, vert2.pos.y))));
        if (col < bboxmin_x || col > bboxmax_x || row < bboxmin_y || row > bboxmax_y) {
            continue;
        }
        
        Vec3f_cuda lambda = baryCentric(vert0, vert1, vert2, (float)(col+0.5f), (float)(row+0.5f));
        if (lambda.x >= 0.f && lambda.y >= 0.f && lambda.z >= 0.f) {
            Fragment_cuda frag = perspInterpolate(vert0, vert1, vert2, lambda, col, row);
            float z = frag.z;
            if (z < z_buffer[idx]) {
                Color_cuda c = Color_cuda(255, 0, 0, 0);
                setPixel(col, row, c, image, width, bytespp);
            }
        }
    }

}

__device__ Vec3f_cuda baryCentric(Vert_cuda &vert0, Vert_cuda &vert1, Vert_cuda &vert2, float x, float y) {
    float v0_x = vert0.pos.x;   float v0_y = vert0.pos.y;
    float v1_x = vert1.pos.x;   float v1_y = vert1.pos.y;
    float v2_x = vert2.pos.x;   float v2_y = vert2.pos.y;

    // gain vec1, vec2
    float vec1_x = v2_x - v0_x; float vec1_y = v1_x - v0_x; float vec1_z = v0_x - x;
    float vec2_x = v2_y - v0_y; float vec2_y = v1_y - v0_y; float vec2_z = v0_y - y;

    // cross vec1 and vec2 to gain u
    float u_x = vec1_y*vec2_z - vec1_z*vec2_y;
    float u_y = vec1_z*vec2_x - vec1_x*vec2_z;
    float u_z = vec1_x*vec2_y - vec1_y*vec2_x;

    if (fabsf(u_z) < 1e-6) return Vec3f_cuda(-1.f, 1.f, 1.f);
    return Vec3f_cuda(1.f-(u_x+u_y)/u_z, u_y/u_z, u_x/u_z);
}

__device__ Fragment_cuda perspInterpolate(Vert_cuda &vert0, Vert_cuda &vert1, Vert_cuda &vert2, Vec3f_cuda &lambda, int x, int y) {
    float z0 = vert0.pos.z < 1e-6 ? 1e-6 : vert0.pos.z;
    float z1 = vert1.pos.z < 1e-6 ? 1e-6 : vert1.pos.z;
    float z2 = vert2.pos.z < 1e-6 ? 1e-6 : vert2.pos.z;

    float z = 1.f / (lambda.x/z0+lambda.y/z1+lambda.z/z2);

    Vec2f_cuda tex_coord;
    tex_coord.x = z*(lambda.x*vert0.tex.x/z0+lambda.y*vert1.tex.x/z1+lambda.z*vert2.tex.x/z2);
    tex_coord.y = z*(lambda.x*vert0.tex.y/z0+lambda.y*vert1.tex.y/z1+lambda.z*vert2.tex.y/z2);

    Vec3f_cuda norm;
    norm.x = z*(lambda.x*vert0.norm.x/z0+lambda.y*vert1.norm.x/z1+lambda.z*vert2.norm.x/z2);
    norm.y = z*(lambda.x*vert0.norm.y/z0+lambda.y*vert1.norm.y/z1+lambda.z*vert2.norm.y/z2);
    norm.z = z*(lambda.x*vert0.norm.z/z0+lambda.y*vert1.norm.z/z1+lambda.z*vert2.norm.z/z2);

    Vec3f_cuda pos_view;
    pos_view.x = z*(lambda.x*vert0.pos_view.x/z0+lambda.y*vert1.pos_view.x/z1+lambda.z*vert2.pos_view.x/z2);
    pos_view.y = z*(lambda.x*vert0.pos_view.y/z0+lambda.y*vert1.pos_view.y/z1+lambda.z*vert2.pos_view.y/z2);
    pos_view.z = z*(lambda.x*vert0.pos_view.z/z0+lambda.y*vert1.pos_view.z/z1+lambda.z*vert2.pos_view.z/z2);


    return Fragment_cuda(Vec2i_cuda(x, y), tex_coord, norm, pos_view, z);
}

__device__ void setPixel(int x, int y, Color_cuda &c, unsigned char* image, int width, int bytespp) {
    int st = (x+y*width)*bytespp;
    for (int i = 0; i < bytespp; ++i) {
        image[st+i] = c.raw[i];
    }
}