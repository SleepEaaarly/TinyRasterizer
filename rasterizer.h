#ifndef RASTERIZER_H
#define RASTERIZER_H

#include "shader.h"
#include "buffer.h"

class Rasterizer {
private:
    Shader *shader;
    Buffer *z_buffer;
    TGAImage *image;

    // cuda pointers
    Vert* d_verts_scr;
    float* d_z_buffer;
    unsigned char* d_image;
    int num_verts_scr;

    void drawTriangle(Triangle &tri);
    Vec3f barycentric(Vec3f *t, Vec2f p);
    void zeroCheck(Vec3f &depth);
    float depth_persp_interpolate(Vec3f depth, Vec3f &lambda);
    Vec2f vec2_persp_interpolate(Vec2f* vec2_arr, float depth, Vec3f depth_arr, Vec3f &lambda);
    Vec3f vec3_persp_interpolate(Vec3f* vec3_arr, float depth, Vec3f depth_arr, Vec3f &lambda);


public:
    Rasterizer(Shader &shader, Buffer &z_buffer, TGAImage &image);
    void rasterizeTriangles(std::vector<Triangle> &triangles);
    
    void cudaInit(Vert* d_verts_rst_in, int num_verts_rst);
    void cudaRelease();
    void cudaUpdateZBuffer();
    void rasterizeVertsCuda();
};  

#endif