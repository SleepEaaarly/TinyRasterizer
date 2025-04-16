#include "rasterizer_cuda.cuh"

__global__ void rasterization(Vert_cuda *verts, int verts_size, unsigned char *image, float *z_buffer, int width, int height, int bytespp, 
                                unsigned char *texture, int tex_width, int tex_height, int tex_bytespp, Vec3f_cuda *light_color, Vec3f_cuda *light_dir) {
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
                z_buffer[idx] = z;
                Color_cuda c = BlinnPhongShade(frag, texture, tex_width, tex_height, tex_bytespp, light_color, light_dir);
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

__device__ Color_cuda getTextureColor(unsigned char* texture, int tex_width, int tex_height, int tex_bytespp, float u, float v) {
    int x = (int)(u * tex_width);
    int y = (int)(v * tex_height);
    if (!texture || x<0 || y<0 || x>=tex_width || y>=tex_height) {
		return Color_cuda();
	}
    Color_cuda rst;
    int st = (x+y*tex_width)*tex_bytespp;
    for (int i = 0; i < tex_bytespp; ++i) {
        rst.raw[i] = texture[st+i];
    }
    return rst;
}

__device__ Color_cuda BlinnPhongShade(Fragment_cuda &frag, unsigned char* texture, int tex_width, int tex_height, int tex_bytespp, Vec3f_cuda* light_color, Vec3f_cuda* light_dir) {
    Color_cuda tex_color = getTextureColor(texture, tex_width, tex_height, tex_bytespp, frag.tex.x, frag.tex.y);
    Vec3f_cuda ka = Vec3f_cuda(0.005, 0.005, 0.005);
    Vec3f_cuda kd = CudaUchar2Float(tex_color).value();
    Vec3f_cuda ks = Vec3f_cuda(0.5, 0.5, 0.5);

    Vec3f_cuda ambient(ka.x*light_color->x, ka.y*light_color->y, ka.z*light_color->z);

    Vec3f_cuda vec_l = Vec3f_cuda(-light_dir->x, -light_dir->y, -light_dir->z);
    float NdotL = vec_l.x * frag.norm.x + vec_l.y * frag.norm.y + vec_l.z * frag.norm.z;
    float diff = fmaxf(0.f, NdotL);
    Vec3f_cuda diffuse(kd.x*light_color->x*diff, kd.y*light_color->y*diff, kd.z*light_color->z*diff);

    float p = 128.f;
    Vec3f_cuda vec_v(-frag.pos_view.x, -frag.pos_view.y, -frag.pos_view.z);
    Vec3f_cuda vec_h = vec3f_add(vec_v, vec_l).normalize();
    float NdotH = frag.norm.x * vec_h.x + frag.norm.y * vec_h.y + frag.norm.z * vec_h.z;
    float spec = fmaxf(0.f, powf(NdotH, p));
    Vec3f_cuda specular(ks.x*light_color->x*spec, ks.y*light_color->y*spec, ks.z*light_color->z*spec);

    Vec3f_cuda rst = vec3f_add(vec3f_add(ambient, diffuse), specular);

    return CudaFloat2Uchar(Vec4f_cuda(rst.x, rst.y, rst.z, 1.0f));
}

__device__ Vec3f_cuda vec3f_add(Vec3f_cuda &v0, Vec3f_cuda &v1) {
    Vec3f_cuda rst;
    rst.x = v0.x + v1.x;
    rst.y = v0.y + v1.y;
    rst.z = v0.z + v1.z;
    return rst;
}

__device__ Vec4f_cuda CudaUchar2Float(Color_cuda c) {
    return Vec4f_cuda(c.r/255., c.g/255., c.b/255., c.a/255.);
}

__device__ Color_cuda CudaFloat2Uchar(Vec4f_cuda f) {
    float x, y, z, w;
    x = fmaxf(0.f, fminf(1.f, f.x));
    y = fmaxf(0.f, fminf(1.f, f.y));
    z = fmaxf(0.f, fminf(1.f, f.z));
    w = fmaxf(0.f, fminf(1.f, f.w));
    return Color_cuda((unsigned char)(x*255+.5f), (unsigned char)(y*255+.5f), (unsigned char)(z*255+.5f), (unsigned char)(w*255+.5f));
}