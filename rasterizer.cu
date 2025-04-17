#include "rasterizer.h"
#include "rasterizer_cuda.cuh"
#include <chrono>

Rasterizer::Rasterizer(Shader &shader, Buffer &z_buffer, TGAImage &image) {
    this->shader = &shader;
    this->z_buffer = &z_buffer;
	this->image = &image;
}

void Rasterizer::rasterizeTriangles(std::vector<Triangle> &triangles) {
    for (int i = 0; i < triangles.size(); ++i) {
        drawTriangle(triangles[i]);
    }
}

void Rasterizer::drawTriangle(Triangle &tri) {
    Vec3f *pts = tri.poses;
	Vec2f *texs = tri.texs;
	Vec3f *norms = tri.norms;
	Vec3f *pos_views = tri.poses_view;
    int width = image->get_width();
    int height = image->get_height();
	int bboxmin_x = std::max(0, static_cast<int>(std::min(pts[0].x, std::min(pts[1].x, pts[2].x))));
	int bboxmax_x = std::min(width-1, static_cast<int>(std::max(pts[0].x, std::max(pts[1].x, pts[2].x))));
	int bboxmin_y = std::max(0, static_cast<int>(std::min(pts[0].y, std::min(pts[1].y, pts[2].y))));
	int bboxmax_y = std::min(height-1, static_cast<int>(std::max(pts[0].y, std::max(pts[1].y, pts[2].y))));

	for (int x = bboxmin_x; x <= bboxmax_x; ++x) {
		for (int y = bboxmin_y; y <= bboxmax_y; ++y) {
			Vec3f lambda = barycentric(pts, Vec2f(x+.5, y+.5));
			if (lambda[0] >= 0.f && lambda[1] >= 0.f && lambda[2] >= 0.f) {	// !!! 等于号 , 否则会有边缘空洞
				float z = 0;	Vec2f tex_coord;	Vec3f norm;	Vec3f pos_view;
				Vec3f depth_arr(pts[0].z, pts[1].z, pts[2].z);
				z = depth_persp_interpolate(depth_arr, lambda);
				tex_coord = vec2_persp_interpolate(texs, z, depth_arr, lambda);
				norm = vec3_persp_interpolate(norms, z, depth_arr, lambda);
				pos_view = vec3_persp_interpolate(pos_views, z, depth_arr, lambda);
				if (z < z_buffer->get(x, y)) {
					Fragment frag(Vec2i(x,y), tex_coord, norm, pos_view);
					// window.drawPixel(x, y, shader.shadeFragment(frag));
					image->set(x, y, shader->shadeFragment(frag));
					z_buffer->set(x, y, z);
				}
			}
		}
	}
}

Vec3f Rasterizer::barycentric(Vec3f *t, Vec2f p) {
    Vec2f t0(t[0].x, t[0].y);
    Vec2f t1(t[1].x, t[1].y);
    Vec2f t2(t[2].x, t[2].y);

    Vec3f vec1 = Vec3f(t2.x-t0.x, t1.x-t0.x, t0.x-p.x);
    Vec3f vec2 = Vec3f(t2.y-t0.y, t1.y-t0.y, t0.y-p.y);
    Vec3f u = cross(vec1, vec2);

    if (std::abs(u.z) < 1e-6)	return Vec3f(-1.f,1.f,1.f);
    return Vec3f(1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z);
}

void Rasterizer::zeroCheck(Vec3f &depth) {
	for (int i = 0; i < 3; i++) {
		if (depth[i] < 1e-6) {
			depth[i] = 1e-6;
		}
	}
}

float Rasterizer::depth_persp_interpolate(Vec3f depth, Vec3f &lambda) {
	zeroCheck(depth);
	return 1. / (lambda[0] / (depth[0]) + lambda[1] / (depth[1]) + lambda[2] / (depth[2]));
}

Vec2f Rasterizer::vec2_persp_interpolate(Vec2f* vec2_arr, float depth, Vec3f depth_arr, Vec3f &lambda) {
	zeroCheck(depth_arr);
	Vec2f vec2 = depth*(lambda[0]*vec2_arr[0]/depth_arr[0]+lambda[1]*vec2_arr[1]/depth_arr[1]+lambda[2]*vec2_arr[2]/depth_arr[2]);
	return vec2;
}

Vec3f Rasterizer::vec3_persp_interpolate(Vec3f* vec3_arr, float depth, Vec3f depth_arr, Vec3f &lambda) {
	zeroCheck(depth_arr);
	Vec3f vec3 = depth*(lambda[0]*vec3_arr[0]/depth_arr[0]+lambda[1]*vec3_arr[1]/depth_arr[1]+lambda[2]*vec3_arr[2]/depth_arr[2]);
	return vec3;
}

void Rasterizer::cudaInit(Vert *d_verts_rst_in, int num_verts_rst) {
	d_verts_scr = d_verts_rst_in;
	num_verts_scr = num_verts_rst;

	int image_size = image->get_width() * image->get_height() * image->get_bytespp();
	int buf_size = z_buffer->getWidth() * z_buffer->getHeight();
	TGAImage *texture_image = shader->getTexturePtr()->getImagePtr();
	int texture_size = texture_image->get_width() * texture_image->get_height() * texture_image->get_bytespp();
	cudaMalloc((void**)&d_z_buffer, buf_size*sizeof(float));
	// no need to cpy z_buffer data cause we will init z_buffer in kernel
	cudaMalloc((void**)&d_image, image_size*sizeof(unsigned char));
	// no need to cpy image data cause we will init image in kernel
	cudaMalloc((void**)&d_texture, texture_size*sizeof(unsigned char));
	cudaMemcpy(d_texture, texture_image->buffer(), texture_size*sizeof(unsigned char), cudaMemcpyHostToDevice);

	Light *light = shader->getLightPtr();

	cudaMalloc((void**)&d_light_color, sizeof(Vec3f));
	cudaMalloc((void**)&d_light_dir, sizeof(Vec3f));
	cudaMemcpy(d_light_color, &light->color, sizeof(Vec3f), cudaMemcpyHostToDevice);
	cudaMemcpy(d_light_dir, &light->dir, sizeof(Vec3f), cudaMemcpyHostToDevice);

}

void Rasterizer::cudaUpdateZBuffer() {
	int buf_size = z_buffer->getWidth() * z_buffer->getHeight();
	cudaMemcpy(z_buffer->getBufPtr(), d_z_buffer, buf_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Rasterizer::cudaRelease() {
	// d_verts_rst has been released in Transform
	cudaFree(d_z_buffer);
	cudaFree(d_image);
	cudaFree(d_texture);
}

void Rasterizer::rasterizeVertsCuda() {
	int width = image->get_width();
	int height = image->get_height();
	int bytespp = image->get_bytespp();
	TGAImage *texture_image = shader->getTexturePtr()->getImagePtr();
	int tex_width = texture_image->get_width();
	int tex_height = texture_image->get_height();
	int tex_bytespp = texture_image->get_bytespp();

	dim3 grid_dim((width-1)/16+1, (height-1)/16+1, 1);
	dim3 block_dim(16, 16, 1);
	
	auto start = std::chrono::high_resolution_clock::now();
	rasterizationBlinnPhong<<<grid_dim, block_dim>>>((Vert_cuda*)d_verts_scr, num_verts_scr, d_image, d_z_buffer, width, height, bytespp, 
											d_texture, tex_width, tex_height, tex_bytespp, (Vec3f_cuda*)d_light_color, (Vec3f_cuda*)d_light_dir);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "fragment kernel function in rasterizer: " << duration.count() << " milliseconds" << std::endl;

	int image_size = width * height * bytespp;
	cudaMemcpy(image->buffer(), d_image, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

}