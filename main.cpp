#include "tgaimage.h"
#include "geometry.h"
#include "model.h"
#include "texture.h"
#include "buffer.h"
#include "transform.h"
#include "shader.h"
#include "light.h"
#include "camera.h"
#include "window.h"
#include <iostream>
#include <vector>
#include <assert.h>

const Color white = Color(255, 255, 255, 255);
const Color red   = Color(255, 0,   0,   255);
const Color green = Color(0  , 255, 0, 255);

const int WIDTH = 800;
const int HEIGHT = 800;

TGAImage image(WIDTH, HEIGHT, TGAImage::RGB);
Buffer z_buffer(WIDTH, HEIGHT, true);
Light light(Vec3f(1.f,1.f,1.f), Vec3f(1.f,-1.f,-1.f));
Window window(WIDTH, HEIGHT);

float edge(Vec2f &p0, Vec2f &p1, Vec2f &p) {
	Vec2f e0 = p1 - p0;
	Vec2f e1 = p - p0;
	return e0.x * e1.y - e1.x * e0.y;
}

Vec3f barycentric(Vec3f *t, Vec2f p) {
	Vec2f t0(t[0].x, t[0].y);
	Vec2f t1(t[1].x, t[1].y);
	Vec2f t2(t[2].x, t[2].y);

	float tri_area = edge(t0, t1, t2);
	if (std::abs(tri_area) < 1e-6)
		return Vec3f(-1.f, -1.f, -1.f);
	float lambda0 = edge(t1, t2, p) / tri_area;
	float lambda1 = edge(t2, t0, p) / tri_area;
	float lambda2 = edge(t0, t1, p) / tri_area;

	return Vec3f(lambda0, lambda1, lambda2);
}

// float persp_interpolate(Vec3f depth, Vec3f lambda) {
// 	return 1. / (lambda[0] / depth[0] + lambda[1] / depth[1] + lambda[2] / depth[2]);
// }

void check_division(Vec3f &depth) {
	for (int i = 0; i < 3; i++) {
		if (depth[i] < 1e-6) {
			depth[i] += 1e-6;
		}
	}
}

float depth_persp_interpolate(Vec3f depth, Vec3f lambda) {
	check_division(depth);
	return 1. / (lambda[0] / (depth[0]) + lambda[1] / (depth[1]) + lambda[2] / (depth[2]));
}

Vec2f vec2_persp_interpolate(Vec2f* vec2_arr, float depth, Vec3f depth_arr, Vec3f lambda) {
	check_division(depth_arr);
	Vec2f vec2 = depth*(lambda[0]*vec2_arr[0]/depth_arr[0]+lambda[1]*vec2_arr[1]/depth_arr[1]+lambda[2]*vec2_arr[2]/depth_arr[2]);
	return vec2;
}

Vec3f vec3_persp_interpolate(Vec3f* vec3_arr, float depth, Vec3f depth_arr, Vec3f lambda) {
	check_division(depth_arr);
	Vec3f vec3 = depth*(lambda[0]*vec3_arr[0]/depth_arr[0]+lambda[1]*vec3_arr[1]/depth_arr[1]+lambda[2]*vec3_arr[2]/depth_arr[2]);
	return vec3;
}

void draw_triangle(Triangle &tri, TGAImage &image, Shader &shader) {
	Vec3f *pts = tri.poses;
	Vec2f *texs = tri.texs;
	Vec3f *norms = tri.norms;
	Vec3f *pos_views = tri.poses_view;
	int bboxmin_x = std::max(0, static_cast<int>(std::min(pts[0].x, std::min(pts[1].x, pts[2].x))));
	int bboxmax_x = std::min(WIDTH-1, static_cast<int>(std::max(pts[0].x, std::max(pts[1].x, pts[2].x))));
	int bboxmin_y = std::max(0, static_cast<int>(std::min(pts[0].y, std::min(pts[1].y, pts[2].y))));
	int bboxmax_y = std::min(HEIGHT-1, static_cast<int>(std::max(pts[0].y, std::max(pts[1].y, pts[2].y))));

	for (int x = bboxmin_x; x <= bboxmax_x; ++x) {
		for (int y = bboxmin_y; y <= bboxmax_y; ++y) {
			Vec3f lambda = barycentric(pts, Vec2f(x+.5, y+.5));
			if (lambda[0] >= 0 && lambda[1] >= 0 && lambda[2] >= 0) {	// !!! 等于号 , 否则会有边缘空洞
				float z = 0;	Vec2f tex_coord;	Vec3f norm;	Vec3f pos_view;
				// for (int i = 0; i < 3; i++)		z += pts[i].z * lambda[i];
				// tex_coord = texs[0] * lambda[0] + texs[1] * lambda[1] + texs[2] * lambda[2];
				// norm = norms[0]*lambda[0]+norms[1]*lambda[1]+norms[2]*lambda[2];
				// pos_view = pos_views[0]*lambda[0]+pos_views[1]*lambda[1]+pos_views[2]*lambda[2];
				Vec3f depth_arr(pts[0].z, pts[1].z, pts[2].z);
				z = depth_persp_interpolate(depth_arr, lambda);
				tex_coord = vec2_persp_interpolate(texs, z, depth_arr, lambda);
				norm = vec3_persp_interpolate(norms, z, depth_arr, lambda);
				pos_view = vec3_persp_interpolate(pos_views, z, depth_arr, lambda);
				if (z < z_buffer.get(x, y)) {
					Fragment frag(Vec2i(x,y), tex_coord, norm, pos_view);
					// window.drawPixel(x, y, shader.shadeFragment(frag));
					image.set(x, y, shader.shadeFragment(frag));
					z_buffer.set(x, y, z);
				}
			}
		}
	}
}

#include <chrono>


int main(int argc, char** argv) {
	Model *model = new Model("obj/african_head.obj");
	Texture texture("tex/african_head_diffuse.tga");
	// PhongShader shader(texture, light);
	TextureShader shader(texture);

	// Model *model = new Model("obj/cube.obj");
	// Shader shader;

	Camera camera;
	Transform vShader(WIDTH, HEIGHT, z_buffer.getUpperRange());

	// std::vector<Mesh> meshes = model->getMeshes();
	std::vector<Vert> verts = model->getVerts();
	std::vector<Vert> verts_debug = {verts[6302], verts[6303], verts[6301]};

	assert(verts.size() % 3 == 0);
	// window.init();

	// while (!Window::screenExit && Window::screenKeys[VK_ESCAPE] == 0) {
	// 	z_buffer.clearBuffer();
	// 	window.clearScreen();

	// 	window.pollMessage();
	// 	camera.processInput();
	// 	for (int i = 0; i < meshes.size(); i++) {
	// 		std::vector<Triangle> triangles = vShader.transform(meshes[i], camera);
	// 		for (int j = 0; j < triangles.size(); ++j) 
	// 			draw_triangle(triangles[j], image, shader);
	// 	}
	// 	window.updateScreen();
	// } 
	z_buffer.clearBuffer();

	std::vector<Triangle> triangles;
	
	// vShader.test();
	// std::cout << sizeof(Vert) << std::endl;
	// exit(-1);

	auto start = std::chrono::high_resolution_clock::now();
	triangles = vShader.transform(verts);
	std::cout << triangles.size() << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "cpu: " << duration.count() << " milliseconds" << std::endl;

	// auto start = std::chrono::high_resolution_clock::now();
	// triangles = vShader.transformCuda(verts);	
	// std::cout << triangles.size() << std::endl;
	// auto end = std::chrono::high_resolution_clock::now();
	// auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "cuda: " << duration.count() << " milliseconds" << std::endl;

	// exit(-1);
	// for (int i = 0; i < meshes.size(); ++i) {
	// 	std::vector<Triangle> mesh_tri = vShader.transform(meshes[i], camera);
	// 	triangles.insert(triangles.end(), mesh_tri.begin(), mesh_tri.end());
	// }

	// start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < triangles.size(); ++i) {
		draw_triangle(triangles[i], image, shader);
	}

	// end = std::chrono::high_resolution_clock::now();
	// duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "fragment: " << duration.count() << " milliseconds" << std::endl;

	// std::cout << sizeof(Mesh) << std::endl;
	// for (int i = 0; i < meshes.size(); i++) {
	// 	std::vector<Triangle> triangles = vShader.transform(meshes[i], camera);
	// 	for (int j = 0; j < triangles.size(); ++j) 
	// 		draw_triangle(triangles[j], image, shader);
	// }

	image.write_tga_file("output.tga");

	return 0;
}

