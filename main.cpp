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

const Color white = Color(255, 255, 255, 255);
const Color red   = Color(255, 0,   0,   255);
const Color green = Color(0  , 255, 0, 255);

const int WIDTH = 800;
const int HEIGHT = 800;

TGAImage image(WIDTH, HEIGHT, TGAImage::RGB);
Buffer z_buffer(WIDTH, HEIGHT, true);
Light light(Vec3f(1.f,1.f,1.f), Vec3f(1.f,-1.f,-1.f));
Window window(WIDTH, HEIGHT);

bool check_edge(int x, int y, int width, int height) {
	if (x < 0 || x >= width || y < 0 || y >= height)
		return false;
	return true;
}

void draw_line(int x0, int y0, int x1, int y1, TGAImage &image, Color c) {
	// 单点
	if (x0 == x1 && y0 == y1) {
		image.set(x0, y0, c);
		return;
	}
	
	// 同行或同列
	if (x0 == x1) {
		if (y0 > y1)	std::swap(y0, y1);
		for (int y = y0; y <= y1; ++y) {
			image.set(x0, y, c);
		}
		return;
	}
	else if (y0 == y1) {
		if (x0 > x1)	std::swap(x0, x1);
		for (int x = x0; x <= x1; ++x) {
			image.set(x, y0, c);
		}
		return;
	}

	// 普通情况
	bool trans = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
		std::swap(x0, y0);
		std::swap(x1, y1);
		trans = true;
	}
	if (x0 > x1) {					// from left to right
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	int dx = x1 - x0;
	int dy = y1 - y0;
	float ddy = std::abs(dy/float(dx));
	float inc = 0;
	int y = y0;
	int width = image.get_width();
	int height = image.get_height();

	for (int x = x0; x <= x1; x++) {
		if (trans) {
			if (check_edge(y, x, width, height))
				image.set(y, x, c);
		} else {
			if (check_edge(x, y, width, height))
				image.set(x, y, c);
		}
		inc += ddy;
		if (inc > .5) {
			y += dy > 0 ? 1 : -1;
			inc -= 1.f;
		}
	}
}

float edge(Vec2f p0, Vec2f p1, Vec2f p) {
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
					window.drawPixel(x, y, shader.shadeFragment(frag));
					// image.set(x, y, shader.shadeFragment(frag));
					z_buffer.set(x, y, z);
				}
			}
		}
	}
}


int main(int argc, char** argv) {
	Model *model = new Model("obj/african_head.obj");
	Texture texture("tex/african_head_diffuse.tga");
	// PhongShader shader(texture, light);
	TextureShader shader(texture);

	// Model *model = new Model("obj/cube.obj");
	// Shader shader;

	Camera camera;
	Transform vShader(WIDTH, HEIGHT, z_buffer.getUpperRange());

	std::vector<Mesh> meshes = model->getMeshes();

	// window.init();

	while (!Window::screenExit && Window::screenKeys[VK_ESCAPE] == 0) {
		z_buffer.clearBuffer();
		window.clearScreen();

		window.pollMessage();
		camera.processInput();
		for (int i=0; i<meshes.size(); i++) {
			// Triangle tri = vShader.transform(meshes[i], camera);
			// std::cout << "mesh.face: " << i << std::endl;
			std::vector<Triangle> triangles = vShader.transform(meshes[i], camera);
			for (int j = 0; j < triangles.size(); ++j) 
				draw_triangle(triangles[j], image, shader);
			// draw_triangle(tri, image, shader);
		}
		window.updateScreen();
		// std::cout << "1" << std::endl;
	} 

	// image.flip_vertically();
	// image.write_tga_file("output.tga");

	return 0;
}

// #include <iostream>

// void outputVector(Vec3f v) {
// 	for (int i = 0; i < 3; i++) {
// 		std::cout << v[i] << " ";
// 	}
// 	std::cout << std::endl;
// }

// void outputMatrix(Matrix m) {
// 	for (int i = 0; i < 4; i++) {
// 		for (int j = 0; j < 4; j++) {
// 			std::cout << m[i][j] << " ";
// 		}
// 		std::cout << std::endl;
// 	}
// 	std::cout << std::endl;
// }