#include "image.h"
#include "geometry.h"
#include "model.h"
#include "texture.h"
#include "buffer.h"
#include "transform.h"
#include "shader.h"
#include "light.h"
#include "camera.h"
#include "window.h"
#include "rasterizer.h"
#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <sstream>
#define MY_PI 3.14159f

const Color white = Color(255, 255, 255, 255);
const Color red   = Color(255, 0,   0,   255);
const Color green = Color(0  , 255, 0, 255);

const int WIDTH = 1920;
const int HEIGHT = 2560;

Image image(WIDTH, HEIGHT, Image::RGB);
Buffer z_buffer(WIDTH, HEIGHT, true);
Light light(Vec3f(1.f,1.f,1.f), Vec3f(1.f,-1.f,-1.f));
Window window(WIDTH, HEIGHT);

#include <chrono>


int main(int argc, char** argv) {
	// Model *model = new Model("obj/african_head.obj");
	Model *model = new Model("obj/Marry.obj");
	// Texture texture("tex/african_head_diffuse.tga");
	Texture texture("tex/MC003_Kozakura_Mari.png");
	PhongShader shader(texture, light);
	// TextureShader shader(texture);

	Camera camera;
	Transform vShader(WIDTH, HEIGHT, z_buffer.getUpperRange());
	Rasterizer rasterizer(shader, z_buffer, image);
	// std::vector<Mesh> meshes = model->getMeshes();
	std::vector<Vert> verts = model->getVerts();
	if (verts.size() == 0) {
		image.write_png_file("output.png");
		return 0;
	}

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
	std::vector<Triangle> triangles;
	int camera_num = 12;
	float radius = 3.0f;

	vShader.cudaInit(verts);
	rasterizer.cudaInit(vShader.getDeviceVertsRstPtr(), vShader.getDeviceVertsRstNum());

	auto loop_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < camera_num; ++i) {
		float angle = 2 * MY_PI * i / camera_num;
		Vec3f pos = Vec3f(radius*std::sinf(angle), 0.f, radius*std::cosf(angle));
		Vec3f dir = Vec3f(-std::sinf(angle), 0.f, -std::cosf(angle));
		camera.setPos(pos);
		camera.setFrontVector(dir);

		auto start = std::chrono::high_resolution_clock::now();
		vShader.transformCuda(verts, camera);	
		// triangles = vShader.transform(verts, camera);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "vertex: " << duration.count() << " milliseconds" << std::endl;
		
		start = std::chrono::high_resolution_clock::now();
		rasterizer.rasterizeVertsCuda();
		// z_buffer.clearBuffer();
		// image.clear();
		// rasterizer.rasterizeTriangles(triangles);
	
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "fragment: " << duration.count() << " milliseconds" << std::endl;

		std::ostringstream oss;
		oss << "rst/output" << i << ".png";
		image.write_png_file(oss.str().c_str());
	}
	auto loop_end = std::chrono::high_resolution_clock::now();
	auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start);
	std::cout << "Total time: " << loop_duration.count() << " milliseconds" << std::endl;

	vShader.cudaRelease();
	rasterizer.cudaRelease();

	// image.write_png_file("output.png");

	return 0;
}

