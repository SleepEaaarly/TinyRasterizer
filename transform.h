#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "geometry.h"
#include "primitive.h"
#include "camera.h"

extern const int WIDTH;
extern const int HEIGHT;
extern const float UPPER;

class Transform
{
private:
	int screen_width;
	int screen_height;
	float depth_upper;

	Matrix model;
	Matrix view;
	Matrix persp;
	Matrix vp;
	Matrix model_inv;
	std::vector<Vec4f> clipPlanes;

	// cuda device pointers
	Vert* d_verts;
	float* d_model_view;
    float* d_model_view_inv_trans;
    float* d_model_view_persp;
	float* d_vp;
	Vert* d_verts_rst;	
	// need to free pointers
	Vert* verts_rst;
	// other properties
	int num_verts;
	int num_verts_rst;


	void init();
	bool insideWayPlane(Vert &v, Vec4f &plane);
	std::vector<Vert> sutherlandHodgeman(Vert &v0, Vert &v1, Vert &v2);
	bool allInsideClipCube(std::vector<Vert> &verts);
	Vert intersect(Vert &v0, Vert &v1, Vec4f &plane);
	Vert transformVert(Vert &vert, Matrix &model_view, Matrix &model_view_inv_trans, Matrix &model_view_persp);
	std::vector<Triangle> transform(Vert &vert0, Vert &vert1, Vert &vert2, Matrix &model_view, Matrix &model_view_inv_trans, Matrix &model_view_persp);

	void updateViewMatrix(Camera &camera);
	void cudaUpdateMatrix();

public:

	Transform(int w, int h, int d);

    Matrix lookAt(Vec3f eye, Vec3f center, Vec3f up = Vec3f(0., 1., 0.));
	Matrix perspective(float fovY, float aspect, float near, float far);
	Matrix viewport(int x, int y, int w, int h);
	std::vector<Triangle> transform(Mesh &mesh);
	std::vector<Triangle> transform(Mesh &mesh, Camera &camera);
	std::vector<Triangle> transform(std::vector<Vert> &verts);
	std::vector<Triangle> transform(std::vector<Vert> &verts, Camera &camera);
	std::vector<Triangle> transformCuda(std::vector<Vert> &verts);
	std::vector<Triangle> transformCuda(std::vector<Vert> &verts, Camera &camera);
	Vert* getDeviceVertsRstPtr();
	int getDeviceVertsRstNum();
	void cudaInit(std::vector<Vert> &verts);
	void cudaRelease();
	void test();
};

#endif