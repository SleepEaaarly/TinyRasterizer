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
	std::vector<Vec4f> clipPlanes;

	void init();
	bool insideWayPlane(Vert &v, Vec4f &plane);
	std::vector<Vert> sutherlandHodgeman(Vert &v0, Vert &v1, Vert &v2);
	bool allInsideClipCube(std::vector<Vert> &verts);
	Vert intersect(Vert &v0, Vert &v1, Vec4f &plane);
	std::vector<Triangle> transform(Mesh &mesh, Matrix &m_model, Matrix &m_view, Matrix &m_persp, Matrix &m_vp);

public:

	Transform(int w, int h, int d);

    Matrix lookAt(Vec3f eye, Vec3f center, Vec3f up = Vec3f(0., 1., 0.));
	Matrix perspective(float fovY, float aspect, float near, float far);
	Matrix viewport(int x, int y, int w, int h);
	std::vector<Triangle> transform(Mesh &mesh);
	std::vector<Triangle> transform(Mesh &mesh, Camera &camera);
};

#endif