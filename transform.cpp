#include "transform.h"

Transform::Transform(int w, int h, int d) {
    screen_width = w;
	screen_height = h;
	depth_upper = d;
    init();
}

void Transform::init() {
    model = Matrix::identity();
    view = lookAt(Vec3f(0.f, 0.f, 3.f), Vec3f(0.f, 0.f, 0.f), Vec3f(0.f, 1.f, 0.f));
    persp = perspective(45.f, 1.f, 0.1f, 600.f);
    vp = viewport(0, 0, screen_width, screen_height);
	clipPlanes = {
		// near
		Vec4f(0.f, 0.f, 1.f, 1.f),
		// far
		Vec4f(0.f, 0.f, -1.f, 1.f)
		// xy方向由于rasterization部分采用了AABB所以可以不裁剪
		// left
		// Vec4f(1.f, 0.f, 0.f, 1.f),
		// // right
		// Vec4f(-1.f, 0.f, 0.f, 1.f),
		// // top
		// Vec4f(0.f, -1.f, 0.f, 1.f),
		// // bottom
		// Vec4f(0.f, 1.f, 0.f, 1.f)
	};
}

Matrix Transform::lookAt(Vec3f eye, Vec3f center, Vec3f up) {
	Vec3f z_cam = (eye - center).normalize();
	Vec3f x_cam = cross(up, z_cam);
	Vec3f y_cam = cross(z_cam, x_cam);

	Matrix mat;
	mat[0][0] = x_cam[0];	mat[0][1] = x_cam[1];	mat[0][2] = x_cam[2];	mat[0][3] = 0. - eye * x_cam;
	mat[1][0] = y_cam[0];	mat[1][1] = y_cam[1];	mat[1][2] = y_cam[2];	mat[1][3] = 0. - eye * y_cam;
	mat[2][0] = z_cam[0];	mat[2][1] = z_cam[1];	mat[2][2] = z_cam[2];	mat[2][3] = 0. - eye * z_cam;
	mat[3][0] = 0.;			mat[3][1] = 0.;			mat[3][2] = 0.;			mat[3][3] = 1.;

	return mat;
}

Matrix Transform::perspective(float fovY, float aspect, float near, float far) {
	// 推导 先调转z轴！(要求最后NDC为左手系)，然后透视再平移缩放比例
	Matrix mat;
	mat[0][0] = 1./(tanf(fovY / 2.)*aspect); mat[0][1] = 0.; mat[0][2] = 0.; mat[0][3] = 0.;
	mat[1][0] = 0.;	mat[1][1] = 1./tanf(fovY/2.); mat[1][2] = 0.; mat[1][3] = 0.;
	mat[2][0] = 0.; mat[2][1] = 0.; mat[2][2] = -(near+far)/(far-near); mat[2][3] = -2.*near*far/(far-near);
	mat[3][0] = 0.;	mat[3][1] = 0.;	mat[3][2] = -1.; mat[3][3] = 0.;

	return mat;
}


Matrix Transform::viewport(int x, int y, int w, int h) {
	// 有两个操作，一是反转y轴使其变为左上角原点，二是保证满足opengl的viewport操作
	Matrix mat;
	mat[0][0] = w/2.f;	mat[0][1] = 0.f;	mat[0][2] = 0.f;	mat[0][3] = x+w/2.f;
	mat[1][0] = 0.f;	mat[1][1] = -h/2.f;	mat[1][2] = 0.f;	mat[1][3] = -y+h/2.f;
	mat[2][0] = 0.f;	mat[2][1] = 0.f;	mat[2][2] = depth_upper/2.f;	mat[2][3] = depth_upper/2.f;
	mat[3][0] = 0.f;			mat[3][1] = 0.f;			mat[3][2] = 0.f;			mat[3][3] = 1.f;
	
	return mat;
}

bool Transform::insideWayPlane(Vert &v, Vec4f &plane) {
	if (v.pos * plane >= 1e-6f) {
		return true;
	}
	return false;
}

bool Transform::allInsideClipCube(std::vector<Vert> &verts) {
	for (int i = 0; i < verts.size(); ++i) {
		for (int j = 0; j < clipPlanes.size(); ++j) {
			if (!insideWayPlane(verts[i], clipPlanes[j])) {
				return false;
			}
		}
	}
	return true;
}

Vert Transform::intersect(Vert &v0, Vert &v1, Vec4f &plane) {
	float dist0 = std::abs(v0.pos * plane);
	float dist1 = std::abs(v1.pos * plane);

	float t = dist0 / (dist0 + dist1);
	Vert lerp_vert = v0*(1.f-t) + v1*t;
	return lerp_vert;
}

std::vector<Vert> Transform::sutherlandHodgeman(Vert &v0, Vert &v1, Vert &v2) {
	std::vector<Vert> output = {v0, v1, v2};
	if (allInsideClipCube(output)) {
		return output;
	}
	for (int i = 0; i < clipPlanes.size(); i++) {
		std::vector<Vert> input(output);
		output.clear();
		for (int j = 0; j < input.size(); j++) {
			Vert cur = input[j];
			Vert last = input[(j+input.size()-1)%input.size()];
			if (insideWayPlane(cur, clipPlanes[i])) {
				if (!insideWayPlane(last, clipPlanes[i])) {
					Vert intersectPoint = intersect(last, cur, clipPlanes[i]);
					output.push_back(intersectPoint);
				}
				output.push_back(cur);
			} else if (insideWayPlane(last, clipPlanes[i])) {
				Vert intersectPoint = intersect(last, cur, clipPlanes[i]);
				output.push_back(intersectPoint);
			}
		}
	}
	return output;
}

std::vector<Triangle> Transform::transform(Mesh &mesh, Matrix &m_model, Matrix &m_view, Matrix &m_persp, Matrix &m_vp) {
	Vert vert[3];
	for (int i = 0; i < 3; ++i) {
		Vec4f pos_view = m_view * m_model * Vec4f(mesh.poses[i]);
		vert[i].pos_view = pos_view.value();
		vert[i].pos = m_persp * pos_view;
		vert[i].norm = ((m_view*m_model).invert_transpose()*Vec4f(mesh.norms[i])).value();
		vert[i].tex = mesh.texs[i];
	}

	// for (int i = 0; i < 3; i++) {
	// 	std::cout << vert[i].pos.x << ", " << vert[i].pos.y << ", " << vert[i].pos.z << ", " << vert[i].pos.w << std::endl;
	// }
	std::vector<Vert> multi_verts = sutherlandHodgeman(vert[0], vert[1], vert[2]);

	std::vector<Triangle> triangles;
	if (multi_verts.size() < 3) {
		return triangles;
	}

	// clip and viewport transform
	for (int i = 0; i < multi_verts.size(); ++i) {
		multi_verts[i].pos = m_vp*(multi_verts[i].pos.clip());
	}

	for (int i = 1; i < multi_verts.size()-1; ++i) {
		Triangle tri = toTriangle(multi_verts[0], multi_verts[i], multi_verts[i+1]);
		triangles.push_back(tri);
	}

	return triangles;
}

std::vector<Triangle> Transform::transform(Mesh &mesh) {
	return transform(mesh, model, view, persp, vp);
}

std::vector<Triangle> Transform::transform(Mesh &mesh, Camera &camera) {
	Matrix cam_view = lookAt(camera.position, camera.position+camera.front, camera.worldUp);
	return transform(mesh, model, cam_view, persp, vp);
}
