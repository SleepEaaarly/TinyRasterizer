#include "transform.h"
#include "transform_cuda.cuh"
#include "debug.h"
#include <chrono>

Transform::Transform(int w, int h, int d) {
    screen_width = w;
	screen_height = h;
	depth_upper = d;
    init();
}

void Transform::init() {
    model = Matrix::identity();
	model_inv = model.invert();
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

std::vector<Triangle> Transform::transform(Mesh &mesh) {
	Vert vert[3];
	Matrix m_view_inv = view.transpose();
	for (int i = 0; i < 3; ++i) {
		Vec4f pos_view = view * model * Vec4f(mesh.poses[i]);
		vert[i].pos_view = pos_view.value();
		vert[i].pos = persp * pos_view;
		vert[i].norm = ((model_inv*m_view_inv).transpose()*Vec4f(mesh.norms[i], 0.0f)).value();
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
		multi_verts[i].pos = vp*(multi_verts[i].pos.clip());
	}

	for (int i = 1; i < multi_verts.size()-1; ++i) {
		Triangle tri = toTriangle(multi_verts[0], multi_verts[i], multi_verts[i+1]);
		triangles.push_back(tri);
	}

	return triangles;
}

std::vector<Triangle> Transform::transform(Mesh &mesh, Camera &camera) {
	updateViewMatrix(camera);
	return transform(mesh);
}

std::vector<Triangle> Transform::transform(std::vector<Vert> &verts) {
	std::vector<Triangle> triangles;
	Matrix model_view = view * model;
	Matrix model_view_inv_trans = (model_inv * (view.transpose())).transpose();
	Matrix model_view_persp = persp * model_view;
	for (int i = 0; i < verts.size() / 3; ++i) {
	 	std::vector<Triangle> tri = transform(verts[3*i], verts[3*i+1], verts[3*i+2], model_view, model_view_inv_trans, model_view_persp);
	 	triangles.insert(triangles.end(), tri.begin(), tri.end());
	}
	return triangles;
}

std::vector<Triangle> Transform::transform(std::vector<Vert> &verts, Camera &camera) {
	updateViewMatrix(camera);
	return transform(verts);
}

Vert Transform::transformVert(Vert &vert, Matrix &model_view, Matrix &model_view_inv_trans, Matrix &model_view_persp) {
	Vert v;
	v.pos_view = (model_view * vert.pos).value();
	v.pos = model_view_persp * vert.pos;
	v.norm = (model_view_inv_trans * Vec4f(vert.norm, 0.0f)).value();
	v.tex = vert.tex;
	return v;
}

std::vector<Triangle> Transform::transform(Vert &vert0, Vert &vert1, Vert &vert2, Matrix &model_view, Matrix &model_view_inv_trans, Matrix &model_view_persp) {
	Vert vert[3];
	vert[0] = transformVert(vert0, model_view, model_view_inv_trans, model_view_persp);
	vert[1] = transformVert(vert1, model_view, model_view_inv_trans, model_view_persp);
	vert[2] = transformVert(vert2, model_view, model_view_inv_trans, model_view_persp);

	// for (int i = 0; i < 3; i++) {
	// 	std::cout << vert[i].pos.x << ", " << vert[i].pos.y << ", " << vert[i].pos.z << ", " << vert[i].pos.w << std::endl;
	// }
	// std::vector<Vert> multi_verts = sutherlandHodgeman(vert[0], vert[1], vert[2]);
	std::vector<Vert> multi_verts = {vert[0], vert[1], vert[2]};

	std::vector<Triangle> triangles;
	if (multi_verts.size() < 3) {
		return triangles;
	}

	// clip and viewport transform
	for (int i = 0; i < multi_verts.size(); ++i) {
		multi_verts[i].pos = vp*(multi_verts[i].pos.clip());
	}

	// outputVert(multi_verts[0]);
	// outputVert(multi_verts[1]);
	// outputVert(multi_verts[2]);

	for (int i = 1; i < multi_verts.size()-1; ++i) {
		Triangle tri = toTriangle(multi_verts[0], multi_verts[i], multi_verts[i+1]);
		triangles.push_back(tri);
	}

	return triangles;
}

void Transform::transformCuda(std::vector<Vert> &verts, Camera &camera) {
	updateViewMatrix(camera);
	cudaUpdateMatrix();
	transformCuda(verts);
}

void Transform::transformCuda(std::vector<Vert> &verts) {
	dim3 grid_dim((num_verts-1)/128+1, 1, 1);
	dim3 block_dim(128, 1, 1);

	auto start = std::chrono::high_resolution_clock::now();
	transformObjectToScreenKernal<<<grid_dim, block_dim>>>(
		(Vert_cuda*)d_verts, d_model_view, d_model_view_inv_trans, d_model_view_persp, d_vp, (Vert_cuda*)d_verts_rst, num_verts
	);
	cudaDeviceSynchronize();
	
	cudaMemcpy(verts_rst, d_verts_rst, num_verts_rst * sizeof(Vert), cudaMemcpyDeviceToHost);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "vertex kernel function: " << duration.count() << " milliseconds" << std::endl;

}

void Transform::cudaInit(std::vector<Vert> &verts) {
	num_verts = verts.size();
	num_verts_rst = num_verts;

	Matrix model_view = view * model;
	Matrix model_view_inv_trans = (model_inv * (view.transpose())).transpose();
	Matrix model_view_persp = persp * model_view;

	verts_rst = new Vert[num_verts_rst];

	cudaMalloc((void**)&d_verts, num_verts * sizeof(Vert));
	cudaMalloc((void**)&d_model_view, sizeof(float) * 16);
	cudaMalloc((void**)&d_model_view_inv_trans, sizeof(float) * 16);
    cudaMalloc((void**)&d_model_view_persp, sizeof(float) * 16);
	cudaMalloc((void**)&d_vp, sizeof(float) * 16);
	cudaMalloc((void**)&d_verts_rst, num_verts_rst * sizeof(Vert));

	cudaMemcpy(d_verts, verts.data(), num_verts * sizeof(Vert), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_view, model_view.get_ptr(), sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_view_inv_trans, model_view_inv_trans.get_ptr(), sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_view_persp, model_view_persp.get_ptr(), sizeof(float) * 16, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vp, vp.get_ptr(), sizeof(float) * 16, cudaMemcpyHostToDevice);

}

void Transform::cudaUpdateMatrix() {
	Matrix model_view = view * model;
	Matrix model_view_inv_trans = (model_inv * (view.transpose())).transpose();
	Matrix model_view_persp = persp * model_view;

	cudaMemcpy(d_model_view, model_view.get_ptr(), sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_view_inv_trans, model_view_inv_trans.get_ptr(), sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_view_persp, model_view_persp.get_ptr(), sizeof(float) * 16, cudaMemcpyHostToDevice);
}

void Transform::cudaRelease() {
	cudaFree(d_verts);
	cudaFree(d_model_view);
	cudaFree(d_model_view_inv_trans);	
	cudaFree(d_model_view_persp);
	cudaFree(d_vp);
	cudaFree(d_verts_rst);
	delete[] verts_rst;
}

void Transform::updateViewMatrix(Camera &camera) {
	Matrix cam_view = lookAt(camera.position, camera.position+camera.front, camera.worldUp);
	this->view = cam_view;
}

Vert* Transform::getDeviceVertsRstPtr() {
	return d_verts_rst;
}

int Transform::getDeviceVertsRstNum() {
	return num_verts_rst;
}

void Transform::test() {
	Vert verts[2];
	verts[0].pos = Vec4f(1.f, 2.f, 3.f, 4.f);
	verts[1].pos = Vec4f(5.f, 6.f, 7.f, 8.f);
	Vert_cuda* d_vert;

	cudaMalloc((void**)&d_vert, 2 * sizeof(Vert));
	cudaMemcpy(d_vert, verts, 2 * sizeof(Vert), cudaMemcpyHostToDevice);
	
	dim3 grid_dim(1, 1, 1);
	dim3 block_dim(32, 1, 1);

	test_cuda<<<grid_dim, block_dim>>>(d_vert);

	cudaFree(d_vert);
}
