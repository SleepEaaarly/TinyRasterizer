#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "geometry.h"
#include "primitive.h"

class Model {
private:
	std::vector<Vec3f> verts_;
	std::vector<Vec2f> texs_;
	std::vector<Vec3f> norms_;
	std::vector<std::vector<Vec3i> > faces_;

	std::vector<Mesh> meshes;

public:
	Model(const char *filename);
	~Model();
	int nverts();
	int ntexs();
	int nfaces();
	int nnorms();

	Vec3f vert(int i);
	Vec2f tex(int i);
	Vec3f norm(int i);
	std::vector<Vec3i> faceIdx(int idx);
	std::vector<int> faceVertIdx(int idx);
	std::vector<int> faceTexIdx(int idx);
	std::vector<int> faceNormIdx(int idx);

	std::vector<Mesh> getMeshes();
};

#endif //__MODEL_H__
