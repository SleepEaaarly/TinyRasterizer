#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "model.h"

Model::Model(const char *filename) : verts_(), texs_(), norms_(), faces_() {
    std::ifstream in;
    in.open (filename, std::ifstream::in);
    if (in.fail()) return;
    std::string line;
    while (std::getline(in, line)) {
        // std::getline(in, line);
        std::istringstream iss(line.c_str());
        char trash;
        // std::cout << line << std::endl;
        if (!line.compare(0, 2, "v ")) {
            iss >> trash;
            Vec3f v;
            for (int i=0;i<3;i++) iss >> v[i];
            verts_.push_back(v);
        } else if (!line.compare(0, 2, "f ")) {
            std::vector<Vec3i> f;
            int vidx, tidx, nidx;
            iss >> trash;
            while (iss >> vidx >> trash >> tidx >> trash >> nidx) {
                vidx--; // in wavefront obj all indices start at 1, not zero
                tidx--;
                nidx--;
                f.push_back(Vec3i(vidx, tidx, nidx));
            }
            faces_.push_back(f);
        } else if (!line.compare(0, 3, "vt ")) {
            iss >> trash >> trash;
            Vec2f vt;
            iss >> vt[0] >> vt[1];
            texs_.push_back(vt);
        } else if (!line.compare(0, 3, "vn ")) {
            iss >> trash >> trash;
            Vec3f vn;
            iss >> vn[0] >> vn[1] >> vn[2];
            norms_.push_back(vn);
        }
    }
    std::cerr << "# v# " << verts_.size() << " vt# " << texs_.size() << " vn# " << norms_.size() << " f# "  << faces_.size() << std::endl;
}

Model::~Model() {
}

int Model::nverts() {
    return (int)verts_.size();
}

int Model::ntexs() {
    return (int)texs_.size();
}

int Model::nnorms() {
    return (int)norms_.size();
}

int Model::nfaces() {
    return (int)faces_.size();
}

std::vector<int> Model::faceVertIdx(int idx) {
    std::vector<int> poses{faces_[idx][0][0], faces_[idx][1][0], faces_[idx][2][0]};
    return poses;
}

std::vector<int> Model::faceTexIdx(int idx) {
    std::vector<int> texs{faces_[idx][0][1], faces_[idx][1][1], faces_[idx][2][1]};
    return texs;
}

std::vector<int> Model::faceNormIdx(int idx) {
    std::vector<int> norms{faces_[idx][0][2], faces_[idx][1][2], faces_[idx][2][2]};
    return norms;
}

std::vector<Vec3i> Model::faceIdx(int idx) {
    return faces_[idx];
}

Vec3f Model::vert(int i) {
    return verts_[i];
}

Vec2f Model::tex(int i) {
    return texs_[i];
}

Vec3f Model::norm(int i) {
    return norms_[i];
}

std::vector<Mesh> Model::getMeshes() {
    if (meshes.empty()) {       // parse meshes
        for (int i=0; i<nfaces(); i++) {
            Mesh mesh;
            // std::cout << "1" << std::endl;
            for (int j=0; j<3; j++) {
                mesh.poses[j] = vert(faceVertIdx(i)[j]);        // 第i个face的第j(j=0,1,2)个点的数据
                mesh.texs[j] = tex(faceTexIdx(i)[j]);
                mesh.norms[j] = norm(faceNormIdx(i)[j]);
            }
            meshes.push_back(mesh);
        }
    }
    return meshes;
}

std::vector<Vert> Model::getVerts() {
    if (verts.empty()) {
        for (int i=0; i<nfaces(); i++) {
            for (int j=0; j<3; j++) {
                Vert _vert;
                _vert.pos = vert(faceVertIdx(i)[j]);
                _vert.tex = tex(faceTexIdx(i)[j]);
                _vert.norm = norm(faceNormIdx(i)[j]);
                verts.push_back(_vert);
            }
        }
    }
    return verts;
}