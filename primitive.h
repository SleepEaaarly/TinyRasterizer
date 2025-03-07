#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "geometry.h"

struct Mesh {
    Vec3f poses[3];
    Vec2f texs[3];
    Vec3f norms[3];
};

struct Triangle {
    Vec3f poses[3];
    Vec2f texs[3];
    Vec3f norms[3];
    Vec3f poses_view[3];
};

struct Fragment {
    Vec2i pos;
    Vec2f tex;
    Vec3f norm;
    Vec3f pos_view;
    Fragment(){}
    Fragment(Vec2i pos, Vec2f tex, Vec3f norm, Vec3f pos_view) : pos(pos), tex(tex), norm(norm), pos_view(pos_view) {}
};

struct Vert {
    Vec4f pos;
    Vec3f norm;
    Vec2f tex;
    Vec3f pos_view;

    Vert() {}
    Vert(Vec4f p, Vec3f n, Vec2f t, Vec3f p_v) : pos(p), norm(n), tex(t), pos_view(p_v) {}

    friend Vert operator+(const Vert &lhs, const Vert &rhs);
    friend Vert operator*(float t, const Vert &rhs);
    friend Vert operator*(const Vert &lhs, float t);
    friend Triangle toTriangle(Vert &v0, Vert &v1, Vert &v2);
};

#endif
