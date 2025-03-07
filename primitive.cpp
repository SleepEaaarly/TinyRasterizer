#include "primitive.h"

Vert operator+(const Vert &lhs, const Vert &rhs) {
    return Vert(lhs.pos + rhs.pos, lhs.norm + rhs.norm, lhs.tex + rhs.tex, lhs.pos_view + rhs.pos_view);
}

Vert operator*(float t, const Vert &rhs) {
    return Vert(rhs.pos * t, rhs.norm * t, rhs.tex * t, rhs.pos_view * t);
}

Vert operator*(const Vert &lhs, float t) {
    return Vert(lhs.pos * t, lhs.norm * t, lhs.tex * t, lhs.pos_view * t);
}

Triangle toTriangle(Vert &v0, Vert &v1, Vert &v2) {
    Triangle t;
    t.poses[0] = v0.pos.value();
    t.norms[0] = v0.norm;
    t.texs[0] = v0.tex;
    t.poses_view[0] = v0.pos_view;

    t.poses[1] = v1.pos.value();
    t.norms[1] = v1.norm;
    t.texs[1] = v1.tex;
    t.poses_view[1] = v1.pos_view;
    
    t.poses[2] = v2.pos.value();
    t.norms[2] = v2.norm;
    t.texs[2] = v2.tex;
    t.poses_view[2] = v2.pos_view;
    return t;
}