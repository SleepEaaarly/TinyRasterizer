#ifndef LIGHT_H
#define LIGHT_H

#include "geometry.h"

struct Light {
    Vec3f color;
    Vec3f dir;

    Light() {}
    Light(Vec3f c, Vec3f d) : color(c), dir(d) {}
};

#endif