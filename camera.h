#ifndef CAMERA_H
#define CAMERA_H

#include "geometry.h"

class Camera
{
private:
    /* data */
public:
    Vec3f position;
    Vec3f front;
    Vec3f up;
    Vec3f right;
    Vec3f worldUp;
    float yaw;      // rotation around y-axis
    float pitch;    // rotation around x-axis

    Camera(Vec3f pos = Vec3f(0.f, 0.f, 3.f), Vec3f front = Vec3f(0.f, 0.f, -1.f), Vec3f up = Vec3f(0.f, 1.f, 0.f), float yaw = 180.f, float pitch = 0.f);
    void updateVector();
    void processInput();
    void setPos(Vec3f &pos);
    void setFrontVector(Vec3f &front_vec);

};
#endif