#include "camera.h"
#include <cmath>
#include "window.h"
#define  PI 3.1415926f
float deg2rad(float rad) {
    return PI * rad / 180.f;
}

Camera::Camera(Vec3f pos, Vec3f front, Vec3f up, float yaw, float pitch)
{
    this->position = pos;
    this->front = front.normalize();
    this->worldUp = up.normalize();    
    this->right = cross(this->worldUp, -1.f*this->front);
    this->up = cross(-1.f*this->front, this->right);
    this->yaw = yaw;
    this->pitch = pitch;
    updateVector();
}

void Camera::updateVector() {
    float x = std::cos(deg2rad(pitch)) * std::sin(deg2rad(yaw));
    float y = std::sin(deg2rad(pitch));
    float z = std::cos(deg2rad(pitch)) * std::cos(deg2rad(yaw));

    // std::cout << yaw << ", " << pitch << std::endl;
    // std::cout << x << ", " << y << ", " << z << std::endl;

    this->front = Vec3f(x, y, z).normalize();
    this->right = cross(worldUp, -1.f*front).normalize();
    this->up = cross(-1.f*front, right).normalize();    

}

void Camera::setPos(Vec3f &pos) {
    position = pos;
}

void Camera::setFrontVector(Vec3f &front_vec) {
    front = front_vec;
}

void Camera::processInput() {
    if (Window::screenKeys[VK_LEFT]) {
        position = position - .2f * right;
    }
    if (Window::screenKeys[VK_RIGHT]) {
        position = position + .2f * right;
    }
    if (Window::screenKeys[VK_UP]) {
        position = position + .2f * front;
    }
    if (Window::screenKeys[VK_DOWN]) {
        position = position - .2f * front;
    }

    if (Window::screenKeys['A']) {
        // std::cout << "A" << std::endl;
        yaw = yaw + 5.f;
    }
    if (Window::screenKeys['D']) {
        yaw = yaw - 5.f;
    }
    if (Window::screenKeys['W']) {
        pitch = pitch + 5.f;
    }
    if (Window::screenKeys['S']) {
        pitch = pitch - 5.f;
    }
    
    if (pitch > 89.f) {
        pitch = 89.f;
    } 
    if (pitch < -89.f) {
        pitch = -89.f;
    }

    updateVector();
    // std::cout << front.x << ", " << front.y << ", " << front.z << std::endl;
    // std::cout << position.x << ", " << position.y << ", " << position.z << std::endl;
}