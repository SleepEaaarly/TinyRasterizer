#ifndef BUFFER_H
#define BUFFER_H

#include <limits>

const float UPPER = 255.f;

class Buffer
{
private:
    int width;
    int height;
    float *buf;
    bool limit;
    float upper_range;

    void set_limit_value();

public:
    Buffer();
    Buffer(int w, int h);
    Buffer(int w, int h, bool l);
    ~Buffer();
    float get(int x, int y);
    void set(int x, int y, float v);
    float getUpperRange();
    void clearBuffer();
    float* getBufPtr();
    int getWidth();
    int getHeight();
};

#endif