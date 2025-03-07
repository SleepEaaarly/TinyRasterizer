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
    Buffer(int w, int h);
    Buffer(int w, int h, bool l);
    ~Buffer();
    float get(int x, int y);
    void set(int x, int y, float v);
    float getUpperRange();
    void clearBuffer();
};

Buffer::Buffer(int w, int h) : width(w), height(h), limit(false), upper_range(UPPER)
{
    buf = new float[w * h];
}

Buffer::Buffer(int w, int h, bool l) : width(w), height(h), limit(l), upper_range(UPPER)
{
    buf = new float[w * h];
    if (limit)  set_limit_value();
}

Buffer::~Buffer()
{
    delete buf;
}

void Buffer::set_limit_value() {
    for (int i = 0; i < width * height; ++i)
        buf[i] = std::numeric_limits<float>::max();
}

float Buffer::get(int x, int y) {
    if (x<0 || y<0 || x>=width || y>=height) {
		// std::cout << "buffer out of range!" << std::endl;
        return 0.;
	}
    return buf[y*width+x];
}

void Buffer::set(int x, int y, float v) {
    buf[y*width+x] = v;
}

float Buffer::getUpperRange() {
    return upper_range;
}

void Buffer::clearBuffer() {
    set_limit_value();
}

#endif