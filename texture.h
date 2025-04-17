#ifndef TEXTURE_H
#define TEXTURE_H

#include "image.h"
#include "geometry.h"

class Texture
{
private:
    Image image;
public:
    Texture(const char *filename);
    ~Texture();
    Color get_color(float u, float v);
    Image *getImagePtr();
};
#endif