#ifndef TEXTURE_H
#define TEXTURE_H

#include "tgaimage.h"
#include "geometry.h"

class Texture
{
private:
    TGAImage image;
public:
    Texture(const char *filename);
    ~Texture();
    Color get_color(float u, float v);
};
#endif