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

Texture::Texture(const char *filename)
{
    image.read_tga_file(filename);
    image.flip_vertically();
}

Texture::~Texture()
{

}

Color Texture::get_color(float u, float v) {
    int width = image.get_width();
    int height = image.get_height();
    return image.get((int)(u * width + .5), (int)(v * height + .5));
}

#endif