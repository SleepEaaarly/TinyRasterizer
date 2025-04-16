#include "texture.h"

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
    return image.get((int)(u * width), (int)(v * height));
}

TGAImage* Texture::getImagePtr() {
    return &image;
}