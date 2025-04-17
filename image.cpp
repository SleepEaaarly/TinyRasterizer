#include "image.h"
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


Image::Image() : data(NULL), width(0), height(0), bytespp(0) {
}

Image::Image(const char *filename) {
    data = stbi_load(filename, &width, &height, &bytespp, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << stbi_failure_reason() << std::endl;
        exit(-1);
    }
}

Image::Image(int w, int h, int bpp) : data(NULL), width(w), height(h), bytespp(bpp) {
	unsigned long nbytes = width*height*bytespp;
	data = new unsigned char[nbytes];
	memset(data, 0, nbytes);
}

Image::Image(const Image &img) {
	width = img.width;
	height = img.height;
	bytespp = img.bytespp;
	unsigned long nbytes = width*height*bytespp;
	data = new unsigned char[nbytes];
	memcpy(data, img.data, nbytes);
}

Image::~Image() {
	if (data) stbi_image_free(data);
}

void Image::read_file(const char *filename) {
    data = stbi_load(filename, &width, &height, &bytespp, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << stbi_failure_reason() << std::endl;
        exit(-1);
    }
}

void Image::write_png_file(const char *filename) {
    stbi_write_png(filename, width, height, bytespp, data, 0);
}

Color Image::get(int x, int y) {
	if (!data || x<0 || y<0 || x>=width || y>=height) {
		return Color();
	}
	return Color(data+(x+y*width)*bytespp, bytespp);    
}

bool Image::set(int x, int y, Color c) {
    if (!data || x<0 || y<0 || x>=width || y>=height) {
		return false;
	}
	memcpy(data+(x+y*width)*bytespp, c.raw, bytespp);
	return true;
}

int Image::get_bytespp() {
	return bytespp;
}

int Image::get_width() {
	return width;
}

int Image::get_height() {
	return height;
}

unsigned char *Image::buffer() {
	return data;
}

void Image::clear() {
	memset((void *)data, 0, width*height*bytespp);
}


bool Image::flip_horizontally() {
	if (!data) return false;
	int half = width>>1;
	for (int i=0; i<half; i++) {
		for (int j=0; j<height; j++) {
			Color c1 = get(i, j);
			Color c2 = get(width-1-i, j);
			set(i, j, c2);
			set(width-1-i, j, c1);
		}
	}
	return true;
}

bool Image::flip_vertically() {
	if (!data) return false;
	unsigned long bytes_per_line = width*bytespp;
	unsigned char *line = new unsigned char[bytes_per_line];
	int half = height>>1;
	for (int j=0; j<half; j++) {
		unsigned long l1 = j*bytes_per_line;
		unsigned long l2 = (height-1-j)*bytes_per_line;
		memmove((void *)line,      (void *)(data+l1), bytes_per_line);
		memmove((void *)(data+l1), (void *)(data+l2), bytes_per_line);
		memmove((void *)(data+l2), (void *)line,      bytes_per_line);
	}
	delete [] line;
	return true;
}