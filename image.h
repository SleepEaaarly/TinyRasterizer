#ifndef IMAGE_H
#define IMAGE_H

#include <fstream>

struct Color {
	union {
		struct {
			unsigned char r, g, b, a;
		};
		unsigned char raw[4];
		unsigned int val;
	};
	int bytespp;

	Color() : val(0), bytespp(1) {
	}

	Color(unsigned char R, unsigned char G, unsigned char B, unsigned char A) : r(R), g(G), b(B), a(A), bytespp(4) {
	}

	Color(int v, int bpp) : val(v), bytespp(bpp) {
	}

	Color(const Color &c) : val(c.val), bytespp(c.bytespp) {
	}

	Color(const unsigned char *p, int bpp) : val(0), bytespp(bpp) {
		for (int i=0; i<bpp; i++) {
			raw[i] = p[i];
		}
	}

	Color & operator =(const Color &c) {
		if (this != &c) {
			bytespp = c.bytespp;
			val = c.val;
		}
		return *this;
	}
};


class Image {
protected:
	unsigned char* data;
	int width;
	int height;
	int bytespp;

public:
	enum Format {
		GRAYSCALE=1, RGB=3, RGBA=4
	};

	Image();
    Image(int w, int h, int bpp);
	Image(const char *filename);
	Image(const Image &img);
    void read_file(const char *filename);
	void write_png_file(const char *filename);
	Color get(int x, int y);
	bool set(int x, int y, Color c);
	~Image();
	int get_width();
	int get_height();
	int get_bytespp();
	unsigned char *buffer();
	void clear();
    bool flip_horizontally();
    bool flip_vertically();
};

#endif //__IMAGE_H__
