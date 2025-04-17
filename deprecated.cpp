#include "image.h"


bool check_edge(int x, int y, int width, int height) {
	if (x < 0 || x >= width || y < 0 || y >= height)
		return false;
	return true;
}

void draw_line(int x0, int y0, int x1, int y1, Image &image, Color c) {
	// 单点
	if (x0 == x1 && y0 == y1) {
		image.set(x0, y0, c);
		return;
	}
	
	// 同行或同列
	if (x0 == x1) {
		if (y0 > y1)	std::swap(y0, y1);
		for (int y = y0; y <= y1; ++y) {
			image.set(x0, y, c);
		}
		return;
	}
	else if (y0 == y1) {
		if (x0 > x1)	std::swap(x0, x1);
		for (int x = x0; x <= x1; ++x) {
			image.set(x, y0, c);
		}
		return;
	}

	// 普通情况
	bool trans = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
		std::swap(x0, y0);
		std::swap(x1, y1);
		trans = true;
	}
	if (x0 > x1) {					// from left to right
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	int dx = x1 - x0;
	int dy = y1 - y0;
	float ddy = std::abs(dy/float(dx));
	float inc = 0;
	int y = y0;
	int width = image.get_width();
	int height = image.get_height();

	for (int x = x0; x <= x1; x++) {
		if (trans) {
			if (check_edge(y, x, width, height))
				image.set(y, x, c);
		} else {
			if (check_edge(x, y, width, height))
				image.set(x, y, c);
		}
		inc += ddy;
		if (inc > .5) {
			y += dy > 0 ? 1 : -1;
			inc -= 1.f;
		}
	}
}

// #include <iostream>

// void outputVector(Vec3f v) {
// 	for (int i = 0; i < 3; i++) {
// 		std::cout << v[i] << " ";
// 	}
// 	std::cout << std::endl;
// }

// void outputMatrix(Matrix m) {
// 	for (int i = 0; i < 4; i++) {
// 		for (int j = 0; j < 4; j++) {
// 			std::cout << m[i][j] << " ";
// 		}
// 		std::cout << std::endl;
// 	}
// 	std::cout << std::endl;
// }