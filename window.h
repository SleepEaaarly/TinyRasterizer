#ifndef WINDOW_H
#define WINDOW_H

#include <windows.h>
#include "tgaimage.h"

class Window
{
public:
	HWND window;
	int windowWidth;
	int windowHeight;
	HDC screenHDC;
	static int screenKeys[512];
	static bool screenExit;


	Window(int w, int h);
	void init();
	void pollMessage();
	void updateScreen();
	void drawPixel(int x, int y, Color c);
	void clearScreen();
	// LRESULT screenCallback(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};


#endif