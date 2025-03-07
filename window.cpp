#include "window.h"
#include <iostream>

int Window::screenKeys[512] = {0};
bool Window::screenExit = false;

LRESULT CALLBACK keyboardCallback(HWND hWnd, UINT msg, 
	WPARAM wParam, LPARAM lParam) {
	switch (msg) {
	case WM_CLOSE: Window::screenExit = true; break;
	case WM_KEYDOWN: Window::screenKeys[wParam & 511] = 1; break;
	case WM_KEYUP: Window::screenKeys[wParam & 511] = 0; break;
	default: return DefWindowProc(hWnd, msg, wParam, lParam);
	}
	return 0;
}

void Window::init() {
    // 描述窗口的属性
    WNDCLASS wndClass = { CS_BYTEALIGNCLIENT, (WNDPROC)keyboardCallback, 0, 0, 0, NULL, NULL, NULL, NULL, TEXT("Screen") };
	// 获得当前进程的实例句柄
    wndClass.hInstance = GetModuleHandle(NULL);

    if (!RegisterClass(&wndClass)) {
        std::cout << "RegisterClass failed" << std::endl;
        return;
    }

    // 创建窗口
    window = CreateWindow(TEXT("Screen"), TEXT("TinyRasterizer"),
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        0, 0, 0, 0, NULL, NULL, wndClass.hInstance, NULL);
    if (window == NULL) {
        std::cout << "CreateWindow failed" << std::endl;
        return;
    }

    // 获取设备上下文
    HDC hdc = GetDC((window));
    // 创建一个与窗口设备上下文兼容的内存设备上下文，用于双缓冲绘图
	screenHDC = CreateCompatibleDC(hdc);
    
    // 创建位图
    BITMAPINFO bitmapInfo = { { sizeof(BITMAPINFOHEADER),windowWidth, windowHeight, 1, 32, BI_RGB, windowWidth * windowHeight * 4, 0, 0, 0, 0 } };
    LPVOID ptr;
    HBITMAP bitmapHandler = CreateDIBSection(screenHDC, &bitmapInfo, DIB_RGB_COLORS, &ptr, 0, 0);
	if (bitmapHandler == NULL)
		return;
    
    // 将创建的位图对象bitmapHandler 放入内存设备上下文screenHDC
    HBITMAP screenObject = (HBITMAP)SelectObject(screenHDC, bitmapHandler);
    
    // 获取屏幕的宽度和高度
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    // 计算窗口左上角的坐标，使得窗口中心与屏幕中心对齐
    int windowX = (screenWidth - windowWidth) / 2;
    int windowY = (screenHeight - windowHeight) / 2;
    
    SetWindowPos(window, NULL, windowX, windowY, windowWidth, windowHeight, (SWP_NOCOPYBITS | SWP_NOZORDER | SWP_SHOWWINDOW));

	ShowWindow(window, SW_NORMAL);
    UpdateWindow(window);  
    // pollMessage();
    // memset(screenKeys, 0, sizeof(int)*512);
    memset(ptr, 0, windowWidth*windowHeight*4);
    // screenExit = false;  
}

Window::Window(int w, int h) : windowWidth(w), windowHeight(h) {
    init();
}

void Window::pollMessage() {
	MSG msg;
	while (1) {
		if (!PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE)) break;
		if (!GetMessage(&msg, NULL, 0, 0)) break;
		DispatchMessage(&msg);
	}
}

void Window::updateScreen() {
	HDC hdc = GetDC(window);
    BitBlt(hdc, 0, 0, windowWidth, windowHeight, screenHDC, 0, 0, SRCCOPY);
	ReleaseDC(window, hdc);
    // pollMessage();
}

void Window::drawPixel(int x, int y, Color c) {
    SetPixel(screenHDC, x, y, RGB(c.r, c.g, c.b));
}

void Window::clearScreen() {
    PatBlt(screenHDC, 0, 0, windowWidth, windowHeight, BLACKNESS);
}
