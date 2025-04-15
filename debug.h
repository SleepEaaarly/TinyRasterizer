#include "primitive.h"
#include <iostream>

void outputVert(Vert& vert) {
    std::cout << "Vert pos: ";
    for (int i = 0; i < 4; i++) {
        std::cout << vert.pos[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "Vert norm: ";
    for (int i = 0; i < 3; i++) {
        std::cout << vert.norm[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "Vert tex: ";
    for (int i = 0; i < 2; i++) {
        std::cout << vert.tex[i] << ", ";
    }
    std::cout << std::endl; 
    
    std::cout << "Vert pos_view: ";
    for (int i = 0; i < 3; i++) {
        std::cout << vert.pos_view[i] << ", ";
    }
    std::cout << std::endl; 
}