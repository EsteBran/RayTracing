#pragma once
#include <vector_types.h>
#include "math_util.h"


void kernelBindPbo(GLuint pixelBufferObj);
void kernelUpdate(int width, int height, int samples, int iteration);
void kernelExit(GLuint pixelBufferObj);

void allocateMemory(int width, int height);
void deallocateMemory();
//void writeToFile(int width, int height);

void setCamera(float ox, float oy, float oz, float dx, float dy, float dz, float fov_);
void changeCamera(float movex, float movey, float movez, float theta);

void renderKernel(int width, int height, int iteration, int samples);



