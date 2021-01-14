#include <iostream>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "pathTracer.h"

#define WIDTH 1400	
#define HEIGHT 1400
#define SAMPLES 2000



GLuint pbo;
using namespace std;


int main() {
	if (!glfwInit()) return -1;
	if (atexit(glfwTerminate)) {
		glfwTerminate();
		return -1;
	}

	GLFWwindow* window;
	window = glfwCreateWindow(WIDTH, HEIGHT, "gl-cuda-test", NULL, NULL);
	if (!window) return -1;

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); //turns on vertical sync, buffers swap to new every frame

	if (glewInit() != GLEW_OK) return -1;

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLubyte) * WIDTH * HEIGHT, NULL, GL_DYNAMIC_DRAW);

	kernelBindPbo(pbo); //bind pbo to cuda
	allocateMemory(WIDTH, HEIGHT); //allocate memory for cuda

	//set up main viewpoint
	
	setCamera(50.0f, 52.0f, 350.0f, 0.0f, 0.0f, -1.0f, 0.5f);
	glfwSetTime(0.0);
	int iteration = 0;
	int samples = 0;
	
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);
		
		if (samples < SAMPLES) { 
			
			iteration++;
			samples++;
			/*if (iteration == 59) {
				changeCamera(0.0f, 0.0f, 0.0f, 5.0f);
				iteration = 0;
			}*/
			
			renderKernel(WIDTH, HEIGHT, iteration, SAMPLES);
		
			// Measure speed
			if (iteration == SAMPLES - 1) {
				cout << "Done" << endl;
			}

			if (glfwGetTime() >= 1.0 && iteration < SAMPLES - 1) {
				cout << samples << " samples/second" << endl;
				samples = 0;
				glfwSetTime(0.0);
			}

			glfwPollEvents();
			
		}
		glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
		
		
	
	}
	
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	kernelExit(pbo);
	deallocateMemory(); //deallocate cuda memory
	glDeleteBuffers(1, &pbo);

	//getchar();
	
	return 0;
}


//void keyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods){
//	if (action == GLFW_PRESS) {
//		switch (key) {
//		case GLFW_KEY_ESCAPE:
//			glfwSetWindowShouldClose(window, GL_TRUE);
//			break;
//		case GLFW_KEY_DOWN:  camChanged = true; theta = -0.1f; break;
//		case GLFW_KEY_UP:    camChanged = true; theta = +0.1f; break;
//		case GLFW_KEY_RIGHT: camChanged = true; phi = -0.1f; break;
//		case GLFW_KEY_LEFT:  camChanged = true; phi = +0.1f; break;
//		case GLFW_KEY_A:     camChanged = true; x -= 0.1f; break;
//		case GLFW_KEY_D:     camChanged = true; x += 0.1f; break;
//		case GLFW_KEY_W:     camChanged = true; y += 0.1f; break;
//		case GLFW_KEY_S:     camChanged = true; y -= 0.1f; break;
//		case GLFW_KEY_R:     camChanged = true; z += 0.1f; break;
//		case GLFW_KEY_F:     camChanged = true; z -= 0.1f; break;
//		}
//	}
//}

