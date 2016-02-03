#include <cuda.h>
#include <iostream>
#include <math.h>
#include <ctime>
#include <gl\glew.h>
#include <gl\glut.h>
#include <cuda_runtime.h> 
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include "definitions.h"
#include "kernel.h"
#include "GLCode.h"


int main(int argc, char** argv)
{

	initGL(argc, argv);

	initData(INPUTFILENAME);

	//Create VBO
	createVBO();
	initializeVBOs();

	glutMainLoop();

	//Free CUDA variables
	return 0;
}
