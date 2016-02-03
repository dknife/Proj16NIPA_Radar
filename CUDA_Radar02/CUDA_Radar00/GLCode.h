#ifndef _GL_CODES_FOR_CUDARADAR_HH
#define _GL_CODES_FOR_CUDARADAR_HH

#include <cuda.h>
#include <iostream>
#include <math.h>
#include <ctime>
#include <gl\glew.h>
#include <gl\glut.h>
#include <cuda_runtime.h> 
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <stdio.h>
#include "definitions.h"

void processData(void);
void initData(const char *fname);
void keyboard(unsigned char key, int, int);
void reshape(int w, int h);
void initializeVBOs();
static void idle(void);
static void display(void);
void createVBO();
void initGL(int argc, char **argv);
void printStringSmall(const char *str, float x, float y, float z, float color[4]=NULL);
void printString(const char *str, float x, float y, float z, float color[4]=NULL);

#endif