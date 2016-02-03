#include <iostream>
#include "rader.h"

#pragma comment(lib, "freeglut.lib")

Data test_M;


void display()
{
	test_M.renderScence();
}

void processNormalKeys(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 's': test_M.bSaving = true; break;
	case 27:
		exit(0);
		break;
	}
}

void init(void) {
	

	test_M.init();

}

void main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(300, 300);
	glutInitWindowSize(128, 128);
	glutCreateWindow("Radar");
	
	init();
	
	
	glutDisplayFunc(display);
	glutKeyboardFunc(processNormalKeys);
	glutIdleFunc(display);
	glutMainLoop();
}