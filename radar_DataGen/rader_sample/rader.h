#ifndef RADER
#define RADER

#include <GL\freeglut.h>
#include <stdio.h>

#define MAXDATA 120
#define NUMOBJECTS 50
struct loc {
	float x;
	float y;
	float vx;
	float vy;
	float size;
};

class Data{
	unsigned char *image;
	loc locs[NUMOBJECTS];
	FILE *pFile;
	
public:
	bool bSaving;
	int  count;
	void renderScence(void);
	void init();
	void createdata();
	void number_create();
	void data_save(FILE *pFile);
	
};
#endif