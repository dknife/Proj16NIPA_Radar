#include "rader.h"

#include <vector>
#include <fstream>
#include <glm\glm.hpp>


void Data::init()
{
	glEnable(GL_DEPTH_TEST);
	glClearColor(1.f, 1.f, 1.f, 0.f);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45, 1, 0.01, 1000);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, 1024, 1024);

	image = new unsigned char[4096 * 4096];
	
	for (int i = 0; i < NUMOBJECTS; i++) {
		float r = float(rand() % 10001) / 10000.0;
		float angle = float(rand() % 10001) * 3.141592 / 500.0;
		locs[i].x = r * cos(angle);
		locs[i].y = r * sin(angle);
		locs[i].vx = rand() % 10001 / 500000.0 - 0.01;
		locs[i].vy = rand() % 10001 / 500000.0 - 0.01;
		locs[i].size = 0.005*(rand() % 3 + 1);

		printf("r: %f  angle: %f  (%f, %f) (%f , %f)\n", r, angle, locs[i].x, locs[i].y, locs[i].vx, locs[i].vy);
	}
	bSaving = false;
	count = 0;
	char fname[256];
	sprintf_s(fname, "radarData.txt");
	fopen_s(&pFile, fname, "wb");

}

void Data::renderScence(void)
{
	glClearColor(0.8f, 0.8f, 0.8f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, 1024, 1024);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(0, 0, 800, 0, 0, 0, 0, 1, 0);


	if (bSaving) {
		data_save(pFile);
	}

	glutSwapBuffers();
}


void Data::data_save(FILE *pFile)
{
	
	

	printf("count = %d\n", count);
	if (count >= MAXDATA) return;

	for (int i = 0; i < 4096; i++) {
		for (int j = 0; j < 4096; j++) {
			float r = ((rand() % 1000) / 1000.0)*((rand() % 1000) / 1000.0)*((rand() % 1000) / 1000.0)*((rand() % 1000) / 1000.0);
			r *= r;
			image[i * 4096 + j] = r*256;
		}
	}

	for (int obj = 0; obj < NUMOBJECTS; obj++) {
		locs[obj].vx *= ((rand() % 2001 / 10000.0) + 0.9);
		locs[obj].vy *= ((rand() % 2001 / 10000.0) + 0.9);
		locs[obj].x += locs[obj].vx;
		locs[obj].y += locs[obj].vy;

		if (sqrt(locs[obj].x*locs[obj].x + locs[obj].y*locs[obj].y) > 1.0) {
			locs[obj].x = -locs[obj].x;
			locs[obj].y = -locs[obj].y;
		}
		float vx = locs[obj].vx;
		float vy = locs[obj].vy;
		float vlen = sqrt(vx*vx + vy*vy);
		float shipLen = locs[obj].size;
		float step = 0.001;
		int nSteps = int(shipLen / step);
		if (vlen > 0) {
			vx /= vlen; vy /= vlen;
		}

		int dT = int(shipLen * 4096);
		
		float r=0.0;
		for (int s = 0; s < 4096; s++) {
			float x, y;
			int i, j;
			for (i = 0, x = locs[obj].x - vx*shipLen*0.5, y = locs[obj].y - vy*shipLen*0.5; i<nSteps && r < 1.0; x += vx*step, y+= vy*step, i++) {
				r = sqrt(x*x + y*y);
				if (r > 1.0) {
					continue;
				}

				float angle = 2.0*3.141592 * s / 4095.0;
				int t = 4095 * r;

				for (int T = t - dT/2; T < t + dT/2; T++) {
					if (T < 0) continue;
					if (T >= 4096) continue;
					float R = T / 4096.0;
					float rx = R * cos(angle);
					float ry = R * sin(angle);

					float d = sqrt((rx - x)*(rx - x) + (ry - y)*(ry - y));
					if (d < shipLen) {
						unsigned char val = image[T + 4096 * s];
						unsigned char newVal = 255 * (shipLen - d) / shipLen;
						if (newVal > val) image[T + 4096 * s]  = newVal;
					}
				}
				
			}
		}

		

	}

	printf("writing file\n");
	int maxData = MAXDATA;
	if(count==0) fwrite(&maxData, 1, sizeof(int) * 1, pFile);
	fwrite(image, 1, sizeof(unsigned char) * 4096 * 4096, pFile);

	count++;
	
	
	

	
}

