#include "kernel.h"

#include "definitions.h"
#include <stdio.h>
#include <gl\glew.h>
#include <gl\glut.h>
#include "GLCode.h"
#include "cudaPerformance.h"
#include <driver_types.h>

FILE *inputData;
int nData = 0;
int frameCount = 0;
bool bFilter = false;
bool bRunning = true;
bool bCircleDraw = true;
bool bCCL = false;
bool bVisualizeTrackingLines = true;
bool bVisualizeBounds = true;

// host memory data read from file
unsigned char *radarData;
int *blobLabels;
int *blobBounds;
int nBlobs;

// device memory data
unsigned char *d_radarData;
unsigned char *d_xformedData;
unsigned char *d_filteredData;

// data for CCL (connected component labeling)
int *d_blobLabels;
int *d_blobBounds;
int *d_nBlobs;
int *d_boundLock;
CCudaPerf cudaPerf;

int *d_label;


int window_width = 2048;
int window_height = 1024;

const int mesh_width = MESHWIDTH;
const int mesh_height = MESHWIDTH;

//VBO variables
GLuint vboRadar;
#ifdef RAWDATAVISUALIZATION
GLuint vboRawData;
#endif
float3 *dptr; // cuda device memory for pointing Vertex Buffers


float viewTX = 0;
float viewTY = 0;
float zoom = 1.0;


void initData(const char *fname) {
	inputData = fopen(fname, "rb");
	if (!inputData) {
		printf("file not found\n");
		exit(0);
	}

	fread(&nData, sizeof(int), 1, inputData);
	printf("nData = %d\n", nData);

	///////////// host
	radarData = new unsigned char[mesh_width*mesh_height];


	//blobLabels = new int[MAXBLOBS];
	//blobBounds = new int[MAXBLOBS*4];
	cudaMallocHost((void **)&blobLabels, MAXBLOBS*sizeof(int));
	cudaMallocHost((void **)&blobBounds, 4* MAXBLOBS*sizeof(int));
	nBlobs = 0;

	//////////// device
	cudaMalloc((void **)&d_radarData, mesh_width*mesh_height * sizeof(unsigned char));
	cudaMalloc((void **)&d_xformedData, mesh_width*mesh_height * sizeof(unsigned char));
	cudaMalloc((void **)&d_filteredData, mesh_width*mesh_height * sizeof(unsigned char));
	cudaMalloc((void **)&d_label, mesh_width*mesh_height * sizeof(int) + 1);

	cudaMalloc((void **)&d_blobLabels, MAXBLOBS * sizeof(int)); // label
	cudaMalloc((void **)&d_blobBounds, MAXBLOBS * 4 *  sizeof(int)); // minX, maxX, minY, maxY
	cudaMalloc((void **)&d_nBlobs, sizeof(int));
	cudaMalloc((void **)&d_boundLock, 1 * sizeof(int));
	int boundLockHost=0;
	cudaMemcpy(d_boundLock, &boundLockHost, sizeof(int)*1, cudaMemcpyHostToDevice);


	printf("data preparation done\n");
}

void keyboard(unsigned char key, int, int)
{
	switch (key) {
	case(27) :
		free(radarData);
		cudaFreeHost(blobLabels);
		//free(blobLabels);
		cudaFreeHost(blobBounds);
		//free(blobBounds);

		cudaFree(d_radarData);
		cudaFree(d_xformedData);
		cudaFree(d_filteredData);
		cudaFree(d_label);
		cudaFree(d_blobLabels);
		cudaFree(d_blobBounds);
		cudaFree(d_nBlobs);
		cudaFree(d_boundLock);
		exit(0);
		break;
	case 'c': bCCL = bCCL? false: true; if(bCCL) bFilter=true; processData(); break;
	case 'f': bFilter = bFilter ? false : true; processData(); break;
	case ' ': bRunning = bRunning ? false : true; processData();  break;
	case 'b': bVisualizeBounds = bVisualizeBounds?false:true; processData(); break;
	case 'l': bVisualizeTrackingLines = bVisualizeTrackingLines? false:true; processData(); break;
	case 'r': viewTX = viewTY = 0.0; zoom=1.0; break;
	case 'w': viewTY += zoom*0.1; break;
	case 's': viewTY -= zoom*0.1; break;
	case 'a': viewTX -= zoom*0.1; break;
	case 'd': viewTX += zoom*0.1; break;

	case '-': zoom *= 1.05; break;
	case '=': zoom *= 0.95; break;
	}



	glutPostRedisplay();
}



void reshape(int w, int h) {
	window_width = w;
	window_height = h;
	//Viewport

	glutPostRedisplay();
}


void initializeVBOs()
{
	//Map OpenGL buffer object for writing from CUDA
	cudaGLMapBufferObject((void**)&dptr, vboRadar);
	//Run the initialization kernel
	int blocksPerGrid = (mesh_width * mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	callInitialize_VBORadar( blocksPerGrid, THREADSPERBLOCK , dptr);
	//Unmap buffer object
	cudaGLUnmapBufferObject(vboRadar);

#ifdef RAWDATAVISUALIZATION
	cudaGLMapBufferObject((void**)&dptr, vboRawData);
	//Run the initialization kernel
	blocksPerGrid = (mesh_width * mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	callInitialize_VBORawData(blocksPerGrid, THREADSPERBLOCK, dptr);
	//Unmap buffer object
	cudaGLUnmapBufferObject(vboRawData);
#endif

}

void processData(void) {
	// update transformed data
	int blocksPerGrid = (mesh_width * mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	callTransformRawData(blocksPerGrid, THREADSPERBLOCK, d_xformedData, d_radarData);
	/////////////////////////////////////////////////////

	// filtering out the noises and blobs detection
	
	blocksPerGrid = (mesh_width * mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	callFilterOut(blocksPerGrid, THREADSPERBLOCK, d_filteredData, d_xformedData);

	blocksPerGrid = (mesh_width * mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	callInitColumnLabel(blocksPerGrid, THREADSPERBLOCK, d_label, d_filteredData);

	if(bCCL) {
		float eT;
		float totalET=0.0;
		cudaPerf.start();
		// finding connected component within a column
		// #columns threads are generated row by row ( 1 computation is enough while previous approach needed log(height) computations )
		blocksPerGrid = (mesh_width + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		callUpdateColumnLabel(blocksPerGrid, THREADSPERBLOCK, d_label);
		eT = cudaPerf.stop();
		printf("CUDA CCL run : %f ms\n", eT); totalET+=eT;

		
		cudaPerf.start();
		blocksPerGrid = (mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		callLabelMerge(blocksPerGrid, THREADSPERBLOCK, d_label);
		eT = cudaPerf.stop();
		printf("CUDA CCL hierarchy : %f ms\n", eT); totalET+=eT;

		cudaPerf.start();
		blocksPerGrid = (mesh_width * mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
		callRelabel(blocksPerGrid, THREADSPERBLOCK, d_label, d_filteredData);
		eT = cudaPerf.stop();
		printf("CUDA CCL relabel : %f ms\n", eT); totalET+=eT;

		cudaPerf.start();
		callCountBlobs(blocksPerGrid, THREADSPERBLOCK, d_label,  d_nBlobs, d_blobLabels, d_blobBounds) ;
		eT = cudaPerf.stop();
		printf("CUDA CCL counting : %f ms\n", eT); totalET+=eT;

		cudaPerf.start();
		callExtractBounds(blocksPerGrid, THREADSPERBLOCK, d_boundLock, d_label, d_blobLabels, d_nBlobs, d_blobBounds);
		eT = cudaPerf.stop();
		printf("CUDA CCL bounds : %f ms\n", eT); totalET+=eT;

		cudaPerf.start();
		cudaMemcpy(blobLabels, d_blobLabels, sizeof(int)*MAXBLOBS, cudaMemcpyDeviceToHost);
		cudaMemcpy(blobBounds, d_blobBounds, sizeof(int)*MAXBLOBS*4, cudaMemcpyDeviceToHost);
		eT = cudaPerf.stop();
		printf("CUDA CCL mem cpy : %f ms\n-------------\n", eT); totalET+=eT;

		printf("total elapsed time: %f ms\n-------------\n", totalET); 

	}


	cudaGLMapBufferObject((void**)&dptr, vboRadar);
	// update Radar VBO
	blocksPerGrid = (mesh_width * mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	unsigned char *data = bFilter ? d_filteredData : d_xformedData;
	callChange_VBORadar(blocksPerGrid, THREADSPERBLOCK, dptr, data, d_label, bCCL);
	//Unmap buffer object
	cudaGLUnmapBufferObject(vboRadar);

#ifdef RAWDATAVISUALIZATION
	cudaGLMapBufferObject((void**)&dptr, vboRawData);
	// update Raw Data VBO
	blocksPerGrid = (mesh_width * mesh_height + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	callChange_VBORawData(blocksPerGrid, THREADSPERBLOCK, dptr, d_radarData);
	//Unmap buffer object
	cudaGLUnmapBufferObject(vboRawData);
#endif
}

static void idle(void) {

	if (bRunning) {
		// read next frame
		fread(radarData, sizeof(unsigned char), mesh_width*mesh_height, inputData);
		cudaMemcpy(d_radarData, radarData, sizeof(unsigned char)*mesh_width*mesh_height, cudaMemcpyHostToDevice);

		processData();

		frameCount++;
	}

	

	
	if (frameCount >= nData) {
		frameCount = 0;
		fclose(inputData);

		inputData = fopen(INPUTFILENAME, "rb");
		if (!inputData) {
			printf("file not found\n");
			exit(0);
		}

		fread(&nData, sizeof(int), 1, inputData);
		printf("nData = %d\n", nData);
	}

	glutPostRedisplay();

}

void drawCircles(float zoom) {

	glColor3f(0, 0, 255);
	glBegin(GL_LINES);
	glVertex2f(-1.0, 0.0);
	glVertex2f(1.0, 0.0);
	glVertex2f(0.0, -1.0);
	glVertex2f(0.0, 1.0);
	glEnd();
	for (float r = 0.2; r < 1.0; r += 0.2) {
		glBegin(GL_LINE_LOOP);
		for (int i = 0; i < 360; i++) {
			float angle = 2.0*3.141592*float(i) / 360.0;

			glVertex3f(r*cos(angle), r*sin(angle), 0.0);
		}
		glEnd();
	}
	
}

static void display(void)
{
	char msg[256];
	glClear(GL_COLOR_BUFFER_BIT);

	glViewport(0, 0, window_width / 2, window_height);
	//Projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1 * zoom, 1 * zoom, -1 * zoom, 1 * zoom, -1.0, 1.0);
	//View matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glPushMatrix();
	glTranslatef(viewTX, viewTY, 0);
	//Render from VBO
	glBindBuffer(GL_ARRAY_BUFFER, vboRadar);
	glVertexPointer(3, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	drawCircles(zoom);

	nBlobs=0;
	while(blobLabels[nBlobs]>0||nBlobs>MAXBLOBS)nBlobs++;

	if(bCCL && bVisualizeTrackingLines) {
		glColor3f(0.3, 0.3, 0.3);
		glBegin(GL_LINES);
		
		for(int i=0;i<nBlobs;i++) {
			int idx = blobLabels[i];
			int X = idx % MESHWIDTH;
			int Y = idx / MESHWIDTH;

			float x = 2.0*float(X) / float(MESHWIDTH-1) - 1.0;
			float y = 2.0*float(Y) / float(MESHHEIGHT-1) - 1.0;
			glVertex2f(0,0);
			glVertex2f(x,y);
		}
		glEnd();

	}
	if(bCCL && bVisualizeBounds) {

		
		int minX, maxX, minY, maxY;
		for(int i=0;i<nBlobs;i++) {
			minX = blobBounds[i*4+0];
			maxX = blobBounds[i*4+1];
			minY = blobBounds[i*4+2];
			maxY = blobBounds[i*4+3];
			

			float mx = 2.0*float(minX) / float(MESHWIDTH-1) - 1.0;
			float my = 2.0*float(minY) / float(MESHHEIGHT - 1) - 1.0;
			float mX = 2.0*float(maxX) / float(MESHWIDTH - 1) - 1.0;
			float mY = 2.0*float(maxY) / float(MESHHEIGHT - 1) - 1.0;
			float dx = mX-mx;
			float dy = mY-my;
			
			float size = (maxX-minX+maxY-minY)/100.0;
			float rate = dy/(dx+dy);
			char orient;
			if(rate<0.45) orient = 'X';
			else if(rate>0.55) orient = 'Y';
			else orient = '=';
			sprintf(msg, "%0.2f%c%0.2f", size, orient, rate);
			printStringSmall(msg, mx, mY, 0);
			float rColor=(orient=='X' || orient=='=')?1.0:0.0;
			float gColor=orient=='='?1.0:0.0;
			float bColor=orient=='Y'?1.0:0.0;
			glColor3f(rColor, gColor, bColor );
			glBegin(GL_LINE_LOOP);
			glVertex2f(mx, my);
			glVertex2f(mX, my);
			glVertex2f(mX, mY);
			glVertex2f(mx, mY);
			glEnd();
		}
		

	}

	glPopMatrix();

	
	sprintf(msg, "Data Rate: 20RPM (frame: %d)", frameCount);
	printString(msg, -1 * zoom, 0.9 * zoom, 0);
	sprintf(msg, "Noise Filter: %s", bFilter ? "on" : "off");	
	printString(msg, -1 * zoom, 0.85 * zoom, 0);
	sprintf(msg, "CCL: %s (Vessel Detected: %d", bCCL?"on":"off", bCCL?nBlobs:0);
	printString(msg, -1 * zoom, 0.8 * zoom, 0);
	 
	

#ifdef RAWDATAVISUALIZATION
	glViewport(window_width / 2, 0, window_width / 2, window_height);
	//Projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1, 1, -1, 1, -1.0, 1.0);
	//View matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//Render from VBO
	glBindBuffer(GL_ARRAY_BUFFER, vboRawData);
	glVertexPointer(3, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	printString("Raw Data", -1, 0.9, 0);
	printString("Generated by Young-Min Kang", -1, 0.85, 0);
	char str[256];
	sprintf(str, "Total Data: %dx4096x4096 (%d) bytes", nData, nData * 4096 * 4096);
	printString(str, -1, 0.8, 0);
#endif

	glutSwapBuffers();
}

void createVBO()
{
	//Create vertex buffer object
	glGenBuffers(1, &vboRadar);
	glBindBuffer(GL_ARRAY_BUFFER, vboRadar);

	int size = mesh_width * mesh_height * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Register VBO with CUDA
	cudaGLRegisterBufferObject(vboRadar);

#ifdef RAWDATAVISUALIZATION
	glGenBuffers(1, &vboRawData);
	glBindBuffer(GL_ARRAY_BUFFER, vboRawData);

	size = mesh_width * mesh_height * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGLRegisterBufferObject(vboRawData);
#endif


}

void initGL(int argc, char **argv)
{
	glutInit(&argc, argv);

	//Setup window
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Radar Data - GPU Vis and Processing (TU)");

	//Register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

	//GLEW initialization
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}

	//Clear
	glClearColor(0.0, 0.0, 0.0, 1.0);


	//gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
}

void printStringSmall(const char *str, float x, float y, float z, float color[4])
{
	glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT); // lighting and color mask
	glDisable(GL_LIGHTING);     // need to disable lighting for proper text color
	glDisable(GL_TEXTURE_2D);

	if (color) glColor4fv(color);          // set text color
	else glColor4f(1.0, 1.0, 0.5, 1.0);

	glRasterPos3f(x, y, z);        // place text position

	// loop all characters in the string
	while (*str)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *str);
		++str;
	}

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);
	glPopAttrib();
}

void printString(const char *str, float x, float y, float z, float color[4])
{
	glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT); // lighting and color mask
	glDisable(GL_LIGHTING);     // need to disable lighting for proper text color
	glDisable(GL_TEXTURE_2D);

	if (color) glColor4fv(color);          // set text color
	else glColor4f(1.0, 1.0, 0.5, 1.0);

	glRasterPos3f(x, y, z);        // place text position

	// loop all characters in the string
	while (*str)
	{
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *str);
		++str;
	}
	/*
	#define GLUT_BITMAP_9_BY_15		(&glutBitmap9By15)
	#define GLUT_BITMAP_8_BY_13		(&glutBitmap8By13)
	#define GLUT_BITMAP_TIMES_ROMAN_10	(&glutBitmapTimesRoman10)
	#define GLUT_BITMAP_TIMES_ROMAN_24	(&glutBitmapTimesRoman24)
	#if (GLUT_API_VERSION >= 3)
	#define GLUT_BITMAP_HELVETICA_10	(&glutBitmapHelvetica10)
	#define GLUT_BITMAP_HELVETICA_12	(&glutBitmapHelvetica12)
	#define GLUT_BITMAP_HELVETICA_18	(&glutBitmapHelvetica18)
	*/

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);
	glPopAttrib();
}