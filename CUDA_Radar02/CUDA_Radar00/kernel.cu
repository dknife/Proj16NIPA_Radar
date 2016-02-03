
#include "kernel.h"



void callTransformRawData(int bpg, int tpb, unsigned char *xformedData, unsigned char *rawData) {
	transformRawData << < bpg, tpb >> >(xformedData, rawData);
}

__global__ void transformRawData(unsigned char *xformedData, unsigned char *rawData)
{

	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;

	int X = tId % MESHWIDTH;
	int Y = tId / MESHWIDTH;

	float x = 2.0*float(X) / float(MESHWIDTH-1) - 1.0;
	float y = 2.0*float(Y) / float(MESHHEIGHT-1) - 1.0;

	float r = sqrt(x*x + y*y);
	if (r >= 1.0) {
		xformedData[X + Y*MESHWIDTH] = 0;
		return;
	}
	float angle = acos(x / r);
	if (y < 0) angle = 3.141592*2.0 - angle;
	int t = MESHHEIGHT * r;
	int s = (angle / (3.141592*2.0)) * MESHWIDTH;

	xformedData[X + Y*MESHWIDTH] = rawData[t + s*MESHWIDTH];

}

void callInitColumnLabel(int bpg, int tpb, int *d_label, unsigned char *d_filteredData)  {
	initColumnLabel<<<bpg, tpb>>>(d_label, d_filteredData);
}
__global__ void initColumnLabel( int *d_label, unsigned char *d_filteredData) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;

	if(d_filteredData[tId]>250) d_label[tId] = tId;
	else d_label[tId] = -1;
}

void callUpdateColumnLabel(int bpg, int tpb, int *d_label) {
	updateColumnLabel<<<bpg, tpb>>>(d_label);
}
__global__ void updateColumnLabel(int *d_label) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH) return;

	int X = tId;
	int Y;
	for(int row=MESHHEIGHT-2; row>=0; row--) {
		Y = row;
		int PixelCur   = X + Y*MESHWIDTH;
		int PixelBelow = X + (Y+1)*MESHWIDTH;
		d_label[PixelCur] = (d_label[PixelBelow]>0 && d_label[PixelCur]>0)? d_label[PixelBelow] : d_label[PixelCur];
	}
}



void callLabelMerge(int bpg, int tpb, int *d_label) {
	labelMerge<<<bpg, tpb>>>(d_label);
}

__device__ int getRoot(int *d_label, int idx) {
	if(d_label[idx]<0) return -1;
	while(d_label[idx]!=idx) {
		idx = d_label[idx];
	}
	return idx;
}




__global__ void labelMerge(int *d_label) {

	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHHEIGHT) return;
	
	int Y = tId;
	int rowAdd = Y*MESHHEIGHT;
	int col=MESHWIDTH - 2;
	int rC, rR;
	int min,max;

	while(col>=0) {
		rC = rowAdd + col--;
		rR = rC + 1;

		if (d_label[rC] == -1) continue;
		if (d_label[rR] == -1) continue;

	
		while(d_label[rC]!=rC) { rC = d_label[rC]; }
		while(d_label[rR]!=rR) { rR = d_label[rR]; }
			
		min = rR<rC?rR:rC;
		d_label[min] = rC + rR - min;
		
				
	}
}



void callRelabel(int bpg, int tpb, int *d_label, unsigned char *d_filteredData) {
	relabel<<<bpg, tpb>>>(d_label, d_filteredData);
}

__global__ void relabel(int *d_label, unsigned char *d_filteredData) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return; 

	d_label[tId] = getRoot(d_label, tId);
}

void callCountBlobs(int bpg, int tpb, int *d_label, int *count, int *d_blobLabels, int *blobBounds)  {
	initCount<<<1, 1>>>(count);
	countBlobs<<<bpg, tpb>>>(d_label, count, d_blobLabels, blobBounds);
}

__global__ void initCount(int *count) {
	count[0] = 0;
}

__global__ void countBlobs(int *d_label, int *count, int *blobLabels, int *blobBounds) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;
	
	if(d_label[tId]==tId) {
		// atomic functions require compute_20, sm_20 option!
		// atomic add returns "old"
		int idx = atomicAdd(&count[0], 1);
		blobLabels[idx]=tId;
		blobBounds[idx*4+0]=MESHWIDTH; // minX
		blobBounds[idx*4+1]=0; // maxX
		blobBounds[idx*4+2]=MESHHEIGHT; // minY
		blobBounds[idx*4+3]=0; // maxYb
	}
}

void callExtractBounds(int bpg, int tpb, int *lock, int *d_label, int *d_blobLabels, int *count, int *d_blobBounds) {
	setBlobLabelEndMark<<<1, 1>>>(d_blobLabels, count);
	extractBounds<<<bpg, tpb>>>(lock, d_label, d_blobLabels, count, d_blobBounds);
}

__global__ void setBlobLabelEndMark(int *d_blobLabels, int *count) {
	d_blobLabels[count[0]] = -1;
}

__global__ void extractBounds(int *lock, int *d_label, int *d_blobLabels, int *count, int *d_blobBounds) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;

	int X = tId % MESHWIDTH;
	int Y = tId / MESHWIDTH;

	if(d_label[tId]>0) {
		
		int idx=0;
		while(d_blobLabels[idx] != d_label[tId] && idx<count[0]) idx++;
		
		
		if(idx<count[0]) {
			
			bool leaveLoop=false;
			while(!leaveLoop) {
				if(atomicExch(lock, 1) ==0) {
					// critical section!!!!!! - Performance bottleneck
					if(d_blobBounds[idx*4+0] > X) d_blobBounds[idx*4+0]=X;
					if(d_blobBounds[idx*4+1] < X) d_blobBounds[idx*4+1]=X;
					if(d_blobBounds[idx*4+2] > Y) d_blobBounds[idx*4+2]=Y;
					if(d_blobBounds[idx*4+3] < Y) d_blobBounds[idx*4+3]=Y;
					leaveLoop = true;
					atomicExch(lock, 0);
				}
			}
		}
		
	}
}

void callFilterOut(int bpg, int tpb, unsigned char *d_filteredData, unsigned char *d_xformedData) {
	filterOut << < bpg, tpb >> > (d_filteredData, d_xformedData);
}

__device__ float getUCharData(unsigned char *data, int x, int y) {
	if (x < 0) x = 0;
	else if (x >= MESHWIDTH) y = MESHWIDTH - 1;
	if(y < 0) y = 0;
	else if (y >= MESHHEIGHT) y = MESHHEIGHT - 1;

	return (float)data[x + y*MESHWIDTH];

}
__global__ void filterOut(unsigned char *filteredData, unsigned char *xformedData) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;

	int X = tId % MESHWIDTH;
	int Y = tId / MESHWIDTH;

	float filteredValue = 
		0.1*getUCharData(xformedData, X - 1, Y + 1) +
		0.1*getUCharData(xformedData, X, Y + 1) +
		0.1*getUCharData(xformedData, X + 1, Y + 1) +
		0.1*getUCharData(xformedData, X - 1, Y) +
		0.1*getUCharData(xformedData, X, Y) +
		0.1*getUCharData(xformedData, X + 1, Y) +
		0.1*getUCharData(xformedData, X - 1, Y - 1) +
		0.1*getUCharData(xformedData, X, Y - 1) +
		0.1*getUCharData(xformedData, X + 1, Y - 1);

	filteredData[X + Y*MESHWIDTH] = (filteredValue > 150) ? 255 : 0;
}

void callInitialize_VBORadar(int bpg, int tpb, float3* vboData) {
	initialize_VBORadar << < bpg, tpb >> >(vboData);
}

__global__ void initialize_VBORadar(float3* vboData)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;

	int X = tId % MESHWIDTH;
	int Y = tId / MESHWIDTH;

	
	Color temp;
	temp.components = make_uchar4(255, 255, 255, 1);
	float x = 2.0*float(X) / float(MESHWIDTH-1) - 1.0;
	float y = 2.0*float(Y) / float(MESHHEIGHT-1) - 1.0;
	//Set initial position, color and velocity
	vboData[X + Y*MESHWIDTH] = make_float3(x, y, temp.c);
}


void callChange_VBORadar(int bpg, int tpb, float3* vboData, unsigned char *visualData, int *label, bool bCCL) {
	change_VBORadar << <bpg, tpb >> >(vboData, visualData, label, bCCL);
}



__global__ void change_VBORadar(float3* vboData, unsigned char *data, int *label, bool bCCL)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;

	int X = tId % MESHWIDTH;
	int Y = tId / MESHWIDTH;

	int mod = label[tId]%3;

	unsigned char intensity = data[X + Y*MESHWIDTH];
	//if (intensity < 150) intensity = 50;

	float x = 2.0*float(X) / float(MESHWIDTH-1) - 1.0;
	float y = 2.0*float(Y) / float(MESHHEIGHT-1) - 1.0;
	Color temp;
	temp.components = make_uchar4(0, intensity, 0, 1);
	if ((x*x + y*y) > 1.0) temp.components = make_uchar4(0,0,128, 1);
	//Set initial position, color and velocity
	vboData[X + Y*MESHWIDTH] = make_float3(x, y, temp.c);
}

#ifdef RAWDATAVISUALIZATION

void callInitialize_VBORawData(int bpg, int tpb, float3* vboData) {
	initialize_VBORawData << < bpg, tpb >> >(vboData);
}
__global__ void initialize_VBORawData(float3* vboData)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;

	int t = tId % MESHWIDTH;
	int s = tId / MESHWIDTH;



	float x = 2.0*float(s) / MESHWIDTH - 1.0;
	float y = 2.0*float(t) / MESHHEIGHT - 1.0;
	//Set the initial color
	Color temp;
	temp.components = make_uchar4(255, 255, 255, 1);

	//Set initial position, color and velocity
	vboData[t + s*MESHWIDTH] = make_float3(x, y, temp.c);

}

void callChange_VBORawData(int bpg, int tpb, float3* vboData, unsigned char *radarData) {
	change_VBORawData << <bpg, tpb >> >(vboData, radarData);
}
__global__ void change_VBORawData(float3* vboData, unsigned char *radarData)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= MESHWIDTH*MESHHEIGHT) return;

	int t = tId % MESHWIDTH;
	int s = tId / MESHWIDTH;

	float x = vboData[t + s*MESHWIDTH].x;
	float y = vboData[t + s*MESHWIDTH].y;

	unsigned char intensity = radarData[t + s*MESHWIDTH];
	Color temp;
	temp.components = make_uchar4(intensity, intensity, intensity, 1);

	//Set initial position, color and velocity
	vboData[t + s*MESHWIDTH] = make_float3(x, y, temp.c);
}
#endif

