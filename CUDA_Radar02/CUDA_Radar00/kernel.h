#ifndef _KERNELS_H_H
#define _KERNELS_H_H

#include <cuda.h>
#include <iostream>
#include <math.h>
#include <ctime>
#include <gl\glew.h>
#include <gl\glut.h>
#include <cuda_runtime.h> 
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include "definitions.h"


union Color
{
	float c;
	uchar4 components;
};

void callTransformRawData(int bpg, int tpb, unsigned char *xformedData, unsigned char *rawData);
__global__ void transformRawData(unsigned char *xformedData, unsigned char *rawData);


void callFilterOut(int bpg, int tpb, unsigned char *d_filteredData, unsigned char *d_xformedData);
__global__ void filterOut(unsigned char *d_filteredData, unsigned char *d_xformedData);

void callInitColumnLabel(int bpg, int tpb, int *d_label, unsigned char *d_filteredData);
__global__ void initColumnLabel(int *d_label, unsigned char *d_filteredData);

void callUpdateColumnLabel(int bpg, int tpb, int *d_label);
__global__ void updateColumnLabel(int *d_label); 

void callLabelMerge(int bpg, int tpb, int *d_label);
__global__ void labelMerge(int *d_label); 

void callRelabel(int bpg, int tpb, int *d_label, unsigned char *d_filteredData);
__global__ void relabel(int *d_label, unsigned char *d_filteredData);


void callCountBlobs(int bpg, int tpb, int *d_label, int *count, int *d_blobLabels, int *blobBounds) ;
__global__ void initCount(int *count);
__global__ void countBlobs(int *d_label, int *count, int *d_blobLabels, int *blobBounds);

void callExtractBounds(int bpg, int tpb, int *lock, int *d_label, int *d_blobLabels, int *count, int *d_blobBounds);
__global__ void setBlobLabelEndMark(int *d_blobLabels, int *count);
__global__ void extractBounds(int *lock, int *d_label, int *d_blobLabels, int *count, int *d_blobBounds);

void callInitialize_VBORadar(int bpg, int tpb, float3* vboData);
__global__ void initialize_VBORadar(float3* vboData);

 void callChange_VBORadar(int bpg, int tpb, float3* vboData, unsigned char *visualData, int *label, bool bCCL);
__global__ void change_VBORadar(float3* vboData, unsigned char *visualData, int *label, bool bCCL);

#ifdef RAWDATAVISUALIZATION
void callInitialize_VBORawData(int bpg, int tpb, float3* vboData);
__global__ void initialize_VBORawData(float3* vboData);

void callChange_VBORawData(int bpg, int tpb, float3* vboData, unsigned char *radarData);
__global__ void change_VBORawData(float3* vboData, unsigned char *radarData);
#endif

#endif