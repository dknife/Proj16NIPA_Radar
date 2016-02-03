#include "cudaPerformance.h"

void CCudaPerf::start() {
	cudaEventRecord( eventStart, 0 );
}

float CCudaPerf::stop() {
	cudaEventRecord( eventEnd, 0 );
	cudaEventSynchronize(eventEnd);
	cudaEventElapsedTime( &elapsedTimeInMs, eventStart, eventEnd );
	return elapsedTimeInMs;
}