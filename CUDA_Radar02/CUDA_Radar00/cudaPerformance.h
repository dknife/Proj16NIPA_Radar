#ifndef _cuda_performance_ymkang_hh
#define _cuda_performance_ymkang_hh


#include <driver_types.h>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cuda_runtime_api.h>


class CCudaPerf {
	float elapsedTimeInMs;
	cudaEvent_t eventStart, eventEnd;

public:
	CCudaPerf() : elapsedTimeInMs(0.0f) {
		cudaEventCreate( &eventStart );
		cudaEventCreate( &eventEnd );
	}

	void start(void);
	float stop(void);
};

#endif