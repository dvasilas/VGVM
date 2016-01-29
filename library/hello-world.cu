#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <asm/msr.h>
#include <sys/time.h>

__global__ void add1(int *a, int c, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	a[idx] = c;
}
__global__ void add2(int *a, int *b) {
        a[threadIdx.x] += b[threadIdx.x];
}
__global__ void square_array(float *a, int N) {
int blockId = blockIdx.x 
			 + blockIdx.y * gridDim.x 
			 + gridDim.x * gridDim.y * blockIdx.z; 
	int idx = blockId * blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx] * a[idx];
}		

int main (int argc, char **argv) {
//        timers_t htod, dtoh, launch, time, first;

//	TIMER_RESET(&dtoh); TIMER_RESET(&htod); TIMER_RESET(&launch); TIMER_RESET(&time); TIMER_RESET(&first);
//	TIMER_START(&time);
	int i, Nx = atoi(argv[1]), Ny = atoi(argv[2]), Nz = atoi(argv[3]);

	unsigned long N=Nx*Ny*Nz;
        float *dev_a, *a;
	a = (float *)malloc(N*sizeof(float));
	if (!a) {
		perror("malloc");
		exit(1);
	}

        dim3 block(1, 1);
        dim3 grid(Nx, Ny, Nz);

	printf("size %lu\n", N*sizeof(float));
	
	for (i=0; i<N; i++)
		a[i] = 2.0;

	int *t;
//	TIMER_START(&first);
        cudaMalloc((void**)&t, sizeof(t));
//	TIMER_STOP(&first);
        cudaMalloc((void**)&dev_a, N*sizeof(float));

//	TIMER_START(&htod);
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
//	TIMER_STOP(&htod);
//	TIMER_START(&launch);
	square_array<<<grid,block>>>(dev_a, N);
        cudaDeviceSynchronize();
//	TIMER_STOP(&launch);
//        TIMER_START(&dtoh);
        cudaMemcpy(a, dev_a, N*sizeof(float), cudaMemcpyDeviceToHost);
//        TIMER_STOP(&dtoh);

	int f = 1;
	for (i=0; i<N; i++) {
		if (a[i] != 4.0) {
			f = 0;
			break;
		}
	}
	if (f) printf("passed\n");
	else printf("failed\n");

	cudaFree(dev_a);
	free(a);
	cudaDeviceReset();
//	TIMER_STOP(&time);
//	printf("total %llu\n", TIMER_TOTAL(&time));
//	printf("first %llu\n", TIMER_TOTAL(&first));
//        printf("htod %llu\n", TIMER_TOTAL(&htod));
//	printf("launch %llu\n", TIMER_TOTAL(&launch));
//	printf("dtoh %llu\n", TIMER_TOTAL(&dtoh));
	return 0;
}
