#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/wait.h> 
#include <sys/types.h>
#include <sys/stat.h>
#include "cuda-chrdev.h"
#include "cuda.h"

void printResult(const char *name, CUresult r) {
	printf(name);
	if (r == CUDA_SUCCESS) printf(" success\n");
	else {
	if (r == CUDA_ERROR_INVALID_VALUE) printf(" invalid value\n");
	else if (r == CUDA_ERROR_INVALID_DEVICE) printf(" invalid device\n");
	else if (r == CUDA_ERROR_DEINITIALIZED) printf(" deinitialized\n");
	else if (r == CUDA_ERROR_NOT_INITIALIZED) printf(" not initialized\n");
	else if (r == CUDA_ERROR_INVALID_CONTEXT) printf(" invalid context\n");
	else printf(" error: %d\n",r);
	exit(1);
	}
}

void runtest(int fd) {
	int i, N = 1;
	struct param p;	
	CUdevice dev;
	CUcontext ctx;
	CUmodule mod;
	CUfunction function;

        char error_str[100];

        gethostname(error_str, 100*sizeof(char));
	p.id = error_str[strlen(error_str)-1]-'0';

	//cuInit
        p.result = CUDA_ERROR_INVALID_VALUE;
	p.flags = 0;
	if (ioctl(fd,CUINIT, &p)) {
                perror("ioctl");
	}
	printResult("cuInit",p.result);

	//cuDeviceGet
        p.result = CUDA_ERROR_INVALID_VALUE;
	p.val1 = 0;
	if (ioctl(fd,CUDEVICEGET, &p)) {
                perror("ioctl");
        }
	printResult("cuDeviceGet",p.result);
	dev = p.device;

	//cuCtxCreate
	p.flags = 0;
	p.device = dev;
        p.result = CUDA_ERROR_INVALID_VALUE;
	if(ioctl(fd,CUCTXCREATE, &p)) {
		perror("ioctl");
	}
	printResult("cuCtxCreate",p.result);
        ctx = p.ctx;

	//cuModuleLoad
	p.name = "hello-world.sm_20.cubin";
	p.val1 = strlen("hello-world.sm_20.cubin");
	p.result = CUDA_ERROR_INVALID_VALUE;
	if(ioctl(fd,CUMODULELOAD, &p)) {
                perror("ioctl");
        }
        printResult("cuModuleLoad",p.result);
	mod = p.module;

	//cuMemAlloc
	p.result = CUDA_ERROR_INVALID_VALUE;
	p.bytesize = N*sizeof(int);
	printf("bytesize %lu\n", p.bytesize);
        if(ioctl(fd,CUMEMALLOC, &p)) {
                perror("ioctl");
        }
        printResult("cuMemAlloc",p.result);
	CUdeviceptr ptr = p.dptr;

	//cuMemcpyHtoD
	int *a = (int *)malloc(N*sizeof(int));

	for (i=0; i<N; i++) {
		*((int *)a+i) = i;
	}

	p.result = CUDA_ERROR_INVALID_DEVICE;
	p.host = (void *) a;
	p.dptr = ptr;
	p.bytesize = (size_t) N*sizeof(int);
	if(ioctl(fd,CUMEMCPYHTODV2, &p)) {
                perror("ioctl");
        }
	printResult("cuMemcpyHtoD",p.result);

	//cuModuleGetFunction
	p.result = CUDA_ERROR_INVALID_DEVICE;
	p.module = mod;
	p.name = "_Z4add1Piii";
	p.length = strlen("_Z4add1Piii");
	if(ioctl(fd,CUMODULEGETFUNCTION, &p)) {
                perror("ioctl");
        }
        printResult("cuModuleGetFunction",p.result);
	function = p.function;
	
	//cuLaunchKernel
        p.result = CUDA_ERROR_INVALID_DEVICE;
	p.function = function;
	p.gridDimX = 1;
	p.gridDimY = 1;
	p.gridDimZ = 1;
	p.blockDimX = 1024;
	p.blockDimY = 1;
	p.blockDimZ = 1;
	p.bytecount = 0;
	p.args = malloc(3*sizeof(void *));
        p.size = malloc(3*sizeof(size_t));
	p.args[0] = (void *)&ptr;
        p.size[0] = sizeof(ptr);
	int x = 3;
        p.args[1] = (void *)&x;
	p.size[1] = sizeof(int);
        p.args[2] = (void *)&N;
        p.size[2] = sizeof(int);
	p.val1 = 3;
	if(ioctl(fd,CULAUNCHKERNEL, &p)) {
                perror("ioctl");
        }
        printResult("cuLaunchKernel",p.result);

	//cuMemcpyDtoH
	int *b = malloc(N*sizeof(int));	

        p.result = CUDA_ERROR_INVALID_DEVICE;
        p.host = (void *) b;
        p.dptr = ptr;
        p.bytesize = (size_t) N*sizeof(int);
        if(ioctl(fd,CUMEMCPYDTOHV2, &p)) {
                perror("ioctl");
        }
        printResult("cuMemcpyDtoH",p.result);

	int f = 1;
        for (i=0; i<N; i++) {
//              printf("%d ",a[i]);
                if (b[i] != 3) {
                        f = 0;
			break;
                }
        }
//      printf("\n");
        if (f) printf("passed\n");
        else printf("failed\n");

	//cuMemFree
	p.dptr = ptr;
	if(ioctl(fd,CUMEMFREE, &p)) {
                perror("ioctl");
        }
        printResult("cuMemFree",p.result);

	p.module = mod;
	if(ioctl(fd, CUMODULEUNLOAD, &p)) {
		perror("ioctl");
	}
	printResult("cuModuleUnload", p.result);

	//cuCtxDestroy
	p.ctx = ctx;
        p.result = CUDA_ERROR_INVALID_VALUE;
        if(ioctl(fd,CUCTXDESTROY, &p)) {
                perror("ioctl");
        }
        printResult("cuCtxDestroy",p.result);

}

int main(int argc, char **argv)
{
	int fd = -1;
        char *filename;
        char error_str[100];
	
	filename = "/dev/cuda0";
        fd = open(filename, O_RDWR, 0);
        if (fd < 0) {
                sprintf(error_str, "open %s", filename);
                perror(error_str);
                return 1;
        }
	
	runtest(fd);	

	if (close(fd)) {
		perror("close");
		return 1;
	}
	
	return 0;
}
