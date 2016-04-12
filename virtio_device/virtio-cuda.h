#ifndef VIRTIO_CUDA_H
#define VIRTIO_CUDA_H

#include <cuda.h>
//used to distinguish between CUDA Driver calls
#define CUINIT                          0
#define CUDRIVERGETVERSION              1
#define CUDEVICEGETCOUNT                3
#define CUDEVICEGET                     4
#define CUDEVICECOMPUTECAPABILITY       5
#define CUDEVICEGETNAME                 6
#define CUDEVICEGETATTRIBUTE            7
#define CUCTXCREATE                     8
#define CUCTXDESTROY                    9
#define CUCTXGETCURRENT                 10
#define CUCTXGETDEVICE                  11
#define CUCTXPOPCURRENT                 12
#define CUCTXSETCURRENT                 13
#define CUMODULELOAD                    14
#define CUMEMALLOC                      15
#define CUMEMCPYHTOD                    16
#define CUMEMCPYDTOH                    17
#define CUMEMSETD32                     18
#define CUMEMFREE                       19
#define CUMODULEGETFUNCTION             20
#define CULAUNCHKERNEL                  21
#define CUCTXSYNCHRONIZE                22
#define CUEVENTCREATE                   23
#define CUEVENTRECORD                   24
#define CUEVENTSYNCHRONIZE              25
#define CUEVENTELAPSEDTIME              26
#define CUEVENTDESTROY 			27
#define CUMODULEUNLOAD 			28
#define CUMEMALLOCPITCH 		29
#define CUMODULEGETGLOBAL 		30
#define CUSTREAMCREATE 			31
#define CUMEMCPYHTODASYNC 		32
#define CUMEMCPYDTOHASYNC 		33
#define CUMEMCPYDTODASYNC 		34
#define CUSTREAMSYNCHRONIZE 		35
#define CUSTREAMQUERY 			36
#define CUSTREAMDESTROY 		37

#define TYPE_VIRTIO_CUDA "virtio-cuda"

typedef struct VirtCuda {
    VirtIODevice parent_obj;
} VirtCuda;

//same struct as the one used to transfer data between guest user/kernel space - used to transfer between guest driver to backend
struct param {
        CUresult result;
	unsigned int syscall_type;
        size_t bytesize, total_alloc;
	CUdevice device;
	CUcontext ctx;
	CUmodule module;
	CUfunction function;
	CUdeviceptr dptr,dptr1;
	CUdevice_attribute attrib;
	CUevent event1, event2;
	CUstream stream;
	unsigned int flags;
	int val1, val2, length;
	char *name;
	float pMilliseconds;
	unsigned int bytecount, gridDimX, gridDimY, gridDimZ, 
			blockDimX, blockDimY, blockDimZ;
	unsigned long nr_pages;
	void *host;
	void **args;
	size_t *size, size1, size2, size3;
};
#endif
