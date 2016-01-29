#ifndef _CUDA_CHRDEV_H
#define _CUDA_CHRDEV_H

#define CUDA_CHRDEV_MAJOR 60
#define CUDA_NR_DEVICES   32

//used to distinguish between ioctl calls
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
#define CUMEMCPYHTOD1                   17
#define CUMEMCPYHTOD2                   18
#define CUMEMCPYDTOH                    19
#define CUMEMCPYDTOH1                   20
#define CUMEMSETD32                     21
#define CUMEMFREE                       22
#define CUMODULEGETFUNCTION             23
#define CULAUNCHKERNEL                  24
#define CUCTXSYNCHRONIZE                25
#define CUEVENTCREATE                   26
#define CUEVENTRECORD                   27
#define CUEVENTSYNCHRONIZE              28
#define CUEVENTELAPSEDTIME              29
#define CUMEMCPYHTODV2 			30
#define CUMEMCPYDTOHV2 			31
#define CUMEMALLOCHOST	 		32
#define CUEVENTDESTROY 			33
#define CUMODULEUNLOAD 			34

#include "cuda.h"

int cuda_chrdev_init(void);
void cuda_chrdev_destroy(void);

//struct used to transfer data from user to kernel space through ioctl calls
struct param {
        CUresult result;
        unsigned int syscall_type;
        size_t bytesize;
        CUdevice device;
        CUcontext ctx;
        CUmodule module;
        CUfunction function;
        CUdeviceptr dptr;
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
        size_t *size;
	int id;
};
#endif
