#ifndef _CUDACALLS_UTIL_H
#define _CUDACALLS_UTIL_H

#include <cuda.h>
#include "/root/src/driver/cuda-chrdev.h"

//list that stores function launch arguments
struct argmnt_data {
	const void *arg;
	size_t size;
	struct argmnt_data *next;
};
//struct to store grid/block sizes
//containts the struct argmnt_data list with the rest of the launch data
struct launch_data {
	int argnum;
	int blockDimX;
        int blockDimY;
        int blockDimZ;
	int gridDimX;
        int gridDimY;
        int gridDimZ;
	unsigned int sharedmem;
	struct argmnt_data *args;
};
//list that stores all .cubin file names in the current directories
struct bin_data {
	char *binname;
	int length;
	int loaded;
	CUmodule module;
	struct bin_data *next;
};
//list that stores function names and handlers - points to list node containing .cubin filename where it is defined
struct func_data {
	char *funcname;
	int length;
	CUfunction function;
	struct bin_data *bin;
	struct func_data *next;
};
//list that stores internal representation of function - points to list note containing functnion name/handler
struct hostf_data {
	const char *hostfun;
	struct func_data *devicefun;
	struct hostf_data *next;
};
//struct to store pointers to all required lists + other data
struct lib_data {
        int fd;
        int isinitialized;
        struct bin_data *cubins;
        struct func_data *funcs;
        struct hostf_data *hostfuncs;
        struct launch_data launchconf;
        struct argmnt_data *args;
        struct param p;
};

struct hostf_data *funcToBinary(const void *f, struct lib_data libd);
void registerCudaKernels(struct lib_data *libd);
void freeMem(struct lib_data libd);
#endif
