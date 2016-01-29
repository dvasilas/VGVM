#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "libvrtcuda.h"
#include <string.h>
#include <sys/time.h>

struct lib_data libdata;
//not a very secure way to store each process' CUcontext
//needs improvement
CUcontext gctx;

//initialization required before the first CUDA call of an application
cudaError_t initialize() {
	CUdevice dev;
	//CUresult cuInit(unsigned int Flags)
        libdata.p.flags = 0;
        if (ioctl(libdata.fd,CUINIT, &libdata.p)) {
                perror("ioctl");
		exit(1);
        }
	if (libdata.p.result != CUDA_SUCCESS) 
		goto out;

	//CUresult cuDeviceGet(CUdevice *device, int ordinal)	
        libdata.p.val1 = 0;
        if (ioctl(libdata.fd,CUDEVICEGET, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
        dev = libdata.p.device;
	if (libdata.p.result != CUDA_SUCCESS)
		goto out;

	//CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
        libdata.p.flags = 0;
        libdata.p.device = dev;
        if(ioctl(libdata.fd,CUCTXCREATE, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
	gctx = libdata.p.ctx;
out:
	return libdata.p.result;
}


cudaError_t cudaGetLastError (void) {
	return libdata.p.result;
}

cudaError_t cudaSetDevice (int device) {
	//TODO cuDeviceGet(ordinal) -> cuCtxSetCurrent()
	return cudaSuccess;
}

cudaError_t cudaDeviceReset (void) {
	//CUresult cuCtxDestroy(CUcontext ctx)
        libdata.p.ctx = gctx;
        if(ioctl(libdata.fd,CUCTXDESTROY, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
	return libdata.p.result;
}

cudaError_t cudaGetDeviceProperties (struct cudaDeviceProp *prop, int device) {
        cudaError_t ret;
        if (!libdata.isinitialized) {
                ret = initialize();
                if (ret != cudaSuccess)
                        return ret;
                libdata.isinitialized = 1;
        }
	//CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
        libdata.p.device = (CUdevice) device;
        if (ioctl(libdata.fd,CUDEVICECOMPUTECAPABILITY, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
	if (libdata.p.result != CUDA_SUCCESS)
		goto out;
	prop->major = libdata.p.val1;
        prop->minor = libdata.p.val2;

	//CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
	libdata.p.val1 = 20;
        libdata.p.name = malloc(sizeof(char)*libdata.p.val1);
        libdata.p.device = (CUdevice) device;
        if (ioctl(libdata.fd,CUDEVICEGETNAME, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
        if (libdata.p.result != CUDA_SUCCESS)
                goto out;	
	strcpy(prop->name, libdata.p.name);
	free(libdata.p.name);

	//CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)		
	CUdevice_attribute attrib = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
	libdata.p.attrib = attrib;
	libdata.p.device = (CUdevice) device;
        if (ioctl(libdata.fd,CUDEVICEGETATTRIBUTE, &libdata.p)) {
                perror("ioctl");
        }
        if (libdata.p.result != CUDA_SUCCESS)
                goto out;
	prop->multiProcessorCount = libdata.p.val1;
	
out:
	return libdata.p.result;
}

cudaError_t cudaGetDevice (int *device) {
	//maybe change to cuCtxGetDevice
	//CUresult cuDeviceGet(CUdevice *device, int ordinal)
	libdata.p.val1 = 0;
        if (ioctl(libdata.fd,CUDEVICEGET, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
	*device = (int) libdata.p.device;
        return libdata.p.result;
}

cudaError_t cudaGetDeviceCount	(int *count) {
	//CUresult cuDeviceGetCount (int *count)
        if(ioctl(libdata.fd,CUDEVICEGETCOUNT, &libdata.p)) {
	        perror("ioctl");
		exit(1);
        }
        *count = libdata.p.val1;
	return libdata.p.result;
}
cudaError_t cudaMalloc(void **devPtr, size_t size) {
        cudaError_t ret;
        if (!libdata.isinitialized) {
                ret = initialize();
                if (ret != cudaSuccess)
                        return ret;
		libdata.isinitialized = 1;
        }
	//CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
	libdata.p.ctx = gctx;
	libdata.p.bytesize = size;
        if(ioctl(libdata.fd,CUMEMALLOC, &libdata.p)) {
                perror("ioctl");
		exit(1);
        }
	*devPtr = (void *) libdata.p.dptr;
	return libdata.p.result;
}
cudaError_t cudaHostAlloc (void **pHost, size_t size, unsigned int flags) {
	cudaError_t ret;
	if (!libdata.isinitialized) {
		ret = initialize();
		if (ret != cudaSuccess)
			return ret;
		libdata.isinitialized = 1;
	}
	//CUresult cuMemAllocHost(void **pp, size_t bytesize)
	libdata.p.bytesize = size;
	if(ioctl(libdata.fd,CUMEMALLOCHOST, &libdata.p)) {
		perror("ioctl");
	}
        *pHost = (void *) libdata.p.host;
	return libdata.p.result;
}

cudaError_t cudaMemset (void *devPtr, int value, size_t count) {
        libdata.p.dptr = (CUdeviceptr) devPtr;
	libdata.p.bytecount = value;
	//only works for float for now
	//CUresult cuMemsetD32  (CUdeviceptr dstDevice, unsigned int ui, size_t N)
	libdata.p.bytesize = count/sizeof(float);
        if(ioctl(libdata.fd,CUMEMSETD32, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
        return libdata.p.result;
}
cudaError_t cudaFree (void *devPtr) {
	//CUresult cuMemFree (CUdeviceptr dptr)
	libdata.p.dptr = (CUdeviceptr) devPtr;
        if(ioctl(libdata.fd,CUMEMFREE, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
	return libdata.p.result;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
        cudaError_t ret;
        if (!libdata.isinitialized) {
                ret = initialize();
                if (ret != cudaSuccess)
                        return ret;
		libdata.isinitialized = 1;
        }
	libdata.p.ctx = gctx;
	if (kind == cudaMemcpyHostToDevice) {
		//CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
		libdata.p.host = (void *)src;
		libdata.p.dptr = (CUdeviceptr) dst;
		libdata.p.bytesize = count;
		if(ioctl(libdata.fd,CUMEMCPYHTODV2, &libdata.p)) {
	                perror("ioctl");
	                exit(1);
	        }
	}
	else if (kind == cudaMemcpyDeviceToHost) {
		//CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
                libdata.p.host = dst;
		libdata.p.dptr = (CUdeviceptr) src;
                libdata.p.bytesize = count;	
		if(ioctl(libdata.fd,CUMEMCPYDTOHV2, &libdata.p)) {
                        perror("ioctl");
	                exit(1);
                }
	}
	else  {
		printf("invalid memcpy kind\n");
		return cudaErrorUnknown;	
	}	
        return libdata.p.result;	
}
//intercepts function name, finds in which .cubin file the function's code is and loads the .cubin (if not already loaded)
//doesn't get function handler yet, it needs to be done right before the launch
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
	struct func_data *tt = libdata.funcs;
        while(tt != NULL) {
		int i;
		for (i=0; i<strlen(deviceFun); i++) {
			if (deviceFun[i] != tt->funcname[i])
				break;
		}
		if (i == strlen(deviceFun)) break;
                tt = tt->next;
        }
	struct hostf_data *n = malloc(sizeof(struct hostf_data));
	n->next = libdata.hostfuncs;
	n->hostfun = hostFun;
	n->devicefun = tt;
	libdata.hostfuncs = n;

        cudaError_t ret;
        if (!libdata.isinitialized) {
                ret = initialize();
                if (ret != cudaSuccess) {
                        printf("initialize error: %d\n",ret);
                        exit(1);
		}             
                libdata.isinitialized = 1;
        }
        struct hostf_data *hf = funcToBinary(hostFun, libdata);
        
	if (!hf->devicefun->bin->loaded) {
		//CUresult cuModuleLoad	(CUmodule *module, const char *fname)
		hf->devicefun->bin->binname[hf->devicefun->bin->length-1] = '\0';
		libdata.p.name = hf->devicefun->bin->binname;
                libdata.p.val1 = hf->devicefun->bin->length;
                if(ioctl(libdata.fd,CUMODULELOAD, &libdata.p)) {
                        perror("ioctl");
	                exit(1);
                }
                if (libdata.p.result != CUDA_SUCCESS) {
			printf("cuModuleLoad error: %d\n",libdata.p.result);
                        exit(1);
		}
                hf->devicefun->bin->module = libdata.p.module;
                struct bin_data *t= libdata.cubins;
                while(t != NULL) {
                        t->loaded = 0;
                        t = t->next;
                }
                hf->devicefun->bin->loaded = 1;
	}
}

//intercepts launch arguments (grid/block size) and stores them in structure
//they are retrieved during launch
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
	libdata.launchconf.args = NULL;
	libdata.launchconf.argnum = 0;
	libdata.launchconf.blockDimX = blockDim.x;
	libdata.launchconf.blockDimY = blockDim.y;
	libdata.launchconf.blockDimZ = blockDim.z;
	libdata.launchconf.gridDimX = gridDim.x;
	libdata.launchconf.gridDimY = gridDim.y;
	libdata.launchconf.gridDimZ = gridDim.z;
	libdata.launchconf.sharedmem = (unsigned int) sharedMem;
        return cudaSuccess;
}
//intercepts rest of launch arguments and stores them in list
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	struct argmnt_data *n = malloc(sizeof(struct argmnt_data));
	n->next =NULL;
	n->arg = arg;
	n->size = size;
	struct argmnt_data *t = libdata.launchconf.args;
	if (t == NULL) 
		libdata.launchconf.args = n;
	else {
		while(t->next != NULL) t = t->next;
		t->next = n;
	}
	libdata.launchconf.argnum++;
        return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func) {
	int i;
	cudaError_t ret;

	if (!libdata.isinitialized) {
                ret = initialize();
                if (ret != cudaSuccess)
                        return ret;
		libdata.isinitialized = 1;
        }

	struct hostf_data *hf = funcToBinary(func, libdata);
	
	if (hf->devicefun->function == NULL) {
		//CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)		
		libdata.p.module = hf->devicefun->bin->module;
		libdata.p.name = hf->devicefun->funcname;
		libdata.p.length = hf->devicefun->length;
		if(ioctl(libdata.fd,CUMODULEGETFUNCTION, &libdata.p)) {
	                perror("ioctl");
	                exit(1);
	        }
		if (libdata.p.result != CUDA_SUCCESS)
	                goto out;
		hf->devicefun->function = libdata.p.function;
	}
/*
	CUresult cuLaunchKernel(CUfunction f,
				unsigned int gridDimX,
				unsigned int gridDimY,
				unsigned int gridDimZ,
				unsigned int blockDimX,
				unsigned int blockDimY,
				unsigned int blockDimZ,
				unsigned int sharedMemBytes,
				CUstream hStream,
				void ** kernelParams,
				voidu** extra)	
*/
        libdata.p.ctx = gctx;
	libdata.p.function = hf->devicefun->function;
	libdata.p.gridDimX = libdata.launchconf.gridDimX;
	libdata.p.gridDimY = libdata.launchconf.gridDimY;
	libdata.p.gridDimZ = libdata.launchconf.gridDimZ;
	libdata.p.blockDimX = libdata.launchconf.blockDimX;
	libdata.p.blockDimY = libdata.launchconf.blockDimY;
	libdata.p.blockDimZ = libdata.launchconf.blockDimZ;
	libdata.p.bytecount = libdata.launchconf.sharedmem;
	libdata.p.val1 = libdata.launchconf.argnum;
	libdata.p.args = malloc(libdata.p.val1*sizeof(void *));
	libdata.p.size = malloc(libdata.p.val1*sizeof(size_t));
	struct argmnt_data *args = libdata.launchconf.args;
	for (i=0; i<libdata.p.val1; i++) {
		libdata.p.args[i] = (void *)args->arg;
		libdata.p.size[i] = args->size;
		args = args->next;
	}

	if(ioctl(libdata.fd,CULAUNCHKERNEL, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
	free(libdata.p.args);
	free(libdata.p.size);
out:	
	return libdata.p.result;
}

cudaError_t cudaEventCreate (cudaEvent_t *event) {
	//CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags)
        libdata.p.flags = CU_EVENT_DEFAULT;
        if(ioctl(libdata.fd, CUEVENTCREATE, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
        *event = libdata.p.event1;
        return libdata.p.result;
}
cudaError_t cudaEventDestroy (cudaEvent_t event) {
	//CUresult cuEventDestroy(CUevent hEvent)
	libdata.p.event1 = event;
	if(ioctl(libdata.fd, CUEVENTDESTROY, &libdata.p)) {
		perror("ioctl");
		exit(1);
	}
	return libdata.p.result;
}
cudaError_t cudaEventRecord (cudaEvent_t event, cudaStream_t stream) {
	//CUresult cuEventRecord(CUevent hEvent, CUstream hStream)		
        libdata.p.event1 = event;
        libdata.p.stream = stream;
        if(ioctl(libdata.fd,CUEVENTRECORD, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
        return libdata.p.result;
}
cudaError_t cudaEventSynchronize (cudaEvent_t event) {
	//CUresult cuEventSynchronize(CUevent hEvent)
        libdata.p.event1 = event;
        if(ioctl(libdata.fd,CUEVENTSYNCHRONIZE, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
        return libdata.p.result;
}
cudaError_t cudaEventElapsedTime (float *ms, cudaEvent_t start, cudaEvent_t end) {
	//CUresult cuEventElapsedTime (float *pMilliseconds, CUevent hStart, CUevent hEnd)	
        libdata.p.event1 = start;
        libdata.p.event2 = end;
        if(ioctl(libdata.fd,CUEVENTELAPSEDTIME, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
        *ms = libdata.p.pMilliseconds;
        return libdata.p.result;
}
cudaError_t cudaDeviceSynchronize (void) {
	//CUresult cuCtxSynchronize (void)
        if(ioctl(libdata.fd,CUCTXSYNCHRONIZE, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
	return libdata.p.result;
}

void __cudaUnregisterFatBinary() {
	//CUresult cuCtxSynchronize(void)
	//not supported for the moment
/*
        if(ioctl(libdata.fd,CUMODULEUNLOAD, &libdata.p)) {
                perror("ioctl");
                exit(1);
        }
*/
	freeMem(libdata);
}
//undocumented - can't know what void**
//works if the actual routine is called using dlsym, although the routine must file internally 
void** __cudaRegisterFatBinary(void *fatCubin) {

	registerCudaKernels(&libdata);

        libdata.fd = open("/dev/cuda0", O_RDWR, 0);
        if (libdata.fd < 0) {
                perror("open");
                exit(1);
        }
	libdata.isinitialized = 0;

        char hostname[50];
        gethostname(hostname, 50*sizeof(char));
	libdata.p.id = hostname[strlen(hostname)-1-'0'];

        void *handle = dlopen ("/usr/local/cuda/lib64/libcudart.so.5.0", RTLD_LAZY);
        if (!handle) {
                fputs (dlerror(), stdout);
                exit(1);
        }
        void ** (*__cudaRegisterFatBinary)(void *fatCubin);
        char *error;
        __cudaRegisterFatBinary = dlsym(handle, "__cudaRegisterFatBinary");
        if ((error = dlerror()) != NULL)  {
                fputs(error, stdout);
                exit(1);
        }
        void** ret = __cudaRegisterFatBinary(fatCubin);
        return ret;	
}

//searches list for function name and return node
//calling function retrieves .cubin name from node
struct hostf_data *funcToBinary(const void *f, struct lib_data libd) {
        struct hostf_data *hf = libd.hostfuncs;
        while(hf != NULL) {
                if (f == hf->hostfun)
                        return hf;
                hf = hf->next;
        }
        return NULL;
}
//deallocate all memory allocated to store arguments and function/.cubin names
void freeMem(struct lib_data libd) {
        {
                struct argmnt_data *tn, *t = libd.args;
                while(t != NULL) {
                        tn = t;
                        t = t->next;
                        free(tn);
                }
        }

        {
                struct bin_data *tn, *t = libd.cubins;
                while(t != NULL) {
                        tn = t;
                        t = t->next;
                        free(tn->binname);
                        free(tn);
                }
        }
        {
                struct func_data *tn, *t = libd.funcs;
                while(t != NULL) {
                        tn = t;
                        t = t->next;
                        free(tn->funcname);
                        free(tn);
                }
        }
        {
                struct hostf_data *tn, *t = libd.hostfuncs;
                while(t != NULL) {
                        tn = t;
                        t = t->next;
                        free(tn);
                }
        }
}
