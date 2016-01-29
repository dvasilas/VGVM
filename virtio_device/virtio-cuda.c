#include "hw/virtio/virtio-serial.h"
#include "hw/virtio/virtio-cuda.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <cuda.h>
#include <dlfcn.h>
#include <time.h>
#include <asm/msr.h>

#include <hw/virtio/virtio-trace.h>

#define PATH "/home/users/dimvas/shared/"

static uint32_t get_features(VirtIODevice *vdev, uint32_t features)
{
	DEBUG_IN();
	return features;
}

static void get_config(VirtIODevice *vdev, uint8_t *config_data)
{
	DEBUG_IN();
}

static void set_config(VirtIODevice *vdev, const uint8_t *config_data)
{
	DEBUG_IN();
}

static void set_status(VirtIODevice *vdev, uint8_t status)
{
	DEBUG_IN();
}

static void vser_reset(VirtIODevice *vdev)
{
	DEBUG_IN();
}

static void vq_handle_output(VirtIODevice *vdev, VirtQueue *vq)
{
	VirtQueueElement elem;
	unsigned int syscall_type = 99;
	CUresult res = 999;
	int id;
	
	//pop buffer from virtqueue
	while(virtqueue_pop(vq, &elem)) {
		struct param *par = elem.out_sg[0].iov_base;
	        syscall_type = par->syscall_type;
		//VM id - not used yet - useful for scheduling support
		id = par->id;

		switch (syscall_type) {
		//for all calls : get required arguments from buffer, execute and push results back in virtqueue
		case CUINIT: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuInit(p->flags);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUDRIVERGETVERSION: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuDriverGetVersion(&p->val1);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUDEVICEGETCOUNT: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuDeviceGetCount(&p->val1);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUDEVICEGET: {
                        struct param *p = elem.out_sg[0].iov_base;
			res = cuDeviceGet(&p->device, p->val1);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUDEVICECOMPUTECAPABILITY: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuDeviceComputeCapability(&p->val1, &p->val2, p->device);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUDEVICEGETNAME: {
                        struct param *p = elem.out_sg[0].iov_base;
			res = cuDeviceGetName(elem.in_sg[0].iov_base, p->val1, p->device);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUDEVICEGETATTRIBUTE: {
                        struct param *p = elem.out_sg[0].iov_base;
			res = cuDeviceGetAttribute(&p->val1, p->attrib, p->device);
                        p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUCTXCREATE: {
			res = 0;
                        struct param *p = elem.out_sg[0].iov_base;
                        res = cuCtxCreate(&p->ctx, p->flags, p->device);				
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUCTXDESTROY: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuCtxDestroy(p->ctx);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUCTXGETCURRENT: {
                        struct param *p = elem.out_sg[0].iov_base;
			res = cuCtxGetCurrent(&p->ctx);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUCTXGETDEVICE: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuCtxGetDevice(&p->device);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUCTXPOPCURRENT: {
			struct param *p = elem.out_sg[0].iov_base;			
			res = cuCtxPopCurrent(&p->ctx);
                        p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUCTXSETCURRENT: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuCtxSetCurrent(p->ctx);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
	                break;
		}
	        case CUCTXSYNCHRONIZE: {
			struct param *p = elem.out_sg[0].iov_base;
		        res = cuCtxSynchronize();
			p->result = res;
		        virtqueue_push(vq, &elem, 0);
	                break;
	        }
		case CUMODULELOAD: {
			struct param *p = elem.out_sg[0].iov_base;
			//hardcoded path - needs improvement
			//all .cubin files are in PATH - currently PATH is shared between host and guest with NFS
			char *binname = malloc((strlen((char *)elem.out_sg[1].iov_base)+strlen(PATH)+1)*sizeof(char));
			if (!binname) {
				p->result = 0;
		                virtqueue_push(vq, &elem, 0);
				break;
			}
		        strcpy(binname, PATH);
		        strcat(binname, (char *)elem.out_sg[1].iov_base);
			//changes current CUDA context
			//each CUDA contets has its own virtual memory space - isolation is ensured by switching contexes
                        res = cuCtxSetCurrent(p->ctx);
			res = cuModuleLoad(&p->module, binname);
                        p->result = res;
			free(binname);
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUMODULEUNLOAD: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuModuleUnload(p->module);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;			
		}
		case CUMEMALLOC: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuCtxSetCurrent(p->ctx);
			res = cuMemAlloc(&p->dptr, p->bytesize);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUMEMALLOCHOST: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuMemAllocHost(&p->host, p->bytesize);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;			
		}
		//large buffers are alocated in smaller chuncks in guest kernel space
		//gets each chunck seperately and copies it to device memory
	        case CUMEMCPYHTODV2: {
			int i;
			size_t offset;
                        struct param *p = elem.out_sg[0].iov_base;
                        unsigned long s, nr_pages = p->nr_pages;
			res = cuCtxSetCurrent(p->ctx);
			offset = 0;
			for (i=0; i<nr_pages; i++) {
				s = *(long *)elem.out_sg[1+2*i+1].iov_base;
				res = cuMemcpyHtoD(p->dptr+offset, elem.out_sg[1+2*i].iov_base, s);
				if (res != 0) break;
				offset += s;
			}
			p->result = res;
			virtqueue_push(vq, &elem, 0);
	                break;
		}
	        case CUMEMCPYDTOHV2: {
			int i;
                        struct param *p = elem.out_sg[0].iov_base;
                        unsigned long s, nr_pages = p->nr_pages;
                        res = cuCtxSetCurrent(p->ctx);
			size_t offset = 0;
			for (i=0; i<nr_pages; i++) {
				s = *(long *)elem.in_sg[0+2*i+1].iov_base;
				res = cuMemcpyDtoH(elem.in_sg[0+2*i].iov_base, p->dptr+offset, s);
				if (res != 0) break;
				offset += s;
			}
                        p->result = res;
		        virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUMEMSETD32: {
			struct param *p = elem.out_sg[0].iov_base;
			res =  cuMemsetD32(p->dptr, p->bytecount, p->bytesize);
                        p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
	        case CUMEMFREE: {
			struct param *p = elem.out_sg[0].iov_base;
	                res = cuMemFree(p->dptr);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
	                break;
	        }
		case CUMODULEGETFUNCTION: {
			struct param *p = elem.out_sg[0].iov_base;
			char *name = (char *)elem.out_sg[1].iov_base;
			name[p->length] = '\0';
                        res = cuCtxSetCurrent(p->ctx);
			res = cuModuleGetFunction(&p->function, p->module, name);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;	
		}
		case CULAUNCHKERNEL: {
                        struct param *p = elem.out_sg[0].iov_base;			
			void **args = malloc(p->val1*sizeof(void *));
	                if (!args) {
				p->result = res = 9999;
        	                virtqueue_push(vq, &elem, 0);
	                        break;
        	        }
			int i;
			for (i=0; i<p->val1; i++) {
				args[i] = elem.out_sg[1+i].iov_base;
			}
                        res = cuCtxSetCurrent(p->ctx);
			res = cuLaunchKernel(p->function,
					p->gridDimX, p->gridDimY, p->gridDimZ,
			                p->blockDimX, p->blockDimY, p->blockDimZ,
					p->bytecount, 0, args, 0);
			p->result = res;
			free(args);
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUEVENTCREATE: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuEventCreate(&p->event1, p->flags);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUEVENTDESTROY: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuEventDestroy(p->event1);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUEVENTRECORD: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuEventRecord(p->event1, p->stream);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUEVENTSYNCHRONIZE: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuEventSynchronize(p->event1);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		case CUEVENTELAPSEDTIME: {
			struct param *p = elem.out_sg[0].iov_base;
			res = cuEventElapsedTime(&p->pMilliseconds, p->event1, p->event2);
			p->result = res;
			virtqueue_push(vq, &elem, 0);
			break;
		}
		default:
			printf("Unknown syscall_type\n");
		}

	}
	//notify frontend - trigger virtual interrupt
	virtio_notify(vdev, vq);

	return;
}

static void virtio_cuda_realize(DeviceState *dev, Error **errp)
{
	VirtIODevice *vdev = VIRTIO_DEVICE(dev);

	DEBUG_IN();

	virtio_init(vdev, "virtio-cuda", 13, 0);
	virtio_add_queue(vdev, 1024, vq_handle_output);
}

static void virtio_cuda_unrealize(DeviceState *dev, Error **errp)
{
	DEBUG_IN();
}

static Property virtio_cuda_properties[] = {
	DEFINE_PROP_END_OF_LIST(),
};

static void virtio_cuda_class_init(ObjectClass *klass, void *data)
{
	DeviceClass *dc = DEVICE_CLASS(klass);
	VirtioDeviceClass *k = VIRTIO_DEVICE_CLASS(klass);

	DEBUG_IN();
	dc->props = virtio_cuda_properties;
	set_bit(DEVICE_CATEGORY_INPUT, dc->categories);

	k->realize = virtio_cuda_realize;
	k->unrealize = virtio_cuda_unrealize;
	k->get_features = get_features;
	k->get_config = get_config;
	k->set_config = set_config;
	k->set_status = set_status;
	k->reset = vser_reset;
}

static const TypeInfo virtio_cuda_info = {
	.name          = TYPE_VIRTIO_CUDA,
	.parent        = TYPE_VIRTIO_DEVICE,
	.instance_size = sizeof(VirtCuda),
	.class_init    = virtio_cuda_class_init,
};

static void virtio_cuda_register_types(void)
{
	type_register_static(&virtio_cuda_info);
}

type_init(virtio_cuda_register_types)
