#include "hw/virtio/virtio-serial.h"
#include "hw/virtio/virtio-cuda.h"
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <cuda.h>
#include <dlfcn.h>

static uint32_t get_features(VirtIODevice *vdev, uint32_t features)
{
	return features;
}

static void get_config(VirtIODevice *vdev, uint8_t *config_data)
{
}

static void set_config(VirtIODevice *vdev, const uint8_t *config_data)
{
}

static void set_status(VirtIODevice *vdev, uint8_t status)
{
}

static void vser_reset(VirtIODevice *vdev)
{
}

static void vq_handle_output(VirtIODevice *vdev, VirtQueue *vq)
{
	VirtQueueElement elem;
	
	while(virtqueue_pop(vq, &elem)) {
		struct param *p = elem.out_sg[0].iov_base;
	
		//for all library routines: get required arguments from buffer, execute, and push results back in virtqueue
		switch (p->syscall_type) {
		case CUINIT: {
			p->result = cuInit(p->flags);
			break;
		}
		case CUDRIVERGETVERSION: {
			p->result = cuDriverGetVersion(&p->val1);
			break;
		}
		case CUDEVICEGETCOUNT: {
			p->result = cuDeviceGetCount(&p->val1);
			break;
		}
		case CUDEVICEGET: {
			p->result = cuDeviceGet(&p->device, p->val1);
			break;
		}
		case CUDEVICECOMPUTECAPABILITY: {
			p->result = cuDeviceComputeCapability(&p->val1, &p->val2, p->device);
			break;
		}
		case CUDEVICEGETNAME: {
			p->result = cuDeviceGetName(elem.in_sg[0].iov_base, p->val1, p->device);
			break;
		}
		case CUDEVICEGETATTRIBUTE: {
			p->result = cuDeviceGetAttribute(&p->val1, p->attrib, p->device);
			break;
		}
		case CUCTXCREATE: {
                        p->result = cuCtxCreate(&p->ctx, p->flags, p->device);				
			break;
		}
		case CUCTXDESTROY: {
			p->result = cuCtxDestroy(p->ctx);
			break;
		}
		case CUCTXGETCURRENT: {
			p->result = cuCtxGetCurrent(&p->ctx);
			break;
		}
		case CUCTXGETDEVICE: {
			p->result = cuCtxGetDevice(&p->device);
			break;
		}
		case CUCTXPOPCURRENT: {
			p->result = cuCtxPopCurrent(&p->ctx);
			break;
		}
		case CUCTXSETCURRENT: {
			p->result = cuCtxSetCurrent(p->ctx);
	                break;
		}
	        case CUCTXSYNCHRONIZE: {
		        p->result = cuCtxSynchronize();
	                break;
	        }
		case CUMODULELOAD: {
			//hardcoded path - needs improvement
			//all .cubin files should be stored in $QEMU_NFS_PATH - currently $QEMU_NFS_PATH is shared between host and guest with NFS
			char *binname = malloc((strlen((char *)elem.out_sg[1].iov_base)+strlen(getenv("QEMU_NFS_PATH")+1))*sizeof(char));
			if (!binname) {
				p->result = 0;
		                virtqueue_push(vq, &elem, 0);
				break;
			}
		        strcpy(binname, getenv("QEMU_NFS_PATH"));
		        strcat(binname, (char *)elem.out_sg[1].iov_base);
			//change current CUDA context
			//each CUDA contets has its own virtual memory space - isolation is ensured by switching contexes
                        if (cuCtxSetCurrent(p->ctx) != 0) {
				p->result = 999;
                                break;
			}
			p->result = cuModuleLoad(&p->module, binname);
			free(binname);
			break;
		}
                case CUMODULEGETGLOBAL: {
                        char *name = malloc(100*sizeof(char));
                        if (!name) {
                                p->result = 999;
                                break;
                        }
                        strcpy(name, (char *)elem.out_sg[1].iov_base);
                        p->result = cuModuleGetGlobal(&p->dptr,&p->size1,p->module,(const char *)name);
                        break;
                }
		case CUMODULEUNLOAD: {
			p->result = cuModuleUnload(p->module);
			break;			
		}
		case CUMEMALLOC: {
			if (cuCtxSetCurrent(p->ctx) != 0) {
                                p->result = 999;
                                break;
                        }
			p->result = cuMemAlloc(&p->dptr, p->bytesize);
			break;
		}
                case CUMEMALLOCPITCH: {
                        if (cuCtxSetCurrent(p->ctx) != 0) {
                                p->result = 999;
                                break;
                        }
			p->result = cuMemAllocPitch(&p->dptr, &p->size3, p->size1, p->size2, p->bytesize);
			break;
		}
		//large buffers are alocated in smaller chuncks in guest kernel space
		//gets each chunck seperately and copies it to device memory
	        case CUMEMCPYHTOD: {
			int i;
			size_t offset;
                        unsigned long s, nr_pages = p->nr_pages;
                        if (cuCtxSetCurrent(p->ctx) != 0) {
                                p->result = 999;
                                break;
                        }
			offset = 0;
			for (i=0; i<nr_pages; i++) {
				s = *(long *)elem.out_sg[1+2*i+1].iov_base;
				p->result = cuMemcpyHtoD(p->dptr+offset, elem.out_sg[1+2*i].iov_base, s);
				if (p->result != 0) break;
				offset += s;
			}
	                break;
		}
		case CUMEMCPYHTODASYNC: {
			int i;
                        size_t offset;
                        unsigned long s, nr_pages = p->nr_pages;
                        if (cuCtxSetCurrent(p->ctx) != 0) {
                                p->result = 999;
                                break;
                        }
                        offset = 0;
			for (i=0; i<nr_pages; i++) {
                                s = *(long *)elem.out_sg[1+2*i+1].iov_base;
                                p->result = cuMemcpyHtoDAsync(p->dptr+offset, elem.out_sg[1+2*i].iov_base, s, p->stream);
                                if (p->result != 0) break;
                                offset += s;
                        }
                        break;
		}
		case CUMEMCPYDTODASYNC: {
			p->result = cuMemcpyDtoDAsync(p->dptr, p->dptr1, p->size1, p->stream);
                        break;		
		}
	        case CUMEMCPYDTOH: {
			int i;
                        unsigned long s, nr_pages = p->nr_pages;
                        if (cuCtxSetCurrent(p->ctx) != 0) {
                                p->result = 999;
                                break;
                        }
			size_t offset = 0;
			for (i=0; i<nr_pages; i++) {
				s = *(long *)elem.in_sg[0+2*i+1].iov_base;
				p->result = cuMemcpyDtoH(elem.in_sg[0+2*i].iov_base, p->dptr+offset, s);
				if (p->result != 0) break;
				offset += s;
			}
			break;
		}
		case CUMEMCPYDTOHASYNC: {
			int i;
                        unsigned long s, nr_pages = p->nr_pages;
                        if (cuCtxSetCurrent(p->ctx) != 0) {
                                p->result = 999;
                                break;
                        }
                        size_t offset = 0;
			for (i=0; i<nr_pages; i++) {
                                s = *(long *)elem.in_sg[0+2*i+1].iov_base;
                                p->result = cuMemcpyDtoHAsync(elem.in_sg[0+2*i].iov_base, p->dptr+offset, s, p->stream);
                                if (p->result != 0) break;
                                offset += s;
                        }
			break;
		}
		case CUMEMSETD32: {
			p->result = cuMemsetD32(p->dptr, p->bytecount, p->bytesize);
			break;
		}
	        case CUMEMFREE: {
	                p->result = cuMemFree(p->dptr);
	                break;
	        }
		case CUMODULEGETFUNCTION: {
			char *name = (char *)elem.out_sg[1].iov_base;
			name[p->length] = '\0';
                        if (cuCtxSetCurrent(p->ctx) != 0) {
                                p->result = 999;
                                break;
                        }
			p->result = cuModuleGetFunction(&p->function, p->module, name);
			break;	
		}
		case CULAUNCHKERNEL: {
			void **args = malloc(p->val1*sizeof(void *));
	                if (!args) {
				p->result = 9999;
	                        break;
        	        }
			int i;
			for (i=0; i<p->val1; i++) {
				args[i] = elem.out_sg[1+i].iov_base;
			}
                        if (cuCtxSetCurrent(p->ctx) != 0) {
                                p->result = 999;
                                break;
                        }
			p->result = cuLaunchKernel(p->function,
					p->gridDimX, p->gridDimY, p->gridDimZ,
			                p->blockDimX, p->blockDimY, p->blockDimZ,
					p->bytecount, 0, args, 0);
			free(args);
			break;
		}
		case CUEVENTCREATE: {
			p->result = cuEventCreate(&p->event1, p->flags);
			break;
		}
		case CUEVENTDESTROY: {
			p->result = cuEventDestroy(p->event1);
			break;
		}
		case CUEVENTRECORD: {
			p->result = cuEventRecord(p->event1, p->stream);
			break;
		}
		case CUEVENTSYNCHRONIZE: {
			p->result = cuEventSynchronize(p->event1);
			break;
		}
		case CUEVENTELAPSEDTIME: {
			p->result = cuEventElapsedTime(&p->pMilliseconds, p->event1, p->event2);
			break;
		}
		case CUSTREAMCREATE: {
			p->result =  cuStreamCreate(&p->stream, 0);
			break;
		}		
                case CUSTREAMSYNCHRONIZE: {
                        p->result = cuStreamSynchronize(p->stream);
                        break;
                }
                case CUSTREAMQUERY: {
                        p->result = cuStreamQuery(p->stream);
                        break;
                }
		case CUSTREAMDESTROY: {
                        p->result = cuStreamDestroy(p->stream);
                        break;
                }

		default: 
			printf("Unknown syscall_type\n");
		}
		virtqueue_push(vq, &elem, 0);
	}
	//notify frontend - trigger virtual interrupt
	virtio_notify(vdev, vq);
	return;
}

static void virtio_cuda_realize(DeviceState *dev, Error **errp)
{
	VirtIODevice *vdev = VIRTIO_DEVICE(dev);

	virtio_init(vdev, "virtio-cuda", 13, 0);
	virtio_add_queue(vdev, 1024, vq_handle_output);
}

static void virtio_cuda_unrealize(DeviceState *dev, Error **errp)
{
}

static Property virtio_cuda_properties[] = {
	DEFINE_PROP_END_OF_LIST(),
};

static void virtio_cuda_class_init(ObjectClass *klass, void *data)
{
	DeviceClass *dc = DEVICE_CLASS(klass);
	VirtioDeviceClass *k = VIRTIO_DEVICE_CLASS(klass);

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
