#ifndef _CUDA_DRV_H
#define _CUDA_DRV_H

/*
Two implementation when waiting backend to execute routine and return results
sleep : interruptible state until intrrupt is received
busy_wait : check if a buffer is pushed avaible in a loop
*/

//#define SLEEP
#define BUSY_WAIT

#define SYSCALL_OPEN  0
#define SYSCALL_CLOSE 1
#define SYSCALL_IOCTL 2

#define VIRTIO_ID_CUDA 13

struct cuda_driver_data {
	struct list_head devs;
	unsigned int next_minor;
	spinlock_t lock;
	wait_queue_head_t wq;
};
extern struct cuda_driver_data cudrvdata;

struct cuda_device {
	struct list_head list;
	struct virtio_device *vdev;
	struct virtqueue *vq;
	spinlock_t lock;
	struct semaphore slock;
	unsigned int minor;
};

struct cuda_open_file {
	struct cuda_device *cudev;
	int host_fd;
};
struct wait_struct {
        struct list_head list;
	void *data;
	spinlock_t lock;
};
extern struct wait_struct wait_data;
#endif
