CC = gcc

USER_CFLAGS = -Wall

KERNEL_VERBOSE = 'V=1'
DEBUG = y

EXTRA_CFLAGS += -Wno-unused-variable

obj-m := virtio_cuda.o
virtio_cuda-objs := cuda-module.o cuda-chrdev.o

all: modules

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

modules:
	make -C $(KERNELDIR) M=$(PWD) modules

clean:
	make -C $(KERNELDIR) M=$(PWD) clean
