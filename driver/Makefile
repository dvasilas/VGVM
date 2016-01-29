CC = gcc

USER_CFLAGS = -Wall

KERNEL_VERBOSE = 'V=1'
DEBUG = y

EXTRA_CFLAGS += -Wno-unused-variable

ifeq ($(DEBUG),y)
  EXTRA_CFLAGS += -g -DDEBUG=1 -Werror
else
  EXTRA_CFLAGS += -DDEBUG=0 -Werror
endif

obj-m := virtio_cuda.o
virtio_cuda-objs := cuda-module.o cuda-chrdev.o

all: modules test_cuda

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

modules:
	make -C $(KERNELDIR) M=$(PWD) modules

test_cuda: test_cuda.c
	$(CC) $(USER_CFLAGS) -o $@ $^

clean:
	make -C $(KERNELDIR) M=$(PWD) clean
	rm -f test_cuda
