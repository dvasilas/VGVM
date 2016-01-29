#Makefile
CC	= gcc -Wall
NVCC	= nvcc --compiler-bindir=/usr/local/cuda/bin -arch sm_20 --keep
CFLAGS  = -I /usr/local/cuda/include

all: hello-world libvrtcuda.so 
hello-world: hello-world.cu
	$(NVCC) $< -o $@
	rm -f *.so *.o *.ptx *.i* *.cudafe* *.fatbin* *.hash *.module_id
libvrtcuda.so: libvrtcuda-util.o libvrtcuda.o
	$(CC) $(CFLAGS) -shared -ldl libvrtcuda-util.o libvrtcuda.o -o $@
libvrtcuda-util.o: libvrtcuda-util.c
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
libvrtcuda.o: libvrtcuda.c
	$(CC) -fPIC $(CFLAGS) -c $< -o $@
clean:
	rm -f hello-world *.so *.o *.cubin *.ptx *.i* *.cudafe* *.fatbin* *.hash *.module_id

