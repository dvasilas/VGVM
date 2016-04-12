CC	= gcc -Wall
NVCC	= nvcc --compiler-bindir=/usr/local/cuda/bin -arch sm_20 --keep
CFLAGS  = -I /usr/local/cuda/include

all: libvrtcuda.so
libvrtcuda.so: libvrtcuda.o
	$(CC) $(CFLAGS) -shared -ldl libvrtcuda.o -o $@
libvrtcuda.o: libvrtcuda.c
	$(CC) -fPIC $(CFLAGS) -c $< -o $@
clean:
	rm -f *.so *.o

