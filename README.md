# VGVM: Efficient GPU Capabilities in Virtual Machines

VGVM is an open-source framework which enables CUDA applications to execute within Virtual Machines running with QEMU-KVM. Using this framework, multiple VMs co-located in the same host computer can share physical GPU resources, in order to to accelerate their performance.

VGVM uses paravirtualization and API forwarding techniques in order to enable CUDA applications executing in virtual environments to access the physical GPU. 

To summarize, VGVM offers the following advantages:
- It enables GPU resource sharing among co-located VMs.
- It maintains binary compatibility, so that existing applications can use our framework without any source code modification.

## How to use VGVM

### Load KVM Modules

Install required packages:

```
apt-get install gcc make vim socat bzip2
apt-get install zlib1g-dev libglib2.0-dev xtightvncviewer
apt-get install autoconf libtool flex bison
apt-get install libncurses5 -dev
```

Check for CPU virtualization extensions:

```
egrep '(vmx|svm)' /proc/cpuinfo
```

### Install VGVM backend

Dowload QEMU source code version 2.3.1 from http://wiki.qemu.org/Download, and install using:

```
cd qemu-2.3.1
git clone https://github.com/dimvass/VGVM.git/
patch -p1 < VGVM/virtio_device/qemu-2.3.1_vgvm.patch
./configure --prefix=/path/to/install/dir --enable-kvm --extra-cflags=-I/usr/local/cuda/include --disable-rdma --target-list=x86_64-softmmu
make
make install
```
### Launch a QEMU Virtual Machine

Set an environment variable named ```QEMU_NFS_PATH``` to a directory where CUDA object files (.cubin files) will be stored in order to be loaded to the GPU. CUDA object files can be accessible through a shared filesystem or simply be transferred through scp. 

Then, execute QEMU using the ```-device virtio-cuda-pci``` option. For example a VM can be launched using a raw hard disk image name disk_image.raw by executing:

```
export QEMU_NFS_PATH="/path/to/shared/directory"
/path/to/install/dir/bin/qemu-system-x86_64 -enable-kvm -M pc-0.12 -m 1024 -smp 2 -drive file=disk_image.raw,format=raw,if=virtio -net nic -net user, hostfwd=tcp:127.0.0.1:22223-:22 -vnc 127.0.0.1:0 -nographic -monitor /dev/null -device virtio-cuda-pci
```
The VM can be accessed through ssh : ```ssh -p 22223 <user>@localhost```

### Install VGVM frontend

Download and install CUDA Toolkit 5.0 from https://developer.nvidia.com/cuda-toolkit-50-archive.

Then install VGVM frontend by executing:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-5.0/lib64:/usr/local/cuda-5.0/lib
export PATH=$PATH:/usr/local/cuda-5.0/bin
apt-get update
apt-get install linux-headers-$(uname -r)
git clone https://github.com/dimvass/VGVM.git/
apt-get install gcc-4.4
apt-get install g++-4.4
ln -s /usr/bin/gcc-4.4 /usr/local/cuda/bin/gcc
ln -s /usr/bin/g++-4.4 /usr/local/cuda/bin/g++
cd VGVM/frontend_driver/
make
insmod virtio_cuda.ko
./nodes.sh
cd ../library/
make
```

under construction
