# VGVM: Efficient GPU Capabilities in Virtual Machines

VGVM is an open-source framework which enables CUDA applications to execute within Virtual Machines running with QEMU-KVM. Using this framework, multiple VMs co-located in the same host computer can share physical GPU resources, in order to to accelerate their performance.

VGVM uses paravirtualization and API forwarding techniques in order to enable CUDA applications executing in virtual environments to access the physical GPU. 

To summarize, VGVM offers the following advantages:
- It enables GPU resource sharing among co-located VMs.
- It maintains binary compatibility, so that existing applications can use our framework without any source code modification.

## How to use VGVM

### Download and install QEMU

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

Execute QEMU using the ```-device virtio-cuda-pci``` option 

under construction
