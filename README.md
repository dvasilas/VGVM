# VGVM: Efficient GPU Capabilities in Virtual Machines

VGVM is an open-source framework which enables CUDA applications to execute within Virtual Machines running with QEMU-KVM. Using this framework, multiple VMs co-located in the same host computer can share physical GPU resources, in order to to accelerate their performance.

VGVM uses paravirtualization and API forwarding techniques in order to enable CUDA applications executing in virtual environments to access the physical GPU. 

To summarize, VGVM offers the following advantages:
- It enables GPU resource sharing among co-located VMs.
- It maintains binary compatibility, so that existing applications can use our framework without any source code modification.

This work has been published in the 2016 International Conference on High Performance Computing & Simulation (HPCS):
http://ieeexplore.ieee.org/document/7568395/
