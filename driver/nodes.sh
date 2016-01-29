#!/bin/bash

nr_devices=32

for i in $(seq 0 1 ${nr_devices}); do
	mknod /dev/cuda$i c 60 $i
done
