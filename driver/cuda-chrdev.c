#include <linux/cdev.h>
#include <linux/poll.h>
#include <linux/sched.h>
#include <linux/module.h>
#include <linux/wait.h>
#include <linux/virtio.h>
#include <linux/virtio_config.h>
#include <linux/slab.h>
#include <linux/pagemap.h>
#include <linux/delay.h>
#include <cuda.h>
#include "cuda-util.h"
#include "cuda-chrdev.h"
#include "debug.h"
#include <asm/msr.h>

struct cdev cuda_chrdev_cdev;
extern struct cuda_driver_data cudrvdata;
struct wait_struct wait_data;
struct memcpy_data {
        struct list_head list;
	void *data;
	long size;
};
//called when process wakes up
//checks if the buffer we are waiting for is in the list and deletes it
#ifdef SLEEP
static int check_vqueue_buff(void * data) {
		struct list_head *pos, *q;
		struct wait_struct *wstr, *tmp;
		unsigned long flags;
		spin_lock_irqsave(&wait_data.lock, flags);
		list_for_each_entry_safe(wstr, tmp, &wait_data.list, list) {
			if (wstr->data == data) {
		                list_del(&wstr->list);
		                kfree(wstr);
				spin_unlock_irqrestore(&wait_data.lock, flags);
				return 1;
		        }
		}
		spin_unlock_irqrestore(&wait_data.lock, flags);
		return 0;
}
#endif
static struct cuda_device *get_cuda_dev_by_minor(unsigned int minor)
{
	struct cuda_device *cudev;
	unsigned long flags;

	spin_lock_irqsave(&cudrvdata.lock, flags);
	list_for_each_entry(cudev, &cudrvdata.devs, list) {
		if (cudev->minor == minor)
			goto out;
	}
	cudev = NULL;

out:
	spin_unlock_irqrestore(&cudrvdata.lock, flags);

	return cudev;
}

static int cuda_chrdev_open(struct inode *inode, struct file *filp)
{
	int ret = 0;
	int err;
	unsigned int len;
	struct cuda_open_file *crof;
	struct cuda_device *cudev;

	ret = -ENODEV;
	if ((ret = nonseekable_open(inode, filp)) < 0)
		goto fail;

	cudev = get_cuda_dev_by_minor(iminor(inode));
	if (!cudev) {
		debug("Could not find device with %u minor", 
		      iminor(inode));
		ret = -ENODEV;
		goto fail;
	}

	crof = kzalloc(sizeof(*crof), GFP_KERNEL);
	if (!crof) {
		ret = -ENOMEM;
		goto fail;
	}
	crof->cudev = cudev;
	filp->private_data = crof;
		
fail:
	return ret;
}

static int cuda_chrdev_release(struct inode *inode, struct file *filp)
{
	int ret = 0;
	struct cuda_open_file *crof = filp->private_data;
	struct cuda_device *cudev = crof->cudev;

	kfree(crof);

	return ret;

}

static long cuda_chrdev_ioctl(struct file *filp, unsigned int cmd, 
                                unsigned long arg)
{
	int err, i, j, flag = 0;
        unsigned int num_out, num_in, len;
        long ret = 0, max_s = 4*1024*1024, min_s = 4*1024;
        unsigned long flags;
        size_t *size = NULL, tocopy, offset, s;
        void *mem = NULL, **args = NULL;
        char *name = NULL;
	struct cuda_open_file *crof = filp->private_data;
	struct cuda_device *cudev = crof->cudev;
	struct virtqueue *vq = cudev->vq;
	struct scatterlist input_sg, output_sg, output_sg1, *output_sg_arg = NULL,
	                   **sgs = NULL, *sg[2];
	struct param user_p;
	struct memcpy_data mcd, *tmp, *tmcd;

	num_out = 0;
	num_in = 0;

	//copy arguments from userspace;
	if (copy_from_user((void *)&user_p, (void *)arg, sizeof(struct param))) {
		ret = -EACCES;
		goto out;
	}
	user_p.syscall_type = cmd;

	switch (cmd) {
	//allocate memory for device name and initialize scatterlist
	case CUDEVICEGETNAME:
		name = kzalloc(sizeof(char)*user_p.val1, GFP_KERNEL);
		if (!name) {
	                ret = -ENOMEM;
	                goto out;
        	}
		sg_init_one(&output_sg, &user_p, sizeof(user_p));
		sg[num_out++] = &output_sg;		
		sg_init_one(&input_sg, name, sizeof(char *));
                sg[num_out + num_in++] = &input_sg;
		break;
	//allocate memory for filename and copy string from user space
	case CUMODULELOAD:
		name = kzalloc(user_p.val1*sizeof(char), GFP_KERNEL);
		if (!name) {
	                ret = -ENOMEM;
	                goto out;
	        }
                if (copy_from_user(name, (void *)user_p.name, user_p.val1*sizeof(char))) {
                        ret = -EACCES;
			goto out;
		}
		sg_init_one(&output_sg, &user_p, sizeof(user_p));
		sg[num_out++] = &output_sg;
		sg_init_one(&output_sg1, name, sizeof(char *));
                sg[num_out++] = &output_sg1;
		break;
	//normally allocate memory for the buffer which is to be copied to the device and copy data from user space
	//we need kmalloc (not vmalloc) because virtqueue works with physically contigious memory
	//kmalloc allocates 4KB max
	//for bigger sizes allocate memory in chunks and create linked list
	//allocate and initialize one scatterlist for each allocated buffer
        case CUMEMCPYHTODV2:
	        INIT_LIST_HEAD(&mcd.list);
		tocopy = user_p.bytesize;
		offset = 0;
		user_p.nr_pages = 0;
		while(tocopy > 0) {
			tmp = kmalloc(sizeof(struct memcpy_data *), GFP_KERNEL);
			if (!tmp) {
				ret = -ENOMEM;
				goto out;
			}
			if (tocopy < max_s) tmp->size = tocopy;
			else tmp->size = max_s;
			tmp->data = kmalloc(tmp->size, GFP_KERNEL);
			while (!tmp->data) {
				tmp->size /= 2;
	                        if (tmp->size < min_s) {
					ret = -ENOMEM;
					goto out;
				}
				tmp->data = kmalloc(tmp->size, GFP_KERNEL);
			}
			if (copy_from_user(tmp->data, (void *)user_p.host+offset, tmp->size)) {
				ret = -EACCES;
				goto out;
			}
			user_p.nr_pages++;
			list_add_tail(&(tmp->list), &(mcd.list));
			tocopy -= tmp->size;
			offset += tmp->size;
		}
                output_sg_arg = kmalloc(2*user_p.nr_pages*sizeof(struct scatterlist), GFP_KERNEL);
                if (!output_sg_arg) {
                        ret = -ENOMEM;
                        goto out;
                }
                sgs = kmalloc((1+2*user_p.nr_pages)*sizeof(struct scatterlist *), GFP_KERNEL);
                if (!sgs) {
                        ret = -ENOMEM;
                        goto out;
                }
		sg_init_one(&output_sg, &user_p, sizeof(user_p));
		sgs[num_out++] = &output_sg;
		j = 0;
		list_for_each_entry(tmp, &mcd.list, list) {
                        sg_init_one(&output_sg_arg[j], tmp->data, sizeof(void *));
                        sgs[num_out++] = &output_sg_arg[j];
			j++;
                        sg_init_one(&output_sg_arg[j], &tmp->size, sizeof(long));
                        sgs[num_out++] = &output_sg_arg[j];
			j++;
		}
		break;
        case CUMEMCPYDTOHV2:
                INIT_LIST_HEAD(&mcd.list);
		user_p.nr_pages = 0;
                tocopy = user_p.bytesize;
		while (tocopy > 0) {
                        tmp = kmalloc(sizeof(struct memcpy_data *), GFP_KERNEL);
                        if (!tmp) {
                                ret = -ENOMEM;
                                goto out;
                        }
                        if (tocopy < max_s) tmp->size = tocopy;
                        else tmp->size = max_s;
                        tmp->data = kmalloc(tmp->size, GFP_KERNEL);
                        while (!tmp->data) {
                                tmp->size /= 2;
                                if (tmp->size < min_s) {
                                        ret = -ENOMEM;
                                        goto out;
                                }
                                tmp->data = kmalloc(tmp->size, GFP_KERNEL);
                        }
			tocopy -= tmp->size;
                        list_add_tail(&(tmp->list), &(mcd.list));
			user_p.nr_pages++;
                }
                sgs = kmalloc((1+2*user_p.nr_pages)*sizeof(struct scatterlist *), GFP_KERNEL);
                if (!sgs) {
                        ret = -ENOMEM;
                        goto out;
                }
                output_sg_arg = kmalloc(2*user_p.nr_pages*sizeof(struct scatterlist), GFP_KERNEL);
                if (!output_sg_arg) {
                        ret = -ENOMEM;
                        goto out;
                }
		sg_init_one(&output_sg, &user_p, sizeof(user_p));
		sgs[num_out++] = &output_sg;
                j = 0;
                list_for_each_entry(tmp, &mcd.list, list) {
                        sg_init_one(&output_sg_arg[j], tmp->data, sizeof(void *));
                        sgs[num_out + num_in++] = &output_sg_arg[j];
                        j++;
                        sg_init_one(&output_sg_arg[j], &tmp->size, sizeof(long));
                        sgs[num_out + num_in++] = &output_sg_arg[j];
                        j++;
                }
		break;
	//allocate memory for function name and copy it from user space
	case CUMODULEGETFUNCTION:
		name = kzalloc(user_p.length*sizeof(char), GFP_KERNEL);
		if (!name) {
	                ret = -ENOMEM;
	                goto out;
	        }
                if (copy_from_user(name, (void *)user_p.name, user_p.length*sizeof(char))) {
                        ret = -EACCES;
			goto out;
		}
		sg_init_one(&output_sg, &user_p, sizeof(user_p));
		sg[num_out++] = &output_sg;
		sg_init_one(&output_sg1, name, sizeof(char *));
		sg[num_out++] = &output_sg1;
		break;
	//allocate an array of pointers that stores pointers to launch arguments (count = val1)
	//allocate of array of size_t that stores each argument's size
	//allocate memory for each lauch argument and copy it from user space
	//allocate and initialize one scatterlist entry for each argument
	case CULAUNCHKERNEL:
		args = kmalloc(user_p.val1*sizeof(void *), GFP_KERNEL);
		if (!args) {
	                ret = -ENOMEM;
	                goto out;
	        }
		if (copy_from_user(args, (void *)user_p.args, sizeof(user_p.val1*sizeof(void *)))) {
                        ret = -EACCES;
                        goto out;
                }
		size = kmalloc((size_t)user_p.val1*sizeof(size_t), GFP_KERNEL);
		if (!size) {
	                ret = -ENOMEM;
	                goto out;
	        }
		if (copy_from_user(size, user_p.size, user_p.val1*sizeof(size_t))) {
                        ret = -EACCES;
                        goto out;
                }
		for (i=0; i<user_p.val1; i++) {
			args[i] = kmalloc(size[i], GFP_KERNEL);
			if (!args[i]) {
				ret = -ENOMEM;
	                        goto out;
	                }
			if (copy_from_user(args[i], (void *)user_p.args[i], size[i])) {
                 	       ret = -EACCES;
	                        goto out;
	                }				
		}
                sgs = kmalloc((12+user_p.val1)*sizeof(struct scatterlist *), GFP_KERNEL);
                if (!sgs) {
                        ret = -ENOMEM;
                        goto out;
                }
		sg_init_one(&output_sg, &user_p, sizeof(user_p));
		sgs[num_out++] = &output_sg;
		output_sg_arg = kmalloc(user_p.val1*sizeof(struct scatterlist), GFP_KERNEL);
		for (i=0; i<user_p.val1; i++) {
			sg_init_one(&output_sg_arg[i], args[i], size[i]);
        	        sgs[num_out++] = &output_sg_arg[i];
		}
		break;
	//in all other cases no special allocations needed
	//just initialize scatterlist with struct copied from user (contains necessary arguments for those cases)
	default:
		sg_init_one(&output_sg, &user_p, sizeof(user_p));
		sg[num_out++] = &output_sg;
		break;
	}

	//we use sgs's address to check if the buffer we are waiting for is ready
	if (cmd != CUMEMCPYHTODV2 && cmd != CUMEMCPYDTOHV2 && cmd != CULAUNCHKERNEL)
		sgs = kmalloc(sizeof(char), GFP_KERNEL);

	spin_lock_irqsave(&cudev->lock, flags);

	//push scatterlists to virtqueue
	if (cmd == CUMEMCPYHTODV2 || cmd == CUMEMCPYDTOHV2 || cmd == CULAUNCHKERNEL)
		err = virtqueue_add_sgs(cudev->vq, sgs, num_out, num_in, (void *)sgs, GFP_ATOMIC);
	else
		err = virtqueue_add_sgs(cudev->vq, sg, num_out, num_in, (void *)sgs, GFP_ATOMIC);

	//notify backend
	virtqueue_kick(cudev->vq);

#ifdef SLEEP
	//sleep until buffer is available
	//wake up by interrupt
	spin_unlock_irqrestore(&cudev->lock, flags);
	while(!flag) {
		if (wait_event_interruptible(cudrvdata.wq, (flag = check_vqueue_buff(sgs))))
			return -EFAULT;
	}
#endif

#ifdef BUSY_WAIT
	//check if buffer is available
	while (virtqueue_get_buf(cudev->vq, &len) == NULL)
		;
	spin_unlock_irqrestore(&cudev->lock, flags);
#endif

	kfree(sgs);
	switch(cmd) {
	case CUDEVICEGETNAME:
		if (copy_to_user(user_p.name, name, user_p.val1*sizeof(char))) {
                        ret = -EACCES;
                        goto out;
                }
		kfree(name);
		break;
        case CUMODULELOAD:
		kfree(name);
		break;
        case CUMEMCPYHTODV2:
                list_for_each_entry_safe(tmcd, tmp, &mcd.list, list) {
			list_del(&tmcd->list);
			kfree(tmcd->data);
			kfree(tmcd);
		}
                kfree(output_sg_arg);
		break;
        case CUMEMCPYDTOHV2:
		offset = 0;
                list_for_each_entry_safe(tmcd, tmp, &mcd.list, list) {
                        if (copy_to_user(user_p.host+offset, tmcd->data, tmcd->size)) {
				ret = -EACCES;
				goto out;
			}
			offset += tmcd->size;

                        list_del(&tmcd->list);
                        kfree(tmcd->data);
                        kfree(tmcd);
                }
                kfree(output_sg_arg);
                break;
	case CUMODULEGETFUNCTION:
		kfree(name);
		break;
	case CULAUNCHKERNEL:
		for (i=0; i<user_p.val1; i++)
                        kfree(args[i]);
		kfree(args);
		kfree(size);
		break;
	default:
		break;
	}
        if (copy_to_user((void *)arg, &user_p, sizeof(struct param))) {
                ret = -EACCES;
                goto out;
        }
out:

	return ret;
}

static ssize_t cuda_chrdev_read(struct file *filp, char __user *usrbuf, 
                                  size_t cnt, loff_t *f_pos)
{
	debug("Entering");
	debug("Leaving");
	return -EINVAL;
}

static struct file_operations cuda_chrdev_fops = 
{
	.owner          = THIS_MODULE,
	.open           = cuda_chrdev_open,
	.release        = cuda_chrdev_release,
	.read           = cuda_chrdev_read,
	.unlocked_ioctl = cuda_chrdev_ioctl,
};

int cuda_chrdev_init(void)
{
	int ret;
	dev_t dev_no;
	unsigned int cuda_minor_cnt = CUDA_NR_DEVICES;
	
	debug("Initializing character device...");
	cdev_init(&cuda_chrdev_cdev, &cuda_chrdev_fops);
	cuda_chrdev_cdev.owner = THIS_MODULE;
	
	dev_no = MKDEV(CUDA_CHRDEV_MAJOR, 0);
	ret = register_chrdev_region(dev_no, cuda_minor_cnt, "crypto_devs");
	if (ret < 0) {
		debug("failed to register region, ret = %d", ret);
		goto out;
	}
	ret = cdev_add(&cuda_chrdev_cdev, dev_no, cuda_minor_cnt);
	if (ret < 0) {
		debug("failed to add character device");
		goto out_with_chrdev_region;
	}

	debug("Completed successfully");
	return 0;

out_with_chrdev_region:
	unregister_chrdev_region(dev_no, cuda_minor_cnt);
out:
	return ret;
}

void cuda_chrdev_destroy(void)
{
	dev_t dev_no;
	unsigned int cuda_minor_cnt = CUDA_NR_DEVICES;

	debug("entering");
	dev_no = MKDEV(CUDA_CHRDEV_MAJOR, 0);
	cdev_del(&cuda_chrdev_cdev);
	unregister_chrdev_region(dev_no, cuda_minor_cnt);
	debug("leaving");
}
