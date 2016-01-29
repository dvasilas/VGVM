#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/module.h>
#include <linux/spinlock.h>
#include <linux/virtio.h>
#include <linux/virtio_config.h>
#include <linux/semaphore.h>
#include <linux/kthread.h>
#include "cuda-util.h"
#include "cuda-chrdev.h"
#include "debug.h"
#include <asm/msr.h>

struct cuda_driver_data cudrvdata;
extern struct wait_struct wait_data;
struct cuda_device *gcudev;

//interrupt handler
//get available buffers from virtqueue and add them in list
static void vq_has_data(struct virtqueue *vq)
{
#ifdef SLEEP
        unsigned int len;
	unsigned long flags, flags1;
	struct wait_struct *tmp;
	void *d;
        struct timeval handl_t;
	while((d = virtqueue_get_buf(vq, &len))) {
		tmp = (struct wait_struct *)kmalloc(sizeof(struct wait_struct), GFP_ATOMIC);
		if (!tmp) {
			debug("ENOMEN");
			return;
		}
		tmp->data = d;
                spin_lock_irqsave(&wait_data.lock, flags);
		list_add(&(tmp->list), &(wait_data.list));
                spin_unlock_irqrestore(&wait_data.lock, flags);
	}
	wake_up_interruptible_all(&cudrvdata.wq);
#endif
}

static struct virtqueue *find_vq(struct virtio_device *vdev)
{
	int err;
	struct virtqueue *vq;

	vq = virtio_find_single_vq(vdev, vq_has_data, "cuda-vq");
	if (IS_ERR(vq)) {
		debug("Could not find vq");
		vq = NULL;
	}

	return vq;
}

static int virtcons_probe(struct virtio_device *vdev)
{
	int ret = 0;
	struct cuda_device *cudev;

	cudev = kzalloc(sizeof(*cudev), GFP_KERNEL);
	if (!cudev) {
		ret = -ENOMEM;
		goto out;
	}

	cudev->vdev = vdev;
	vdev->priv = cudev;

	cudev->vq = find_vq(vdev);
	if (!(cudev->vq)) {
		ret = -ENXIO;
		goto out;		
	}

	gcudev = cudev;
	spin_lock_init(&cudev->lock);
	init_waitqueue_head(&cudrvdata.wq);
	spin_lock_init(&wait_data.lock);
	INIT_LIST_HEAD(&wait_data.list);
	sema_init(&cudev->slock, 1);
	
	spin_lock_irq(&cudrvdata.lock);
	cudev->minor = cudrvdata.next_minor++;
	list_add_tail(&cudev->list, &cudrvdata.devs);
	spin_unlock_irq(&cudrvdata.lock);

out:
	return ret;
}

static void virtcons_remove(struct virtio_device *vdev)
{
	struct cuda_device *cudev = vdev->priv;

	spin_lock_irq(&cudrvdata.lock);
	list_del(&cudev->list);
	spin_unlock_irq(&cudrvdata.lock);

	vdev->config->reset(vdev);
	vdev->config->del_vqs(vdev);

	kfree(cudev);
}

static struct virtio_device_id id_table[] = {
	{VIRTIO_ID_CUDA, VIRTIO_DEV_ANY_ID },
	{ 0 },
};

static unsigned int features[] = {
	0
};

static struct virtio_driver virtio_cuda = {
	.feature_table = features,
	.feature_table_size = ARRAY_SIZE(features),
	.driver.name =	KBUILD_MODNAME,
	.driver.owner =	THIS_MODULE,
	.id_table =	id_table,
	.probe =	virtcons_probe,
	.remove =	virtcons_remove,
};

static int __init init(void)
{
	int ret = 0;

	ret = cuda_chrdev_init();
	if (ret < 0) {
		printk(KERN_ALERT "Could not initialize character devices.\n");
		goto out;
	}

	INIT_LIST_HEAD(&cudrvdata.devs);
	spin_lock_init(&cudrvdata.lock);

	ret = register_virtio_driver(&virtio_cuda);
	if (ret < 0) {
		printk(KERN_ALERT "Failed to register virtio driver.\n");
		goto out_with_chrdev;
	}

	return ret;

out_with_chrdev:
	cuda_chrdev_destroy();
out:
	return ret;
}

static void __exit fini(void)
{
	cuda_chrdev_destroy();
	unregister_virtio_driver(&virtio_cuda);
}

module_init(init);
module_exit(fini);

MODULE_DEVICE_TABLE(virtio, id_table);
MODULE_DESCRIPTION("Virtio cuda driver");
MODULE_LICENSE("GPL");
