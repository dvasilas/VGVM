#ifndef _DEBUG_H
#define _DEBUG_H

#if DEBUG
#define debug(fmt,arg...)     printk(KERN_DEBUG "[virtio-cuda] %s: " fmt "\n", __func__ , ##arg)
#else
#define debug(fmt,arg...)     do { } while(0)
#endif

#endif
