#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include "libvrtcuda.h"

//search current directory for all CUDA binary files (.cubin)
//extracts objects (function names) from each binary file
//stores connection between functions and files
//uses linked list - maybe not the most efficient implementation
void registerCudaKernels(struct lib_data *libd) {
	char buff[1024];
	char cmd[1024];
	libd->cubins = NULL;
	libd->funcs = NULL;
	libd->args = NULL;
	FILE *fp = popen("ls | grep .cubin", "r");
	if (fp == NULL) {
		perror("popen");
		exit(1);
	}
        while (fgets(buff, sizeof(buff)-1, fp) != NULL) {
		size_t s = strlen(buff);
		strcpy(cmd,"nm ");
                strcat(cmd, buff);
                FILE *fp1 = popen(cmd, "r");
                if (fp1 == NULL) {
			perror("popen");
                        exit(1);
                }

		struct bin_data *n = malloc(sizeof(struct bin_data));
		n->next = libd->cubins;
		n->length = s;
		n->binname = malloc(s);
		n->loaded = 0;
		strcpy(n->binname, buff);
		libd->cubins = n;	

		while (fgets(buff, sizeof(buff)-1, fp1) != NULL) {		
			buff[strlen(buff)-1] = '\0';
			char *ret = strstr(buff, "_Z");
			if (ret != NULL) {
				s = strlen(ret);
				struct func_data *nf = malloc(sizeof(struct func_data));

				nf->next = libd->funcs;
				nf->length = s;
				nf->function = NULL;
				nf->bin = n;
				nf->funcname = malloc(strlen(ret));
				strcpy(nf->funcname, ret);
                                libd->funcs = nf;
			}
		}
		pclose(fp1);
	}
        pclose(fp);
}
