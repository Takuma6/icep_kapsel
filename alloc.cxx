/*!
  \file alloc.c
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
  \brief Memory allocation routines
 */

#include "alloc.h"

int*       alloc_1d_int(int n1){return alloc_1d<int, int>(n1);}
int**      alloc_2d_int(int n1, int n2){return alloc_2d<int, int>(n1, n2);}
int***     alloc_3d_int(int n1, int n2, int n3){return alloc_3d<int, int>(n1, n2, n3);}
int**      allocview_2d_int(int n1, int n2){return allocview_2d<int, int>(n1, n2);}
int***     allocview_3d_int(int n1, int n2, int n3){return allocview_3d<int, int>(n1, n2, n3);}
void       initview_2d_int(int n1, int n2, int* const i, int** &ii){return initview_2d<int, int>(n1, n2, i, ii);}
void       initview_3d_int(int n1, int n2, int n3, int* const i, int*** &iii){return initview_3d<int, int>(n1, n2, n3, i, iii);}


void       free_1d_int(int *i){return free_1d<int>(i);}
void       free_2d_int(int **ii){return free_2d<int>(ii);}
void       free_3d_int(int ***iii){return free_3d<int>(iii);}
void       freeview_2d_int(int **ii){return freeview_2d<int>(ii);}
void       freeview_3d_int(int ***iii){return freeview_3d<int>(iii);}


double*    alloc_1d_double(int n1){return alloc_1d<double, int>(n1);}
double**   alloc_2d_double(int n1, int n2){return alloc_2d<double, int>(n1, n2);}
double***  alloc_3d_double(int n1, int n2, int n3){return alloc_3d<double, int>(n1, n2, n3);}
double**   allocview_2d_double(int n1, int n2){return allocview_2d<double, int>(n1, n2);}
double***  allocview_3d_double(int n1, int n2, int n3){return allocview_3d<double, int>(n1, n2, n3);}
void       initview_2d_double(int n1, int n2, double* const d, double** &dd){return initview_2d<double, int>(n1, n2, d, dd);}
void       initview_3d_double(int n1, int n2, int n3, double* const d, double*** &ddd){return initview_3d<double, int>(n1, n2, n3, d, ddd);}

void       free_1d_double(double *d){return free_1d<double>(d);}
void       free_2d_double(double **dd){return free_2d<double>(dd);}
void       free_3d_double(double ***ddd){return free_3d<double>(ddd);}
void       freeview_2d_double(double **dd){return freeview_2d<double>(dd);}
void       freeview_3d_double(double ***ddd){return freeview_3d<double>(ddd);}









