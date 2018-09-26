/*!
  \file alloc.h
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
  \brief Memory allocation routines (header file)
 */

#ifndef ALLOC_H
#define ALLOC_H

#include <cstdlib>
#include <cstdio>
#include <mm_malloc.h>
static const int MEMORY_ALIGNMENT = 128;
#define alloc_error_check(p) {			\
    if ((p) == NULL) { \
        fprintf(stderr, "Allocation Failure!\n"); \
        exit(1); \
    } \
}

/*!
  \brief Allocate memory for 1d array
 */
template<typename T, typename Q>  T*  alloc_1d(Q n1){
  T* p = (T*) _mm_malloc(n1*sizeof(T), MEMORY_ALIGNMENT);
  alloc_error_check(p);
  return p;
}


/*!
  \brief Free memory for 1d array
 */
template<typename T>  void free_1d(T* p){
  _mm_free(p);
}


/*!
  \brief Create 2d view of 1d array referenced by i
  \details Allocates array ii of size n1, each element of ii give view of rows of i
  ii[I][J] = i[I*n2 + J]
 */
template<typename T, typename Q> void initview_2d(Q n1, Q n2,  T * const p, T** const &pp){
  pp[0] = p;
  for(Q j = Q(1); j < n1; j++) pp[j] = pp[j-1] + n2;  
}
template<typename T, typename Q> T**  allocview_2d(Q n1, Q n2){
  T** pp = (T **) alloc_1d<T*>(n1);
  alloc_error_check(pp);
  return pp;
}

/*!
  \brief Destroy 2d view
 */
template<typename T> void freeview_2d(T **pp){
  free_1d<T*>(pp);
}

/*!
  \brief Allocate memory for 2d array
  \details Allocates linear array of size n1*n2, position of element
  (I,J) is im = I*n2 + J
 */
template<typename T, typename Q> T** alloc_2d(Q n1, Q n2){
  T*  p  = (T *)  alloc_1d<T>(n1 * n2);
  alloc_error_check(p);
  T** pp = allocview_2d<T>(n1, n2);
  initview_2d<T>(n1, n2, p, pp);
  return pp;
}

/*!
  \brief Free memory for 2d array
 */
template<typename T> void free_2d(T **pp){
  free_1d<T>(pp[0]);
  freeview_2d<T>(pp);
}

/*!
  \brief Create 3d view of 1d array referenced by i
  \details Allocates array iii of size n1, each iii[.] points to 2D array ii such that
  iii[I][J][K] = i[I*n2*n3 + J*n3 + K]
 */
template<typename T, typename Q> void initview_3d(Q n1, Q n2, Q n3, T* const p, T*** const &ppp){
  T **pp = ppp[0];
  pp[0]  = p;
  for(Q j = Q(1); j < n1*n2; j++) pp[j] = pp[j-1] + n3;
}
template<typename T, typename Q> T*** allocview_3d(Q n1, Q n2, Q n3){
  T ***ppp = (T ***) alloc_1d<T**>(n1);
  alloc_error_check(ppp);
  T **pp   = (T **)  alloc_1d<T*>(n1 * n2);
  alloc_error_check(pp);

  ppp[0] = pp;
  for(Q j = Q(1); j < n1; j++) ppp[j] = ppp[j-1] + n2;
  return ppp;
}

/*!
  \brief Destroy 3d view
 */
template<typename T> void freeview_3d(T ***ppp){
  free_1d<T*>(ppp[0]);
  free_1d<T**>(ppp);
}

/*!
  \brief Allocate memory for 3d array
  \details Allocates linear array of size n1*n2*n3, position of element
  (I,J,K) is im = I*n2*n3 + J*n3 + K
 */
template<typename T, typename Q> T ***alloc_3d(Q n1, Q n2, Q n3){
  T*     p = (T *) alloc_1d<T>(n1 * n2 * n3);
  alloc_error_check(p);
  T*** ppp = allocview_3d<T>(n1, n2, n3);
  initview_3d<T>(n1, n2, n3, p, ppp);
  return ppp;
}

/*!
  \brief Free memory for 3d array
 */
template<typename T> void free_3d(T ***ppp){
  free_1d<T>(ppp[0][0]);
  freeview_3d<T>(ppp);
}


int*       alloc_1d_int(int n1);
int**      alloc_2d_int(int n1, int n2);
int***     alloc_3d_int(int n1, int n2, int n3);
int**      allocview_2d_int(int n1, int n2);
int***     allocview_3d_int(int n1, int n2, int n3);
void       initview_2d_int(int n1, int n2, int* const i, int** &ii);
void       initview_3d_int(int n1, int n2, int n3, int* const i, int*** &iii);


void       free_1d_int(int *i);
void       free_2d_int(int **ii);
void       free_3d_int(int ***iii);
void       freeview_2d_int(int **ii);
void       freeview_3d_int(int ***iii);


double*    alloc_1d_double(int n1);
double**   alloc_2d_double(int n1, int n2);
double***  alloc_3d_double(int n1, int n2, int n3);
double**   allocview_2d_double(int n1, int n2);
double***  allocview_3d_double(int n1, int n2, int n3);
void       initview_2d_double(int n1, int n2, double* const d, double** &dd);
void       initview_3d_double(int n1, int n2, int n3, double* const d, double*** &ddd);

void       free_1d_double(double *d);
void       free_2d_double(double **dd);
void       free_3d_double(double ***ddd);
void       freeview_2d_double(double **dd);
void       freeview_3d_double(double ***ddd);
#endif
