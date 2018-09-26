#ifndef PERIODIC_SPLINE_H
#define PERIODIC_SPLINE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string>
#include <cstring>

#include "alloc.h"
#include "macro.h"

/*!
  \file periodic_spline.h
  \author J. Molina
  \date 2014/04/03
  \version 1.1
  \brief Spline interpolation for 1D periodic systems [0,L] using equally
  spaced sample points
 */

//f(x) = S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)**2 + d_i*(x-x_i)**3
typedef struct splineSystem{
  int      n;
  double   dx;
  double*  a;
  double*  b;
  double*  c;
  double*  d;
  double*  Q;
  double*  Aii;
  double*  Ain;
} splineSystem;

/*!
  \brief Evaluate function at x using spline interpolation
  \param[in] *spl splineSystem object with interpolation data
  \param[in] x interpolation position
 */
inline double splineFx(const splineSystem* spl, const double &x){
  assert(x >= 0.0 && x < (spl->n) * (spl->dx));
  int i = x / (spl->dx);
  double delta = (x - (double)i*spl->dx);
  double delta2= delta*delta;
  return (spl->a[i]) + (spl->b[i])*delta + 
    (spl->c[i])*delta2 + (spl->d[i])*delta2*delta;
}

/*!
  \brief Initialize working memory
  \param[in,out] *spl splineSystem object to initialize
  \param[in] n number of grid points for interpolation
  \param[in] dx grid spacing
 */
void splineInit(splineSystem* &spl, const int &n, const double &dx);

/*!
  \brief Free working memory
  \param[in] *spl splineSysbem object to free
 */
void splineFree(splineSystem* &spl);

/*!
  \brief Compute periodic spline interpolation
  \param[in,out] *spl splineSystem
  \param[in] *fx interpolation function values over periodic grid
 */
void splineCompute(splineSystem* spl, const double* fx);


#endif
