#include "periodic_spline.h"

void splineFree(splineSystem* &spl){
  free_1d_double(spl->a);
  free_1d_double(spl->b);
  free_1d_double(spl->c);
  free_1d_double(spl->d);
  free_1d_double(spl->Q);
  free_1d_double(spl->Aii);
  free_1d_double(spl->Ain);
  delete spl;
  spl = NULL;
}
void splineInit(splineSystem* &spl, const int &n, const double &dx){
  assert(n >= 5);
  spl = new splineSystem;
  spl->n    = n;
  spl->dx   = dx;
  spl->a    = alloc_1d_double(n);
  spl->b    = alloc_1d_double(n);
  spl->c    = alloc_1d_double(n);
  spl->d    = alloc_1d_double(n);
  spl->Q    = alloc_1d_double(n);
  spl->Aii  = alloc_1d_double(n);
  spl->Ain  = alloc_1d_double(n);
  for(int i = 0; i < n; i++){
    spl->a[i] = spl->b[i] = spl->c[i] = spl->d[i] = spl->Q[i] = 0.0 ;
    spl->Aii[i] = spl->Ain[i] = 0.0;
  }

  //initialize matrix coefficients
  {
    double* Aii   = spl->Aii;
    double* Ain   = spl->Ain;
    double* alpha = alloc_1d_double(n);
    double sign;
    
    //auxiliary coefficients
    alpha[0]  = 1.0;
    alpha[1]  = 4.0;
    for(int i = 2; i < n; i++)  alpha[i] = 4.0*alpha[i-1] - alpha[i-2];

    //diagonal
    Aii[n-1] = 4.0;
    for(int i = 0; i < n-1; i++) Aii[i] = alpha[i+1] / alpha[i];

    //last column
    Ain[n-1] = 4.0;
    Ain[n-2] = 1.0/alpha[n-2] + 1.0;
    for(int i = n-3; i >= 0; i--){
      sign = (i % 2 == 0 ? 1.0 : -1.0);
      Ain[i] = sign / alpha[i] - Ain[i+1] / Aii[i+1];
    }


    //reduce last row
    Aii[n-1] -= (Ain[0]/Aii[0] + Ain[n-2]/Aii[n-2]);
    Ain[n-1] -= (Ain[0]/Aii[0] + Ain[n-2]/ Aii[n-2]);

    free_1d_double(alpha);
  }
}

void splineCompute(splineSystem* spl, const double* fx){
  int n        = spl->n;
  double dx    = spl->dx;
  double* a    = spl->a;
  double* b    = spl->b;
  double* c    = spl->c;
  double* d    = spl->d;
  double* Q    = spl->Q;
  double* Aii  = spl->Aii;
  double* Ain  = spl->Ain;

  //initialize
  for(int i = 0; i < n; i++) a[i] = fx[i];

  //reduce 
  {
    //lower part
    Q[0]  = 3.0*(fx[1] - 2.0*fx[0] + fx[n-1])/SQ(dx);
    Q[n-1]= 3.0*(fx[0] - 2.0*fx[n-1] + fx[n-2])/SQ(dx);
    for(int i = 1; i < n-1; i++) 
      Q[i] = 3.0*(fx[i+1] - 2.0*fx[i] + fx[i-1])/SQ(dx) - Q[i-1]/Aii[i-1];

    //upper part
    for(int i = n-3; i >= 0; i--) Q[i] -= Q[i+1]/Aii[i+1];
    
    //last row
    Q[n-1] -= (Q[0]/Aii[0] + Q[n-2]/Aii[n-2]);
  }

  //back substitute for c_i coefficient
  {
    double cn = Q[n - 1] / Aii[n - 1];
    c[n - 1] = cn;
    for(int i = n - 2; i >= 0; i--)  c[i] = (Q[i] - Ain[i]*cn) / Aii[i];
  }

  //b_i & d_i coefficient
  {
    int inext;
    for(int i = 0; i < n; i++){
      inext = (i < n-1 ? i+1 : 0);
      d[i] = (c[inext] - c[i])/(3.0*dx);
      b[i] = (a[inext] - a[i])/dx - c[i]*dx - d[i]*SQ(dx);
    }
  }
}
