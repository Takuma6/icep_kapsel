#ifndef PERIODIC_BOUNDARY_H
#define PERIODIC_BOUNDARY_H

#include "input.h"
/*!
  \brief Enforce pbc on position 
  \param[in,out] x position
 */
inline void PBC(double *x){
  for(int d = 0; d < DIM; d++){
    x[d] = fmod(x[d] + L_particle[d], L_particle[d]);
    assert(x[d] >= 0);
    assert(x[d] < L[d]);
  }
}
inline void PBC(double *x, double *xpbc){
  for(int d = 0; d < DIM; d++) xpbc[d] = x[d];
  PBC(xpbc);
}

inline int extract_id(int loc_x, int loc_y, int loc_z){
  int dum_i = (loc_x+NX) % NX;
  int dum_j = (loc_y+NY) % NY;
  int dum_k = (loc_z+NZ) % NZ;
  int im = (dum_i * NY * NZ_) + (dum_j * NZ_) + dum_k;
  return im;
}

inline void Interpolate_vec_on_normal_grid(double **vec){
  int im, im_x_bw, im_y_bw, im_z_bw;
#pragma omp parallel for private(im) 
  for(int i; i<NX; i++){
    for(int j; j<NY; j++){
      for(int k; k<NZ; k++){
        im=(i*NY*NZ_)+(j*NZ_)+k;
        im_x_bw = extract_id(i-1,j,k);
        im_y_bw = extract_id(i,j-1,k);
        im_z_bw = extract_id(i,j,k-1);
        vec[0][im] = (vec[0][im] + vec[0][im_x_bw])/2.0;
        vec[1][im] = (vec[1][im] + vec[1][im_y_bw])/2.0;
        vec[2][im] = (vec[2][im] + vec[2][im_z_bw])/2.0;
      }
    }
  }
}

/*!
  \brief Enforce Lees-Edwards pbc on position
  \param[in,out] x position
  \param[out] delta_vx change in x-velocity due to boundary crossing
 */
inline int PBC_OBL(double *x, double &delta_vx){
  double signY = x[1];
  x[1] = fmod(x[1] + L_particle[1], L_particle[1]);
  signY -= x[1];
  int sign = (int) signY;
  if(!(sign == 0)){
    sign = sign / abs(sign);
  }
  
  
  x[0] -= (double)sign * degree_oblique * L_particle[1];
  x[0] = fmod(x[0] + L_particle[0], L_particle[0]);
  x[2] = fmod(x[2] + L_particle[2], L_particle[2]);
  for(int d = 0; d < DIM; d++){
    assert(x[d] >= 0);
    assert(x[d] < L[d]);
  }
  
  delta_vx = -(double)sign * Shear_rate_eff * L_particle[1];
  return sign;
}
inline int PBC_OBL(double *x, double *xpbc, double &delta_vx){
  for(int d = 0; d < DIM; d++) xpbc[d] = x[d];
  return PBC_OBL(xpbc, delta_vx);
}
#endif
