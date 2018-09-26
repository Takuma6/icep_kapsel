/*!
  \file particle_rotation_solver.h
  \brief Solver routines for particle orientations
  \author J. Molina
  \date 2013/05/15
  \version 1.0
 */
#ifndef PARTICLE_ROTATION_SOLVER_H
#define PARTICLE_ROTATION_SOLVER_H

#include "variable.h"
#include "rigid_body.h"
//Orientation solvers
//First-order Euler
inline void MD_solver_orientation_Euler(Particle &p, const double &dt){
  quaternion dqdt;
  qtn_init(p.q_old, p.q);
  qdot(dqdt, p.q, p.omega, SPACE_FRAME);
  qtn_add(p.q, dqdt, dt);
  qtn_normalize(p.q);
}

//Simo & Wong second-order scheme
inline void MD_solver_orientation_AB2(Particle &p, const double &hdt){
  double wb[DIM];
  double wb_old[DIM];
  //only add angular velocity vectors in body coordinates !
  rigid_body_rotation(wb, p.omega, p.q, SPACE2BODY);
  rigid_body_rotation(wb_old, p.omega_old, p.q_old, SPACE2BODY);
  for(int d = 0; d < DIM; d++){
    wb[d] = 3.0*wb[d] - wb_old[d];
  }
  
  quaternion dqdt;
  qtn_init(p.q_old, p.q);
  qdot(dqdt, p.q, wb, BODY_FRAME);
  qtn_add(p.q, dqdt, hdt);
  qtn_normalize(p.q);
}

// Samuel Buss' second-order scheme
// untested
inline void MD_solver_orientation_SB2(Particle &p, const double &dt){
  double wb[DIM];
  for(int d = 0; d < DIM; d++){
    wb[d] = p.omega[d] + dt/2.0 * IMOI[p.spec]*p.torque_hydro[d];
  }
  
  quaternion dqdt;
  qtn_init(p.q_old, p.q);
  qdot(dqdt, p.q, wb, SPACE_FRAME);
  qtn_add(p.q, dqdt, dt);
  qtn_normalize(p.q);
}

/*!
  \brief Solve Euler equations in body frame
 */
inline void MD_solver_omega_Euler(double omega[DIM],
				  const double torque[DIM],
				  const double Ib[DIM],
				  const quaternion &q,
				  const int free_omega[DIM],
				  const double &dt){

  double omega_b[DIM];
  double torque_b[DIM];
  rigid_body_rotation(omega_b, omega, q, SPACE2BODY);
  rigid_body_rotation(torque_b, torque, q, SPACE2BODY);

  double new_omega_b[DIM] = {omega_b[0], omega_b[1], omega_b[2]};
  
  // Warning: Fixed_omega flag now fixes angular velocity about rigid body frame
  //          Inconsistent with Fixed_vel flag which used lab frame
  if(free_omega[0])
    new_omega_b[0] += (dt/Ib[0]) * (torque_b[0] + (Ib[1] - Ib[2])*omega_b[1]*omega_b[2]);
  
  if(free_omega[1])
    new_omega_b[1] += (dt/Ib[1]) * (torque_b[1] + (Ib[2] - Ib[0])*omega_b[2]*omega_b[0]);
  
  if(free_omega[2])
    new_omega_b[2] += (dt/Ib[2]) * (torque_b[2] + (Ib[0] - Ib[1])*omega_b[0]*omega_b[1]);
  
  rigid_body_rotation(omega, new_omega_b, q, BODY2SPACE);
}
inline void MD_solver_omega_Euler_update(double delta_omega[DIM],
					 const double omega[DIM],
					 const double torque[DIM],
					 const double Ib[DIM],
					 const quaternion &q,
					 const double &dt){
  
  double omega_b[DIM];
  double torque_b[DIM];
  rigid_body_rotation(omega_b, omega, q, SPACE2BODY);
  rigid_body_rotation(torque_b, torque, q, SPACE2BODY);

  delta_omega[0] = (dt/Ib[0]) * (torque_b[0] + (Ib[1] - Ib[2])*omega_b[1]*omega_b[2]);
  delta_omega[1] = (dt/Ib[1]) * (torque_b[1] + (Ib[2] - Ib[0])*omega_b[2]*omega_b[0]);
  delta_omega[2] = (dt/Ib[2]) * (torque_b[2] + (Ib[0] - Ib[1])*omega_b[0]*omega_b[1]);

  rigid_body_rotation(delta_omega, q, BODY2SPACE);
}
/*!
  \brief Solve euler equations in body frame
*/
inline void MD_solver_omega_AB2(double omega[DIM],
				const double torque[DIM],
				const double torque_old[DIM],
				const double Ib[DIM],
				const quaternion &q,
				const quaternion &q_old,
				const int free_omega[DIM],
				const double &hdt){
  double omega_b[DIM];
  double torque_b[DIM];
  double torque_b_old[DIM];
  
  rigid_body_rotation(omega_b, omega, q, SPACE2BODY);
  rigid_body_rotation(torque_b, torque, q, SPACE2BODY);
  rigid_body_rotation(torque_b_old, torque_old, q_old, SPACE2BODY);
  
  double new_omega_b[DIM] = {omega_b[0], omega_b[1], omega_b[2]};
  
  // Warning: Fixed_omega flag now fixes angular velocity about rigid body frame
  //          Inconsistent with Fixed_vel flag which used lab frame
  if(free_omega[0])    
    new_omega_b[0] += (hdt/Ib[0]) * ((torque_b[0] + torque_b_old[0]) + (Ib[1] - Ib[2])*omega_b[1]*omega_b[2]);
  
  if(free_omega[1])
    new_omega_b[1] += (hdt/Ib[1]) * ((torque_b[1] + torque_b_old[1]) + (Ib[2] - Ib[0])*omega_b[2]*omega_b[0]);
  
  if(free_omega[2])
    new_omega_b[2] += (hdt/Ib[2]) * ((torque_b[2] + torque_b_old[2]) + (Ib[0] - Ib[1])*omega_b[0]*omega_b[1]);
  
  rigid_body_rotation(omega, new_omega_b, q, BODY2SPACE);
}

#endif
