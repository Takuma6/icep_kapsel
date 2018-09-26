/*!
  \file rigid.h
  \brief Rigid particles as stiff chains
  \authors T. Kobiki, J. Molina
  \date 2012/12/29
  \version 2.0
*/
#ifndef RIGID_H
#define RIGID_H

#include "input.h"
#include "lad3.h"
#include "Matrix_Inverse.h"
#include "matrix_diagonal.h"
#include "periodic_boundary.h"
#include "rigid_body.h"
#include "particle_rotation_solver.h"

/*!
  \brief Compute initial center of mass position for each
  of the rigid particles, assuming no particle overlap
  \warning Input rigid body coordinates should be given without PBC
  \todo Remove PBC from rigid particles when writing restart files
*/
inline void init_set_xGs(Particle *p){
  //center of mass
#pragma omp parallel for
  for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
    
    for(int d=0; d<DIM; d++) xGs[rigidID][d] = 0.0;
    
    for(int n=Rigid_Particle_Cumul[rigidID]; n<Rigid_Particle_Cumul[rigidID+1]; n++){
      for(int d=0; d<DIM; d++) xGs[rigidID][d] += p[n].x[d];
    }
    
    
    for(int d=0; d<DIM; d++) xGs[rigidID][d] /= (double) Rigid_Particle_Numbers[rigidID];
  }
}

/*!
  \brief Initialize geometry for each of the rigid particles
  by computing relative vectors from COM to individual beads
  \warning Call only after center of mass position xGs (without PBC) has been set
 */
inline void init_set_GRvecs(Particle *p){
  //position vectors from center of mass to individual beads
#pragma omp parallel for
  for(int n=0; n<Particle_Number; n++){
    int rigidID = Particle_RigidID[n];
    for(int d=0; d<DIM; d++) {
      GRvecs[n][d] = p[n].x_nopbc[d] - xGs_nopbc[rigidID][d];
    }
  }
}

/*
  \brief Enforce PBC on rigid bodies
  \warning Call only after center of mass and geometry has been initialized
 */
inline void init_set_PBC(Particle *p){
  //set raw (no-pbc) coordinates for rigid particles
#pragma omp parallel for
  for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
    for(int d = 0; d < DIM; d++){
      xGs_nopbc[rigidID][d]    = xGs[rigidID][d];
      xGs_previous[rigidID][d] = xGs[rigidID][d];
    }
    for(int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID+1]; n++){
      for(int d = 0; d < DIM; d++){
	p[n].x_previous[d] = p[n].x[d];
	p[n].x_nopbc[d]    = p[n].x[d];
      }
    }
  }

  //place rigid particles (and beads) inside simulation box
#pragma omp parallel for
  for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
    PBC(xGs[rigidID]);
    for(int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID+1]; n++)
      PBC(p[n].x);
  }
}

/*
  \brief Enforce PBC on rigid bodies for Lees-Edwards simulations
  \warning Call only after center of mass and geometry has been initialized
 */
inline void init_set_PBC_OBL(Particle *p){
  //set raw (no-pbc) coordinates for rigid particles
#pragma omp parallel for
  for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
    for(int d = 0; d < DIM; d++){
      xGs_nopbc[rigidID][d]    = xGs[rigidID][d];
      xGs_previous[rigidID][d] = xGs[rigidID][d];
    }
    for(int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID+1]; n++){
      for(int d = 0; d < DIM; d++){
	p[n].x_previous[d] = p[n].x[d];
	p[n].x_nopbc[d]    = p[n].x[d];
      }
    }
  }
  
#pragma omp parallel for
  for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
    double dmy_vx;
    //rigid velocities are not reset
    PBC_OBL(xGs[rigidID], dmy_vx);
    for(int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID+1]; n++) 
      PBC_OBL(p[n].x, dmy_vx);
  }
}

/*!
  \brief Diagonalize intertia tensor to determine rigid body frame
 */
inline void init_Rigid_Coordinates(Particle *p){
#pragma omp parallel for
  for(int rigidID = 0; rigidID < Rigid_Number; rigidID++){
    double **eigen_vector;
    double eigen_value[DIM];
    double dmy_R[DIM][DIM];
    quaternion dmy_q;
    int iter;

    eigen_vector = alloc_2d_double(DIM, DIM);
    jacobi(Rigid_Moments[rigidID], eigen_vector, eigen_value, iter, DIM);
    M_coordinate_frame(eigen_vector[2], eigen_vector[0], eigen_vector[1]);
    for(int i = 0; i < DIM; i++){
      for(int j = 0; j < DIM; j++){
        dmy_R[j][i] = eigen_vector[i][j];
      }
    }
    rm_rqtn(dmy_q, dmy_R);
    
    for(int n = Rigid_Particle_Cumul[rigidID]; n < Rigid_Particle_Cumul[rigidID+1]; n++){
      qtn_init(p[n].q, dmy_q);
    }
    free_2d_double(eigen_vector);

    //Rigid_Moments_body gives inertia tensor in body-frame
    //By construction it should be diagonal
    rigid_body_matrix_rotation(Rigid_Moments_body[rigidID][0], Rigid_Moments[rigidID][0], dmy_q, SPACE2BODY);
  }

  //GRvecs_body gives position of all beads in body-frame
#pragma omp parallel for
  for(int n = 0; n < Particle_Number; n++){
    rigid_body_rotation(GRvecs_body[n], GRvecs[n], p[n].q, SPACE2BODY);
  }
}

inline void rigid_Velocity(double *v, double const* r, double const* vg, double const* wg){
  v[0] = vg[0] + (wg[1]*r[2] - wg[2]*r[1]);
  v[1] = vg[1] + (wg[2]*r[0] - wg[0]*r[2]);
  v[2] = vg[2] + (wg[0]*r[1] - wg[1]*r[0]);
}

/*
  \brief Update individual particle positions and velocities for
  current rigid configuration
 */
inline void update_Particle_Configuration(Particle *p){
  int rigidID;
#pragma omp parallel for private(rigidID)
  for(int n=0; n<Particle_Number; n++){
    rigidID = Particle_RigidID[n];
    for(int d=0; d<DIM; d++){
      p[n].x_previous[d] = p[n].x[d];
      p[n].x[d] = xGs[rigidID][d] + GRvecs[n][d];

      p[n].omega_old[d] = p[n].omega[d];
      p[n].v_old[d] = p[n].v[d];
      p[n].omega[d] = omegaGs[rigidID][d];
    }
    rigid_Velocity(p[n].v, GRvecs[n], velocityGs[rigidID], omegaGs[rigidID]);
    PBC(p[n].x);
   }
}

/*
  \brief Update individual particle positions and velocities for
  current rigid configuration under Lees-Edwards boundary conditions
 */
inline void update_Particle_Configuration_OBL(Particle *p){
  int rigidID, sign; 
  double delta_vx;
#pragma omp parallel for private(rigidID, sign, delta_vx)
  for(int n=0; n<Particle_Number; n++){
    rigidID = Particle_RigidID[n];
    for(int d=0; d<DIM; d++){
      p[n].x_previous[d] = p[n].x[d];
      p[n].x[d] = xGs[rigidID][d] + GRvecs[n][d];

      p[n].omega_old[d] = p[n].omega[d];
      p[n].v_old[d] = p[n].v[d];
      p[n].omega[d] = omegaGs[rigidID][d];
    }
    rigid_Velocity(p[n].v, GRvecs[n], velocityGs[rigidID], omegaGs[rigidID]);
    sign = PBC_OBL(p[n].x, delta_vx);
    p[n].v_old[0] += delta_vx;
    p[n].v[0] += delta_vx;
   }
}

/*!
  \brief Set particle velocities using current rigid velocities
  (assuming position of the particles has not changed)
 */
inline void set_Particle_Velocities(Particle *p){
  int rigidID;
#pragma omp parallel for private(rigidID)
  for(int n=0; n<Particle_Number; n++){
    rigidID = Particle_RigidID[n];
    for(int d=0; d<DIM; d++){
      p[n].omega[d] = omegaGs[rigidID][d];
    }
    rigid_Velocity(p[n].v, GRvecs[n], velocityGs[rigidID], omegaGs[rigidID]);
  }
}

/*!
  \brief Set particle velocities using current rigid velocities for 
  Lees-Edwards boundary conditions (assuming position of particles has
  not changed)
 */
inline void set_Particle_Velocities_OBL(Particle *p){
  int rigidID, sign;
  double r[DIM];
  double delta_vx;
#pragma omp parallel for private(rigidID, sign, r, delta_vx)
  for(int n=0; n<Particle_Number; n++){
    rigidID = Particle_RigidID[n];
    for(int d=0; d<DIM; d++){
      p[n].omega[d] = omegaGs[rigidID][d];
      r[d] = xGs[rigidID][d] + GRvecs[n][d];
    }
    rigid_Velocity(p[n].v, GRvecs[n], velocityGs[rigidID], omegaGs[rigidID]);
    sign = PBC_OBL(r, delta_vx);
    p[n].v[0] += delta_vx;
  }
}
inline void init_set_vGs(Particle *p){
  for(int rigidID=0; rigidID < Rigid_Number; rigidID++){
    rigid_body_rotation(omegaGs[rigidID], p[Rigid_Particle_Cumul[rigidID]].q, BODY2SPACE);
  }
  
  if(SW_EQ != Shear_Navier_Stokes_Lees_Edwards){
    set_Particle_Velocities(p);
  }else{
    set_Particle_Velocities_OBL(p);
  }
}

/*!
  \brief rotate GRvecs and Rigig_Moments to match current orientation of rigid body
 */
inline void update_Orientation(Particle *p){
  int rigid_first_n;
  int rigid_last_n;
  quaternion rigidQ;
#pragma omp parallel for private(rigidQ, rigid_first_n, rigid_last_n)
  for(int rigidID = 0; rigidID < Rigid_Number; rigidID++){
    rigid_first_n = Rigid_Particle_Cumul[rigidID];
    rigid_last_n = Rigid_Particle_Cumul[rigidID+1];
    qtn_init(rigidQ, p[rigid_first_n].q);
    
    for(int n = rigid_first_n; n < rigid_last_n; n++){
      rigid_body_rotation(GRvecs[n], GRvecs_body[n], rigidQ, BODY2SPACE);
    }
    rigid_body_matrix_rotation(Rigid_Moments[rigidID][0], Rigid_Moments_body[rigidID][0], 
                               rigidQ, BODY2SPACE);
    Matrix_Inverse(Rigid_Moments[rigidID], Rigid_IMoments[rigidID], DIM);
    check_Inverse(Rigid_Moments[rigidID], Rigid_IMoments[rigidID], DIM);
  }
}

/*! 
  \brief Set masses and inertia tensors for current rigid body configurations
*/
inline void set_Rigid_MMs(Particle *p){
  double dmy_mass, dmy_moi;
#pragma omp parallel for private(dmy_mass, dmy_moi)
  for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
    
    //initialize
    Rigid_Masses[rigidID] = 0.0;
    for(int row=0; row<DIM; row++){
      for(int column=0; column<DIM; column++){
        Rigid_Moments[rigidID][row][column] = 0.0;
        Rigid_IMoments[rigidID][row][column] = 0.0;
      }
    }
    
    dmy_mass = MASS[ RigidID_Components[rigidID] ];
    dmy_moi = MOI[ RigidID_Components[rigidID] ];
    
    //add individual particles contributions to each rigidID
    for(int n=Rigid_Particle_Cumul[rigidID]; n<Rigid_Particle_Cumul[rigidID+1]; n++){
      double &ri_x = GRvecs[n][0];
      double &ri_y = GRvecs[n][1];
      double &ri_z = GRvecs[n][2];
      
      Rigid_Masses[rigidID] += dmy_mass;
      
      Rigid_Moments[rigidID][0][0] += dmy_moi + dmy_mass * ( ri_y*ri_y + ri_z*ri_z );
      Rigid_Moments[rigidID][0][1] += dmy_mass * ( -ri_x*ri_y );
      Rigid_Moments[rigidID][0][2] += dmy_mass * ( -ri_x*ri_z );
      
      Rigid_Moments[rigidID][1][0] += dmy_mass * ( -ri_y*ri_x );
      Rigid_Moments[rigidID][1][1] += dmy_moi + dmy_mass * ( ri_z*ri_z + ri_x*ri_x );
      Rigid_Moments[rigidID][1][2] += dmy_mass * ( -ri_y*ri_z );
      
      Rigid_Moments[rigidID][2][0] += dmy_mass * ( -ri_z*ri_x );
      Rigid_Moments[rigidID][2][1] += dmy_mass * ( -ri_z*ri_y );
      Rigid_Moments[rigidID][2][2] += dmy_moi + dmy_mass * ( ri_x*ri_x + ri_y*ri_y );
    }
    
    //inverse
    Rigid_IMasses[rigidID] = 1.0 / Rigid_Masses[rigidID];
    Matrix_Inverse(Rigid_Moments[rigidID], Rigid_IMoments[rigidID], DIM);
    check_Inverse(Rigid_Moments[rigidID], Rigid_IMoments[rigidID], DIM);
  }
}

/*!
  \brief Update position and orientation of rigid particles
 */
inline void solver_Rigid_Position(Particle *p, const CTime &jikan, string CASE){

  int rigid_first_n;
  int rigid_last_n;
  double delta_x;
  if(CASE == "Euler"){
#pragma omp parallel for private(rigid_first_n, rigid_last_n, delta_x)
    for(int rigidID=0; rigidID<Rigid_Number; rigidID++){

      //center of mass
      for(int d=0; d<DIM; d++){
        xGs_previous[rigidID][d] = xGs[rigidID][d];

        delta_x = jikan.dt_md * velocityGs[rigidID][d];
        xGs[rigidID][d] += delta_x;
        xGs_nopbc[rigidID][d] += delta_x;
      }
      PBC(xGs[rigidID]);

      //orientation
      rigid_first_n = Rigid_Particle_Cumul[rigidID];
      rigid_last_n = Rigid_Particle_Cumul[rigidID+1];
      MD_solver_orientation_Euler(p[rigid_first_n], jikan.dt_md);

      //broadcast new orientation to all beads
      for(int n=rigid_first_n+1; n<rigid_last_n; n++){
        qtn_init(p[n].q_old, p[rigid_first_n].q_old);
        qtn_init(p[n].q, p[rigid_first_n].q);
      }
    }

  }else if(CASE == "AB2"){
#pragma omp parallel for private(rigid_first_n, rigid_last_n, delta_x)
    for(int rigidID=0; rigidID<Rigid_Number; rigidID++){

      //center of mass
      for(int d=0; d<DIM; d++){
        xGs_previous[rigidID][d] = xGs[rigidID][d];

        delta_x = jikan.hdt_md * (3.0 * velocityGs[rigidID][d] - velocityGs_old[rigidID][d]);
        xGs[rigidID][d] += delta_x;
        xGs_nopbc[rigidID][d] += delta_x;
      }
      PBC(xGs[rigidID]);
      
      //orientation
      rigid_first_n = Rigid_Particle_Cumul[rigidID];
      rigid_last_n = Rigid_Particle_Cumul[rigidID+1];
      MD_solver_orientation_AB2(p[rigid_first_n], jikan.hdt_md);
      
      //broadcast new orientatiion to all beads
      for(int n=rigid_first_n+1; n<rigid_last_n; n++){
        qtn_init(p[n].q_old, p[rigid_first_n].q_old);
        qtn_init(p[n].q, p[rigid_first_n].q);
      }
    }

  }else{
    fprintf(stderr, "# error: string CASE in solver_Rigid_Position\n");
    exit_job(EXIT_FAILURE);
  }

  //update relative vectors to bead positions
  update_Orientation(p);
}

/*!
  \brief Update position and orientatino of rigid particles under
  Lees-Edwards boundary conditions
 */
inline void solver_Rigid_Position_OBL(Particle *p, const CTime &jikan, string CASE){

  int sign;
  int rigid_first_n;
  int rigid_last_n;
  double delta_x, delta_vx;
  if(CASE == "Euler"){
#pragma omp parallel for private(rigid_first_n, rigid_last_n, sign, delta_x, delta_vx)
    for(int rigidID=0; rigidID<Rigid_Number; rigidID++){

      //center of mass
      for(int d=0; d<DIM; d++){
        xGs_previous[rigidID][d] = xGs[rigidID][d];

        delta_x = jikan.dt_md * velocityGs[rigidID][d];
        xGs[rigidID][d] += delta_x;
        xGs_nopbc[rigidID][d] += delta_x;
      }
      sign = PBC_OBL(xGs[rigidID], delta_vx);
      velocityGs[rigidID][0] += delta_vx;
      velocityGs_old[rigidID][0] += delta_vx;

      //orientation
      rigid_first_n = Rigid_Particle_Cumul[rigidID];
      rigid_last_n = Rigid_Particle_Cumul[rigidID+1];
      MD_solver_orientation_Euler(p[rigid_first_n], jikan.dt_md);

      //broadcast new orientation to all beads
      for(int n=rigid_first_n+1; n<rigid_last_n; n++){
        qtn_init(p[n].q_old, p[rigid_first_n].q_old);
        qtn_init(p[n].q, p[rigid_first_n].q);
      }
    }

  }else if(CASE == "AB2"){
#pragma omp parallel for private(rigid_first_n, rigid_last_n, sign, delta_x, delta_vx)
    for(int rigidID=0; rigidID<Rigid_Number; rigidID++){

      //center of mass
      for(int d=0; d<DIM; d++){
        xGs_previous[rigidID][d] = xGs[rigidID][d];

        delta_x = jikan.hdt_md * (3.0 * velocityGs[rigidID][d] - velocityGs_old[rigidID][d]);
        xGs[rigidID][d] += delta_x;
        xGs_nopbc[rigidID][d] += delta_x;
      }
      sign = PBC_OBL(xGs[rigidID], delta_vx);
      velocityGs[rigidID][0] += delta_vx;
      velocityGs_old[rigidID][0] += delta_vx;
      
      //orientation
      rigid_first_n = Rigid_Particle_Cumul[rigidID];
      rigid_last_n = Rigid_Particle_Cumul[rigidID+1];
      MD_solver_orientation_AB2(p[rigid_first_n], jikan.hdt_md);
      
      //broadcast new orientatiion to all beads
      for(int n=rigid_first_n+1; n<rigid_last_n; n++){
        qtn_init(p[n].q_old, p[rigid_first_n].q_old);
        qtn_init(p[n].q, p[rigid_first_n].q);
      }
    }

  }else{
    fprintf(stderr, "# error: string CASE in solver_Rigid_Position_OBL\n");
    exit_job(EXIT_FAILURE);
  }

  //update relative vectors to bead positions
  update_Orientation(p);
}

/*!
  \brief Update rigid velocities (VelocityGs) and angular velocities (OmegaGs)
  \note set_Rigid_VOGs() after calculating xGs, Rigid_IMoments, forceGs and torqueGs!!
*/
inline void calc_Rigid_VOGs(Particle *p, const CTime &jikan, string CASE){
#pragma omp parallel for
  for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
    //set olds
    for(int d=0; d<DIM; d++){
      velocityGs_old[rigidID][d] = velocityGs[rigidID][d];
      omegaGs_old[rigidID][d] = omegaGs[rigidID][d];
    }
  }
  
  //calc velocityGs and omegaGs
  if(CASE == "Euler"){
#pragma omp parallel for
    for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
      int rigid_spec = RigidID_Components[rigidID];      

      { // com velocity
	double dmy = jikan.dt_md * Rigid_IMasses[rigidID];
	
	for(int d = 0; d < DIM; d++){
	  if(Rigid_Motions_vel[rigid_spec][d] == 0) continue; // if "fix"
	  velocityGs[rigidID][d] += (dmy * (forceGs[rigidID][d] + forceGrs[rigidID][d]));
	}
      }

      { // angular velocity
	double torque[DIM] = {torqueGs[rigidID][0] + torqueGrs[rigidID][0],
			      torqueGs[rigidID][1] + torqueGrs[rigidID][1],
			      torqueGs[rigidID][2] + torqueGrs[rigidID][2]};
	double Ibody[DIM] = {Rigid_Moments_body[rigidID][0][0],
			     Rigid_Moments_body[rigidID][1][1],
			     Rigid_Moments_body[rigidID][2][2]};
	int rigid_first = Rigid_Particle_Cumul[rigidID];
	
	MD_solver_omega_Euler(omegaGs[rigidID], torque,
			      Ibody, p[rigid_first].q,
			      Rigid_Motions_omega[rigid_spec], jikan.dt_md);
      }
    }
  }else if(CASE == "AB2"){
#pragma omp parallel for 
    for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
      int rigid_spec = RigidID_Components[rigidID];
      
      { // com velocity
	double dmy = jikan.hdt_md * Rigid_IMasses[rigidID];
	
	for(int d = 0; d < DIM; d++){
	  if(Rigid_Motions_vel[rigid_spec][d] == 0) continue; // if "fix"
	  velocityGs[rigidID][d] += dmy * (2.0 * forceGs[rigidID][d]
					   + forceGrs[rigidID][d] + forceGrs_previous[rigidID][d]);
	}
      }
      
      { // angular velocity
	double torque[DIM] = {2.0*torqueGs[rigidID][0] + torqueGrs[rigidID][0],
			      2.0*torqueGs[rigidID][1] + torqueGrs[rigidID][1],
			      2.0*torqueGs[rigidID][2] + torqueGrs[rigidID][2]};
	double torque_prev[DIM] = {torqueGrs_previous[rigidID][0],
				   torqueGrs_previous[rigidID][1],
				   torqueGrs_previous[rigidID][2]};
	double Ibody[DIM] = {Rigid_Moments_body[rigidID][0][0],
			     Rigid_Moments_body[rigidID][1][1],
			     Rigid_Moments_body[rigidID][2][2]};
	int rigid_first = Rigid_Particle_Cumul[rigidID];
	
	MD_solver_omega_AB2(omegaGs[rigidID], torque, torque_prev,
			    Ibody, p[rigid_first].q, p[rigid_first].q_old,
			    Rigid_Motions_omega[rigid_spec], jikan.hdt_md);
      }
    }
  }else{
    fprintf(stderr, "error, string CASE in calc_Rigid_VOGs()");
    exit_job(EXIT_FAILURE);
  }
  
  // renew old previous and initialize
#pragma omp parallel for
  for(int rigidID=0; rigidID<Rigid_Number; rigidID++){
    for(int d=0; d<DIM; d++){
      forceGs_previous[rigidID][d] = forceGs[rigidID][d];
      forceGrs_previous[rigidID][d] = forceGrs[rigidID][d];

      torqueGs_previous[rigidID][d] = torqueGs[rigidID][d];
      torqueGrs_previous[rigidID][d] = torqueGrs[rigidID][d];

      forceGs[rigidID][d] = 0.0;
      forceGrs[rigidID][d] = 0.0;

      torqueGs[rigidID][d] = 0.0;
      torqueGrs[rigidID][d] = 0.0;
    }
  }
}
#endif
