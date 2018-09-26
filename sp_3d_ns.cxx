/*!
  \file sp_3d_ns.cxx
  \author Y. Nakayama
  \date 2006/11/14
  \version 1.5
  \brief Main program file
 */
#include "sp_3d_ns.h"

void (*Time_evolution)(double **zeta, double uk_dc[DIM], double **f, Particle *p, CTime &jikan);

void Time_evolution_noparticle(double **zeta, double uk_dc[DIM], double **f, Particle *p, CTime &jikan){
  const Index_range* ijk_range = ijk_range_two_third_filter;
  const int n_ijk_range = n_ijk_range_two_third_filter;
  if(SW_EQ == Navier_Stokes){
    NS_solver_slavedEuler(zeta, jikan, uk_dc, ijk_range, n_ijk_range, p);
  }else if(SW_EQ == Shear_Navier_Stokes){
    NS_solver_slavedEuler_Shear_PBC(zeta, jikan, uk_dc, ijk_range, n_ijk_range, p, Shear_force);
  }else if(SW_EQ == Electrolyte){
    NSsolute_solver_Euler(zeta, jikan, uk_dc, Concentration,p,ijk_range, n_ijk_range);
  }
}

void Time_evolution_hydro(double **zeta, double uk_dc[DIM], double **f, Particle *p, CTime &jikan){
    
    // Update of Fluid Velocity Field	
    Time_evolution_noparticle(zeta, uk_dc, f, p, jikan);
    
    if(Particle_Number >= 0){
	if(FIX_CELL){ // time-dependent average pressure gradient
	    for(int d=0;d<DIM;d++){
		if(FIX_CELLxyz[d]){
		    uk_dc[d] = 0.0;
		}
	    }
	}
	
	for(int d=0; d<(DIM-1); d++){
	    Truncate_two_third_rule(zeta[d]);
	}
	
	Zeta_k2u(zeta, uk_dc, u);
	
	if(!Fixed_particle){
	    if(jikan.ts == 0){
		MD_solver_position_Euler(p, jikan);
	    }else{
		MD_solver_position_AB2(p, jikan);
	    }
	}
        Reset_phi(phi);
        Reset_phi(phi_sum);
        Make_phi_particle_sum(phi, phi_sum, p);
	
	if(SW_EQ == Electrolyte){
	    double * rescale_factor = new double[N_spec];
	    Rescale_solute(rescale_factor
			   ,Total_solute
			   ,Concentration, p, phi, up[0]);
	    delete [] rescale_factor;
	}
	
	if(SW_EQ == Electrolyte){
	    {
		Reset_u(up);
		Calc_f_hydro_correct_precision(p, phi_sum, u, jikan);
		for(int n=0;n<Particle_Number;n++){
		    for(int d=0;d<DIM;d++){
			p[n].f_hydro1[d] = p[n].f_hydro[d];
			p[n].torque_hydro1[d] = p[n].torque_hydro[d];
		    }
		}
	    }
	    Make_Coulomb_force_x_on_fluid(f, p, Concentration, up[0], up[1], jikan);
	    
	    double dmy = jikan.dt_fluid * IRHO;
#pragma omp parallel for
	    for(int i=0;i<NX;i++){
		for(int j=0;j<NY;j++){
		    for(int k=0;k<NZ;k++){
			u[0][(i*NY*NZ_)+(j*NZ_)+k] += (f[0][(i*NY*NZ_)+(j*NZ_)+k] * dmy);
			u[1][(i*NY*NZ_)+(j*NZ_)+k] += (f[1][(i*NY*NZ_)+(j*NZ_)+k] * dmy);
			u[2][(i*NY*NZ_)+(j*NZ_)+k] += (f[2][(i*NY*NZ_)+(j*NZ_)+k] * dmy);
		    }
		}
	    }
	    Solenoidal_u(u);
	}
	
		
	{// Calculation of hydrodynamic force

	  Reset_u(up);	  
	  Calc_f_hydro_correct_precision(p, phi_sum, u, jikan); //hydrodynamic force

	  if(!SW_JANUS_SLIP){
	    if(!Fixed_particle){
	      if(jikan.ts == 0){
		MD_solver_velocity_Euler(p, jikan);
	      }else{
		MD_solver_velocity_AB2_hydro(p, jikan);
	      }
	    }
	  }else{ // Self-Consistent slip force

	    int slip_converge = 0;
	    int slip_iter = 0;
	    Make_particle_momentum_factor(u, p);         
	    Update_slip_particle_velocity(p, slip_iter); // initial particle velocity for slip profile
	    while(!slip_converge){
	      Reset_u(up);

	      Make_force_u_slip_particle(up, u, p, jikan);
	      Solenoidal_u(up);
	      Calc_f_slip_correct_precision(p, up, jikan); //slip force
	      Add_f_particle(up, u); //up += u

	      // Update particle velocity
	      if(!Fixed_particle){
		if(slip_iter == 0){
		  MD_solver_velocity_slip_iter(p, jikan, start_iter);
		}else{
		  MD_solver_velocity_slip_iter(p, jikan, new_iter);
		}
	      }
              if(PINNING){
                Pinning(p);
              }
	      slip_iter++;

	      if(Slip_particle_convergence(p) < MAX_SLIP_TOL || slip_iter == MAX_SLIP_ITER){
		slip_converge = 1;
		MD_solver_velocity_slip_iter(p, jikan, end_iter);
	      }else{
		Update_slip_particle_velocity(p, slip_iter); // use new particle velocity for new slip profile
		MD_solver_velocity_slip_iter(p, jikan, reset_iter);
	      }
	    }//slip_convergence
	    if(slip_iter == MAX_SLIP_ITER){
	      fprintf(stderr, "#Warning: increase MAX_SLIP_ITER (%d)\n", jikan.ts);
	    }

            Swap_mem(u, up);
	  } // slip 

	}

	if(kBT > 0. && SW_EQ != Electrolyte){
	    Add_random_force_thermostat(p, jikan); 
	}

        if(PINNING){
          Pinning(p);
        }
	
	{
	  //Reset_phi_u(phi, up);
	  //Make_phi_u_particle(phi, up, p);
	  Reset_u(up);
	  Make_u_particle_sum(up, phi_sum, p);
	  Make_f_particle_dt_sole(f, u, up, phi);
	  Add_f_particle(u, f);
	}
	
	
        if(Shear_AC){ 
#pragma omp parallel for  
	    for(int i=0; i<NX; i++){
		for(int j=0; j<NY; j++){
		    for(int k=0; k<NZ; k++){
			ucp[0][(i*NY*NZ_)+(j*NZ_)+k]=u[0][(i*NY*NZ_)+(j*NZ_)+k];
		    }
		}
	    }
	}
	
	U2zeta_k(zeta, uk_dc, u);
    }    
}

void Time_evolution_hydro_OBL(double **zeta, double uk_dc[DIM], double **f, Particle *p, CTime &jikan){
    //AC
    Angular_Frequency = PI2*Shear_frequency;
    //    double ifreq = 1/Angular_Frequency;
    //    double shear_amp = Shear_rate * ifreq;
    //
    const Index_range* ijk_range = ijk_range_two_third_filter;
    const int n_ijk_range = n_ijk_range_two_third_filter;
    NS_solver_slavedEuler_Shear_OBL(zeta, jikan, uk_dc, ijk_range, n_ijk_range, p, Shear_force);
         
    if(Particle_Number >= 0){
	if(FIX_CELL){ // time-dependent average pressure gradient
	    for(int d=0;d<DIM;d++){
		if(FIX_CELLxyz[d]){
		    uk_dc[d] = 0.0;
		}
	    }
	}
	/*
	for(int d = 0; d < DIM - 1; d++){
	    Truncate_two_third_rule(zeta[d]);
	}
	*/

        Zeta_k2u_k_OBL(zeta, uk_dc, u);
        U_k2u(u);
	if(!Shear_AC){
          Shear_rate_eff = Shear_rate;
        }else{
          Shear_rate_eff = Shear_rate * cos(Angular_Frequency*jikan.dt_fluid*jikan.ts);
        }
        degree_oblique += Shear_rate_eff*jikan.dt_fluid;
        if (degree_oblique >= 1.) {
	  int flag =0;
	  Reset_U_OBL(ucp, u,flag);
          Swap_mem(u, ucp);
          degree_oblique -= 1.;
        }
	else if(degree_oblique<0.){
	  int flag = 1;
	  Reset_U_OBL(ucp,u,flag);
	  Swap_mem(u,ucp);
	  degree_oblique +=1.;
	}
        Update_K2_OBL();

        Copy_v3(ucp, u);
        Transform_obl_u(ucp, oblique2cartesian);

        // u   -> velocity field in oblique coordinates
        // ucp -> velocity fild in cartesian coordinates
	//Calc_shear_rate_eff();
	//AC
	if(!Shear_AC) Calc_shear_rate_eff();
	//
	//End Deformation
	
	if(!Fixed_particle){
	    if(jikan.ts == 0){
		MD_solver_position_Euler_OBL(p, jikan);
	    }else{
		MD_solver_position_AB2_OBL(p, jikan);
	    }
	}
	
	{// Calculation of hydrodynamic force
	    Reset_phi(phi);
	    Reset_phi(phi_sum);
	    Reset_u(up);

	    Make_phi_particle_sum_OBL(phi, phi_sum, p);
	    Calc_f_hydro_correct_precision_OBL(p, phi_sum, ucp, jikan);
	    Calc_Reynolds_shear_stress(ucp, Inertia_stress);
	}

	if(!Fixed_particle){// Update of Particle Velocity
	    if(jikan.ts == 0){
		MD_solver_velocity_Euler_OBL(p, jikan);
	    }else{
		MD_solver_velocity_AB2_hydro_OBL(p, jikan);
	    }
	}

	if(kBT > 0. && SW_EQ != Electrolyte){
	    Add_random_force_thermostat(p, jikan); 
	}

        if(PINNING){
          Pinning(p);
        }
	
	{
          Reset_u(up);
          Make_u_particle_sum_OBL(up, phi_sum, p);

          Make_f_particle_dt_nonsole(f, ucp, up, phi);
          Transform_obl_u(f, cartesian2oblique);
          Add_f_particle(u, f);
	}
	
        U2u_k(u);
	Solenoidal_uk_OBL(u);

	U_k2zeta_k_OBL(u, zeta, uk_dc);
    }
}

inline void Mem_alloc_var(double **zeta){

  Mem_alloc_NS_solver();

  if(SW_EQ == Navier_Stokes ||
     SW_EQ == Shear_Navier_Stokes ||
     SW_EQ == Shear_Navier_Stokes_Lees_Edwards){
    ucp = (double **) malloc(sizeof (double *) * DIM);
    for(int d=0; d<DIM; d++){
      ucp[d] = alloc_1d_double(NX*NY*NZ_);
    }
  }else if(SW_EQ == Electrolyte){
    Mem_alloc_charge();
  }
  
  Mem_alloc_f_particle();
  
  for(int d=0;d<DIM-1;d++){
    zeta[d] = alloc_1d_double(NX*NY*NZ_);
  }
  
  u = (double **) malloc(sizeof(double *) * DIM);
  up = (double **) malloc(sizeof(double *) * DIM);
  work_v3 = (double **) malloc(sizeof(double *) * DIM);
  for(int d=0;d<DIM;d++){
    u[d] = alloc_1d_double(NX*NY*NZ_);
    up[d] = alloc_1d_double(NX*NY*NZ_);
    work_v3[d] = alloc_1d_double(NX*NY*NZ_);
  }
  
  work_v2 = (double **) malloc(sizeof(double *) * (DIM - 1));
  for(int d=0; d < DIM - 1; d++){
    work_v2[d] = alloc_1d_double(NX*NY*NZ_);
  }
  
  phi = alloc_1d_double(NX*NY*NZ_);
  phi_sum = alloc_1d_double(NX*NY*NZ_);
  rhop = alloc_1d_double(NX*NY*NZ_);
  work_v1 = alloc_1d_double(NX*NY*NZ_);
  Hydro_force = alloc_1d_double(NX*NY*NZ_);
  Hydro_force_new = alloc_1d_double(NX*NY*NZ_);
}

inline void init_threads(){
#ifdef _OPENMP
  {
    int nthreads, tid, procs, maxt, inpar, dynamic, nested;
#pragma omp parallel private(nthreads, tid)
    {
      tid = omp_get_thread_num();
      if(tid == 0){
	cerr << "#" << endl;
	cerr << "# OMP RUNTIME : " << endl;
	cerr << "# Number of processors        : " << omp_get_thread_num()  << endl;
	cerr << "# Number of threads           : " << omp_get_num_threads() << endl;
	cerr << "# Max OMP threads             : " << omp_get_max_threads() << endl;
#ifdef _FFT_IMKL
	cerr << "# Max MKL threads             : " << mkl_get_max_threads() << endl;
#endif
	cerr << "# Dynamic thread enabled?     : " << omp_get_dynamic()     << endl;
	cerr << "# Nested parallelism enabled? : " << omp_get_nested() << endl;
	cerr << "#" << endl;
      }
    }
  }
#endif
}

int main(int argc, char *argv[]){
#ifndef NDEBUG 
  cerr << "###########################" << endl;
  cerr << "#  " << endl;
  cerr << "# built by debug mode " << endl;
  cerr << "# (NDEBUG is not defined in sp_3d_ns.h)" << endl;
  cerr << "#  " << endl;
  cerr << "###########################" << endl;
#endif
  init_threads();
  wall_timer global_timer, block_timer;

  global_timer.start();
  block_timer.start();
  
  if(argc> 0){
    file_get(argc, argv);
    Gourmet_file_io(In_udf,Out_udf,Sum_udf,Def_udf,Ctrl_udf,Res_udf);
  }
  
  // Main time evolution type 
  if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards) {
    fprintf(stdout, "#Evolution type Shear_Navier_Stokes_Lees_Edwards\n");
    Time_evolution = Time_evolution_hydro_OBL;
  } else {
    if(SW_EQ == Navier_Stokes){
      fprintf(stdout, "#Evolution type Navier_Stokes\n");
    }else if(SW_EQ == Shear_Navier_Stokes){
      fprintf(stdout, "#Evolution type Shear_Navier_Stokes\n");
    }else if(SW_EQ == Electrolyte){
      fprintf(stdout, "#Evolution type Electrolyte\n");
    }
    Time_evolution = Time_evolution_hydro;
  }

  MT_seed(GIVEN_SEED,0);
  //MT_seed(RANDOM_SEED,0);
  Init_fft();
  Init_Transform_obl();
  double uk_dc[DIM];

  double **zeta;
  zeta = (double **) malloc(sizeof(double *) * (DIM-1));
  Mem_alloc_var(zeta);

  static CTime jikan={0, 0.0, DT, DT*0.5, DT, DT*0.5};



  Particle *particles = new Particle [Particle_Number];
  if(Particle_Number > 0){
    Init_Particle(particles);
    //if(SW_PT == chain){
    if((SW_PT == chain) && !(DISTRIBUTION == user_specify)){
      Init_Chain(particles);
    }
    else if((SW_PT == rigid) && !(DISTRIBUTION == user_specify)){
      Init_Rigid(particles);
    }
  }
  Init_output(particles);
  
  Init_zeta_k(zeta, uk_dc);

  {
    Reset_phi_u(phi, up);
    Reset_phi(phi_sum);
    Make_phi_particle_sum(phi, phi_sum, particles);    
    Make_u_particle_sum(up, phi_sum, particles);
    Zeta_k2u(zeta, uk_dc, u);
    
    Make_f_particle_dt_sole(f_particle, u, up, phi);
    Add_f_particle(u, f_particle);
    U2zeta_k(zeta, uk_dc, u);

    if(1){
      for(int d=0;d<DIM;d++){
	uk_dc[d] = .0;
      }
    }
  }
  
  if(SW_EQ==Electrolyte){
    Init_rho_ion(Concentration, particles, jikan);
  }

  //  return EXIT_SUCCESS;

  Show_parameter(particles);
  Show_output_parameter();

  if ((SW_EQ == Shear_Navier_Stokes) || (SW_EQ == Shear_Navier_Stokes_Lees_Edwards)){
    Mean_shear_stress(INIT, stderr, particles, jikan, Shear_rate_eff);
  }else if(SW_EQ == Electrolyte){
    Electrolyte_free_energy(INIT,stderr,particles,Concentration,jikan);
  }

  fprintf(stderr, "# Initialization time (s): %12.3f\n", block_timer.stop());
  block_timer.start();
  
  //////////////////////////////////////////////////////////
  int resumed_ts = 0;
  if(RESUMED) resumed_ts = last_ts;
  for(jikan.ts = resumed_ts; jikan.ts <= MSTEP; jikan.ts++){
    int resumed_and_1st_loop = 0;

    if(RESUMED && (jikan.ts == resumed_ts)){
      resumed_and_1st_loop = 1;
    }
    if( jikan.ts % GTS == 0){
      if(!resumed_and_1st_loop){

	if(SW_OUTFORMAT != OUT_NONE){// Output field & particle data
	  Output_open_frame();
          if(SW_EQ != Electrolyte){
            Output_field_data(zeta, uk_dc, particles, jikan);
          }else if(SW_EQ==Electrolyte){
            Output_charge_field_data(zeta, uk_dc, Concentration, particles, jikan);
          }
	  Output_particle_data(particles, jikan);
	  Output_close_frame();
	}

	if(SW_UDF){// Output_UDF
	  Output_udf(ufout, zeta, uk_dc, particles, jikan);
	}

	if(SW_EQ == Electrolyte){
	    Electrolyte_free_energy(SHOW,stderr,particles, Concentration,jikan);
	}
	if(jikan.ts != resumed_ts){
	  double block_time = block_timer.stop();
	  double global_time = global_timer.stop();
	  fprintf(stderr,
		  "# Step: %9d/%9d\t  Block time (m): %8.3f\t Global time (m): %8.3f/%8.3f\n", 
		  jikan.ts,
		  MSTEP,
		  block_time/60.0,
		  global_time/60.0,
		  ((double)(MSTEP - jikan.ts + 1))/((double)GTS) * block_time/60.0);
	  fflush(stderr);
	  block_timer.start();
	}
      }

    }

    Time_evolution(zeta, uk_dc, f_particle, particles, jikan);
    jikan.time += jikan.dt_fluid;

    if(SW_EQ == Shear_Navier_Stokes){
	Shear_rate_eff = Update_strain(Shear_strain_realized,jikan, zeta, uk_dc, u);
	Mean_shear_stress(SHOW, stderr, particles, jikan, Shear_rate_eff);
    } else if (SW_EQ == Shear_Navier_Stokes_Lees_Edwards){
	Shear_strain_realized += Shear_rate_eff * jikan.dt_fluid;
	Mean_shear_stress(SHOW, stderr, particles, jikan, Shear_rate_eff);
    }

    if(jikan.ts == MSTEP){
      
      Save_Restart_udf(zeta
		       ,uk_dc
		       ,particles
		       ,jikan
		       ,Concentration
		       );
    }

    if(resumed_and_1st_loop){
      Force_restore_parameters(zeta
			       ,uk_dc
			       ,particles
			       ,jikan
			       ,Concentration
			       );
      delete ufin;
      fprintf(stderr,"############################ Parameters are restored.\n");
    }
  }

  if(SW_UDF){
    ufout->write();
    delete ufout;
    fprintf(stderr,"#%s end.\n",Out_udf);
  }
  {
    //Always write restart file
    ufres->write();
    delete ufres;
    fprintf(stderr,"#%s end.\n",Res_udf);
  }
  Free_output();

  {
    double global_time = global_timer.stop();
    fprintf(stderr, "#Simulation has ended!\n");
    fprintf(stderr, "#Total Running Time (s): %10.2f\n", global_time);
    fprintf(stderr, "#                   (m): %10.2f\n", global_time/60.0);
    fprintf(stderr, "#                   (h): %10.2f\n", global_time/3600.0);

    global_time /= (double)(MSTEP - resumed_ts + 1);
    fprintf(stderr, "#Average Step Time  (s): %10.2f\n", global_time);
    fprintf(stderr, "#                   (m): %10.2f\n", global_time/60.0);
    fprintf(stderr, "#                   (h): %10.2f\n", global_time/3600.0);
  }

  Free_Transform_obl();
  Free_fft();
  for(int d = 0; d < DIM - 1; d++){
    free_1d_double(zeta[d]);
  }
  free(zeta);
  delete [] particles;
  return EXIT_SUCCESS;
}
