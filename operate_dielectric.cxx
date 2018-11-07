/*!
  \file operate_dielectric.cxx
  \author T. Oguri
  \date 2018/09/14
  \version 1
  \brief Routines to compute potential distribution and forces in a non-uniform dielectric fiels (header file)  
*/
#include "operate_dielectric.h"

double *Potential;
double **grad_epsilon; 
int have_surface_charge;
Eigen::VectorXd b;                   // the right hand side-vector resulting from the constraints
Eigen::VectorXd dmy_eps;             // this is not a cool way, should be modified
Eigen::VectorXd x_ans;


void Mem_alloc_potential(void){
  Potential    = alloc_1d_double(NX*NY*NZ_);
  grad_epsilon = (double **) malloc(sizeof(double *) * DIM);
  for(int d=0; d<DIM; d++) grad_epsilon[d] = alloc_1d_double(NX*NY*NZ_);
  b            = Eigen::VectorXd::Zero(NX*NY*NZ);             // the right hand side-vector resulting from the constraints
  dmy_eps      = Eigen::VectorXd::Zero(NX*NY*NZ);             // this is not a cool way, should be modified
  x_ans        = Eigen::VectorXd::Zero(NX*NY*NZ);
}
void print_error_index(int index_here, int range){
  if(index_here<0 || index_here>=range)
    fprintf(stderr,"# index is out of range\n");
}

void Make_phi_sin_primitive(Particle *p, 
                  double *phi,
                  const int &SW_UP,
                  const double &dx,
                  const int &np_domain,
                  int **sekibun_cell,
                  const int Nlattice[DIM],
                  const double radius){
  double xp[DIM],vp[DIM],omega_p[DIM];
    int x_int[DIM];
    double residue[DIM];
    int sw_in_cell; 
    int r_mesh[DIM];
    double r[DIM];
    double x[DIM];
    double dmy;
    double dmy_phi;
    int im;

#pragma omp parallel for \
  private(xp,vp,omega_p,x_int,residue,sw_in_cell,r_mesh,r,x,dmy,dmy_phi,im)
    for(int n=0; n < Particle_Number; n++){
    for(int d=0;d<DIM;d++){
        xp[d] = p[n].x[d];
    }
  
  sw_in_cell 
      = Particle_cell(xp, dx, x_int, residue);// {1,0} が返ってくる
  sw_in_cell = 1;
  for(int mesh=0; mesh < np_domain; mesh++){
      Relative_coord(sekibun_cell[mesh]
         , x_int, residue, sw_in_cell, Nlattice, dx, r_mesh, r);
      for(int d=0;d<DIM;d++){
        x[d] = r_mesh[d] * dx;
      }
      dmy = Distance(x, xp);
      dmy_phi= Phi_compact_sin(dmy,radius);
      im = (r_mesh[0] * NY * NZ_) + (r_mesh[1] * NZ_) + r_mesh[2]; 
#pragma omp atomic
      phi[im] += dmy_phi;
      }
    }
}

void Make_phi_sin(double *phi_sin
           ,Particle *p
           ,const double radius
           ){
  const int SW_UP = 0;
  int *nlattice;
  nlattice = Ns;
  Make_phi_sin_primitive(p, phi_sin, SW_UP,DX,NP_domain
           ,Sekibun_cell
           ,nlattice
           ,radius);
}

void Conc_k2charge_field_no_surfase_charge(Particle *p,
                                           double **conc_k,
                                           double *charge_density,
                                           double *phi, // working memory
                                           double *dmy_value){ // working memory
	int im;
	double dmy_conc;
  Reset_phi(phi);
  Make_phi_particle(phi, p);
  for(int n=0;n< N_spec;n++){
    A_k2a_out(conc_k[n], dmy_value);
#pragma omp parallel for private(im,dmy_conc)
    for(int i=0;i<NX;i++){
	    for(int j=0;j<NY;j++){
	      for(int k=0;k<NZ;k++){
		      im=(i*NY*NZ_)+(j*NZ_)+k;
	        dmy_conc = dmy_value[im];
	        charge_density[im] += Valency_e[n] * dmy_conc * (1.- phi[im]);
	      }
      }
    }
  }
}

void Conc_k2charge_field_surface_charge(Particle *p
       ,double *charge_density
       ,double *phi_p // working memory
       ){
  {
    Reset_phi(phi_p);
    Reset_phi(charge_density);
    Make_phi_qq_particle(phi_p, charge_density, p);
  }
}

void Make_janus_normal_dielectric(double r[DIM], const Particle &p){
  double norm_r;
  double ex[DIM] = {1.0, 0.0, 0.0};
  double ey[DIM] = {0.0, 1.0, 0.0};
  double ez[DIM] = {0.0, 0.0, 1.0};
  double *e1, *e2, *e3;

  // determine janus (e3) axis
  int dmy_axis = janus_axis[p.spec];
  if(dmy_axis == z_axis){
    e1 = ex;
    e2 = ey;
    e3 = ez;
  }else if(dmy_axis == y_axis){
    e1 = ez;
    e2 = ex;
    e3 = ey;
  }else if(dmy_axis == x_axis){
    e1 = ey;
    e2 = ez;
    e3 = ex;
  }else{
    fprintf(stderr, "Error: Invalid Janus axis for Spherical_coord\n");
    exit_job(EXIT_FAILURE);
  }
    //r normal vector
  rigid_body_rotation(r, e3, p.q, BODY2SPACE);
  norm_r = sqrt( SQ(r[0]) + SQ(r[1]) + SQ(r[2]) );
  assert(positive_mp(norm_r));
  double dmy_norm = 1.0/norm_r;
  for(int d = 0; d < DIM; d++){
    r[d] *= dmy_norm;
  }
}

void Make_janus_permittivity_dielectric(Particle *p, 
                                        const double eps_p_top,
                                        const double eps_p_bottom,
                                        double *eps,
                                        double *phi_p){
    static const double dmy0 = DX3*RHO; 
    int *nlattice;
    nlattice = Ns;

    double xp[DIM];
    int x_int[DIM];
    double residue[DIM];
    int sw_in_cell;
    int r_mesh[DIM];
    double r[DIM];
    double x[DIM];
    double dmyR;
    double dmy_phi;
    double janus_normal[DIM];
    double janus_r;
    double eps_ave = 0.5*(eps_p_top + eps_p_bottom)*Dielectric_cst;
    double eps_delta = 0.5*(eps_p_top - eps_p_bottom)*Dielectric_cst;
    double dmy_tanh;
    double sharpness = 10;
    int im;

#pragma omp parallel for \
  private(xp,x_int,residue,sw_in_cell,r_mesh,r,x,janus_r,dmyR,dmy_phi,dmy_tanh,im) 
    for(int n = 0; n < Particle_Number; n++){
      for (int d = 0; d < DIM; d++) {
        xp[d] = p[n].x[d];
      }
      
      Make_janus_normal_dielectric(janus_normal, p[n]);

      sw_in_cell = Particle_cell(xp, DX, x_int, residue);
      sw_in_cell = 1;
      
      for(int mesh=0; mesh < NP_domain; mesh++){
        janus_r = 0.0;
        Relative_coord(Sekibun_cell[mesh], x_int, residue, sw_in_cell, nlattice, DX, r_mesh, r);
        for(int d=0;d<DIM;d++){
          x[d]  = r_mesh[d] * DX;
          janus_r += r[d] * janus_normal[d];
        }
        dmyR = Distance(x, xp);
        dmy_phi= Phi_compact_sin(dmyR, RADIUS);
  
        im = (r_mesh[0] * NY * NZ_) + (r_mesh[1] * NZ_) + r_mesh[2];

#pragma omp atomic
        phi[im] += dmy_phi;

        dmy_tanh  = std::tanh(sharpness*dmy_phi*janus_r);
        eps[im]   = dmy_phi*(eps_ave + eps_delta*dmy_tanh);
        
      }// mesh
    }//Particle_number
}


void Make_dielectric_field(Particle *p, 
                           double *eps, 
                           double *phi_p, // working memory
                           const double eps_p_top, 
                           const double eps_p_bottom, 
                           const double eps_f){
  int janus_dielectric = 1;
  int im = 0;
  double dmy_phi;
  int dmy_axis = janus_axis[p[0].spec];
  if(dmy_axis != z_axis && dmy_axis != y_axis && dmy_axis != x_axis){
    janus_dielectric = 0;
  }
  Reset_phi(phi_p);
  Reset_phi(eps);
  if(janus_dielectric){
    Make_janus_permittivity_dielectric(p, eps_p_top, eps_p_bottom, eps, phi_p);
#pragma omp parallel for private(im, dmy_phi)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          dmy_phi = 1-phi_p[im];
#pragma omp atomic
          eps[im] += eps_f*dmy_phi*Dielectric_cst;
        }
      }
    }
  }else{
    Make_phi_sin(phi_p, p);
#pragma omp parallel for private(im)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          eps[im] = eps_f*Dielectric_cst + (eps_p_top-eps_f)*Dielectric_cst*phi_p[im];
        }
      }
    }
  }
}

void insertCoefficient_x(int id1, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, Eigen::VectorXd& eps_eigen, double external_e_field[]){
  int i_star = (i_+NX)%NX;
  int id2    = i_star*NY*NZ + j_*NZ + k_;
  print_error_index(id2, NX*NY*NZ);
  b(id1)    += sign * eps_eigen(id2)*external_e_field[0]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps_eigen(id1)+eps_eigen(id2))/(2*DX*DX)));             
  coeffs.push_back(T(id1, id1, -(eps_eigen(id1)+eps_eigen(id2))/(2*DX*DX))); 
}
void insertCoefficient_y(int id1, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, Eigen::VectorXd& eps_eigen, double external_e_field[]){
  int j_star = (j_+NY)%NY;
  int id2    = i_*NY*NZ + j_star*NZ + k_;
  print_error_index(id2, NX*NY*NZ);
  b(id1)    += sign * eps_eigen(id2)*external_e_field[1]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps_eigen(id1)+eps_eigen(id2))/(2*DX*DX)));
  coeffs.push_back(T(id1, id1, -(eps_eigen(id1)+eps_eigen(id2))/(2*DX*DX))); 
}
void insertCoefficient_z(int id1, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, Eigen::VectorXd& eps_eigen, double external_e_field[]){
  int k_star = (k_+NZ)%NZ;
  int id2    = i_*NY*NZ + j_*NZ + k_star;
  print_error_index(id2, NX*NY*NZ);
  b(id1)    += sign * eps_eigen(id2)*external_e_field[2]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps_eigen(id1)+eps_eigen(id2))/(2*DX*DX)));
  coeffs.push_back(T(id1, id1, -(eps_eigen(id1)+eps_eigen(id2))/(2*DX*DX))); 
}
void buildProblem(std::vector<T>& coefficients, 
                  Eigen::VectorXd& b, 
                  Eigen::VectorXd& eps_eigen, 
                  double external_e_field[],
                  const int m){
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        int im=(i*NY*NZ)+(j*NZ)+k;
        insertCoefficient_x(im, i+1, j  , k  , +1.0, b, coefficients, eps_eigen, external_e_field);
        insertCoefficient_x(im, i-1, j  , k  , -1.0, b, coefficients, eps_eigen, external_e_field);
        insertCoefficient_y(im, i  , j+1, k  , +1.0, b, coefficients, eps_eigen, external_e_field);
        insertCoefficient_y(im, i  , j-1, k  , -1.0, b, coefficients, eps_eigen, external_e_field);
        insertCoefficient_z(im, i  , j  , k+1, +1.0, b, coefficients, eps_eigen, external_e_field);
        insertCoefficient_z(im, i  , j  , k-1, -1.0, b, coefficients, eps_eigen, external_e_field);
      }
    }
  }
}

void Charge_field2potential_dielectric(double *free_charge_density,
                                       double *potential,
                                       double *eps,
                                       double external_e_field[]){
  int m = NX*NY*NZ;
  int im, im_;
  std::vector<T> coefficients;            // list of non-zeros coefficients
#pragma omp parallel for private(im, im_)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im  = i*NY*NZ  + j*NZ  +k;
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        b(im)       = -free_charge_density[im_];
        dmy_eps(im) = eps[im_];
        x_ans(im)   = potential[im_];
      }
    }
  }
  buildProblem(coefficients, b, dmy_eps, external_e_field, m);
  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());
  // solve
  Eigen::GMRES<SpMat> gmres(A);
  gmres.setMaxIterations(MaxIter_potential);
  gmres.setTolerance(Tol_potential);
  gmres.solveWithGuess(b, x_ans);
  x_ans = gmres.solve(b);
  if(gmres.iterations()==MaxIter_potential){
    //fprintf(stderr,"# iterations     :%d\n",gmres.iterations());
    fprintf(stderr,"# estimated error:%e\n",gmres.error());
  }
#pragma omp parallel for private(im, im_)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        im  = i*NY*NZ  + j*NZ  +k;
        potential[im_] = x_ans(im);
      }
    }
  }
}

void Charge_field2potential_dielectric_without_solute(double *free_charge_density,
                                       double *potential,
                                       double *eps,
                                       double external_e_field[]){
  int m = NX*NY*NZ;
  int im, im_;
  std::vector<T> coefficients;            // list of non-zeros coefficients
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im  = i*NY*NZ  + j*NZ  +k;
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        print_error_index(im , NX*NY*NZ );
        print_error_index(im_, NX*NY*NZ_);
        b(im)       = 0.0;
        dmy_eps(im) = eps[im_];
        x_ans(im)   = potential[im_];
      }
    }
  }
  buildProblem(coefficients, b, dmy_eps, external_e_field, m);
  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());
  // solve
  Eigen::GMRES<SpMat> gmres(A);
  gmres.setMaxIterations(MaxIter_potential);
  gmres.setTolerance(Tol_potential);
  gmres.solveWithGuess(b, x_ans);
  x_ans = gmres.solve(b);
  if(gmres.iterations()==MaxIter_potential){
    //fprintf(stderr,"# iterations     :%d\n",gmres.iterations());
    fprintf(stderr,"# without solute, estimated error:%e\n",gmres.error());
  }
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        potential[im_] += x_ans(im);
      }
    }
  }
}

void Init_potential_dielectric(Particle *p,
                               double *dmy_value, //working memory
                               double **conc_k,
                               double *free_charge_density, // working memory
                               double *potential,
                               double *epsilon, // working memory
                               const CTime &jikan){ 
  double external[DIM];
  for(int d=0;d<DIM;d++){
    external[d] = E_ext[d];
    if(AC){
      double time = jikan.time;
      external[d] *= sin(Angular_Frequency * time);
    }
  }
  // fill with 0 as as initial guess x0
  /*
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        int im=(i*NY*NZ_)+(j*NZ_)+k;
        potential[im] = 0.0;
      }
    }
  }
  */
  
  // compute free charge density, assuming particles do not have initial charges
  fprintf(stderr,"# calculate an initial value of Potential\n");
  fprintf(stderr,"# set max iterations : %d\n", MaxIter_potential);
  fprintf(stderr,"# set tolerance : %e\n", Tol_potential);
  if(have_surface_charge){
    Conc_k2charge_field(p, conc_k, free_charge_density, dmy_value, epsilon);
    Make_dielectric_field(p, epsilon, dmy_value, eps_particle_top, eps_particle_bottom, eps_fluid);
    //double dmy_external[DIM] = {0.0, 0.0, 0.0};
    Charge_field2potential_dielectric(free_charge_density, potential, epsilon, external);
    //Charge_field2potential_dielectric_without_solute(free_charge_density, potential, epsilon, external);
    //rm_external_electric_field_x(potential, jikan);
  }else{
    Conc_k2charge_field_no_surfase_charge(p, conc_k, free_charge_density, dmy_value, epsilon);
    Make_dielectric_field(p, epsilon, dmy_value, eps_particle_top, eps_particle_bottom, eps_fluid);
    Charge_field2potential_dielectric(free_charge_density, potential, epsilon, external);
  }
  //A2a_k_out(free_charge_density, potential);
  //Charge_field_k2Coulomb_potential_k_PBC(potential);
  //A_k2a(potential);
  // compute the non-uniform dielectric field 'ep' and the gradient of it, 'dep'
  fprintf(stderr,"# solved\n");
  fprintf(stderr,"############################");
}

void Make_Maxwell_force_x_on_fluid(double **force,
                                   Particle *p,
                                   double **conc_k,
                                   double *free_charge_density, // working memory
                                   double *potential,
                                   double *epsilon, // working memory
                                   double **grad_epsilon, 
                                   const CTime &jikan){

  // set the external electric field
  double external[DIM];
  for(int d=0;d<DIM;d++){
    external[d] = E_ext[d];
    if(AC){
      double time = jikan.time;
      external[d] *= sin(Angular_Frequency * time);
    }
  }
  
  // compute free charge density, assuming particles do not have initial charges
  if(have_surface_charge){
    // calculate potential from surface charge
    Conc_k2charge_field(p, conc_k, free_charge_density, force[0], force[1]);
    Make_dielectric_field(p, epsilon, force[0], eps_particle_top, eps_particle_bottom, eps_fluid);
    //double dmy_external[DIM] = {0.0, 0.0, 0.0};
    Charge_field2potential_dielectric(free_charge_density, potential, epsilon, external);
    //Charge_field2potential_dielectric_without_solute(free_charge_density, potential, epsilon, external);
    //rm_external_electric_field_x(potential, jikan);
  }else{
    Conc_k2charge_field_no_surfase_charge(p, conc_k, free_charge_density, force[0], force[1]);
    // compute the non-uniform dielectric field 'ep' and the gradient of it, 'dep'
    Make_dielectric_field(p, epsilon, force[0], eps_particle_top, eps_particle_bottom, eps_fluid);
    // poisson solver in a non-uniform dielectric
    Charge_field2potential_dielectric(free_charge_density, potential, epsilon, external);
  }

  // compute the gradient of the permittivity field on the staggered grid
  {
    A2a_k(epsilon);
    A_k2da_k(epsilon, grad_epsilon);
    Shift_vec_fw_imag(grad_epsilon);
    A_k2a(epsilon);
    U_k2u(grad_epsilon);
  }
  // compute the electric field from the electric potential on the staggered grid
  {
    A2a_k(potential);
    A_k2da_k(potential, force);
    Shift_vec_fw_imag(force);
    A_k2a(potential);
    U_k2u(force);
  }
  Reset_phi(work_v1);

  int im;
  double electric_field_x_fw, electric_field_y_fw, electric_field_z_fw;
  double electric_field_x_bw, electric_field_y_bw, electric_field_z_bw;
  double electric_field_square;
  double rho_free_interpolate;
  double E2_interpolate;

#pragma omp parallel
  {
//#pragma omp parallel for private(im,electric_field)
#pragma omp  for private(im, electric_field_x_fw, electric_field_y_fw, electric_field_z_fw, \
                             electric_field_x_bw, electric_field_y_bw, electric_field_z_bw, \
                             electric_field_square)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          electric_field_x_fw = -force[0][im];                   //(i+1/2,j    ,k    )
          electric_field_y_fw = -force[1][im];                   //(i    ,j+1/2,k    )
          electric_field_z_fw = -force[2][im];                   //(i    ,j    ,k+1/2)
          electric_field_x_bw = -force[0][extract_id(i-1,j,k)];  //(i-1/2,j    ,k    )
          electric_field_y_bw = -force[1][extract_id(i,j-1,k)];  //(i    ,j-1/2,k    )
          electric_field_z_bw = -force[2][extract_id(i,j,k-1)];  //(i    ,j    ,k-1/2)
          if(External_field){
            electric_field_x_fw += external[0];
            electric_field_y_fw += external[1];
            electric_field_z_fw += external[2];
            electric_field_x_bw += external[0];
            electric_field_y_bw += external[1];
            electric_field_z_bw += external[2];
          }
          electric_field_square =  (SQ(electric_field_x_fw) + SQ(electric_field_x_bw))/2.0
                                  +(SQ(electric_field_y_fw) + SQ(electric_field_y_bw))/2.0
                                  +(SQ(electric_field_z_fw) + SQ(electric_field_z_bw))/2.0;
          work_v1[im] = electric_field_square;
        }
      }
    }

#pragma omp  for private(im, electric_field_x_fw, rho_free_interpolate, E2_interpolate)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          electric_field_x_fw = -force[0][im];
          if(External_field){
            electric_field_x_fw += external[0];
          }
          rho_free_interpolate = (free_charge_density[im] + free_charge_density[extract_id(i+1,j,k)])/2.0;
          E2_interpolate = (work_v1[im] + work_v1[extract_id(i+1,j,k)])/2.0;
          force[0][im] = rho_free_interpolate * electric_field_x_fw - E2_interpolate * grad_epsilon[0][im]/2.0;
        }
      }
    }
#pragma omp  for private(im, electric_field_y_fw, rho_free_interpolate, E2_interpolate)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          electric_field_y_fw = -force[1][im];
          if(External_field){
            electric_field_y_fw += external[1];
          }
          rho_free_interpolate = (free_charge_density[im] + free_charge_density[extract_id(i,j+1,k)])/2.0;
          E2_interpolate = (work_v1[im] + work_v1[extract_id(i,j+1,k)])/2.0;
          force[1][im] = rho_free_interpolate * electric_field_y_fw - E2_interpolate * grad_epsilon[1][im]/2.0;
        }
      }
    }
#pragma omp  for private(im, electric_field_z_fw, rho_free_interpolate, E2_interpolate)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          electric_field_z_fw = -force[2][im];
          if(External_field){
            electric_field_z_fw += external[2];
          }
          rho_free_interpolate = (free_charge_density[im] + free_charge_density[extract_id(i,j,k+1)])/2.0;
          E2_interpolate = (work_v1[im] + work_v1[extract_id(i,j,k+1)])/2.0;
          force[2][im] = rho_free_interpolate * electric_field_z_fw - E2_interpolate * grad_epsilon[2][im]/2.0;
        }
      }
    }

    U2u_k(force);
    Shift_vec_bw_imag(force);
    U_k2u(force);

  }
}
