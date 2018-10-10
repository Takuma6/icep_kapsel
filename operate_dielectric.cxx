/*!
  \file operate_dielectric.cxx
  \author T. Oguri
  \date 2018/09/14
  \version 1
  \brief Routines to compute potential distribution and forces in a non-uniform dielectric fiels (header file)  
*/
#include "operate_dielectric.h"

double *Potential;
int have_surface_charge;
Eigen::VectorXd b;                   // the right hand side-vector resulting from the constraints
Eigen::VectorXd dmy_eps;             // this is not a cool way, should be modified
Eigen::VectorXd x_ans;


void Mem_alloc_potential(void){
  Potential = alloc_1d_double(NX*NY*NZ_);
  b         = Eigen::VectorXd::Zero(NX*NY*NZ);             // the right hand side-vector resulting from the constraints
  dmy_eps   = Eigen::VectorXd::Zero(NX*NY*NZ);             // this is not a cool way, should be modified
  x_ans     = Eigen::VectorXd::Zero(NX*NY*NZ);
}
void print_error_index(int index_here, int range){
  if(index_here<0 || index_here>=range)
    fprintf(stderr,"# index is out of range\n");
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

void Make_dielectric_field(double *eps, double *phi_p, double eps_p, double eps_f){
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        int im=(i*NY*NZ_)+(j*NZ_)+k;
        print_error_index(im, NX*NY*NZ_);
        eps[im] = eps_f*Dielectric_cst + (eps_p-eps_f)*Dielectric_cst*phi_p[im];
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
        print_error_index(im, NX*NY*NZ);
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
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im  = i*NY*NZ  + j*NZ  +k;
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        print_error_index(im , NX*NY*NZ );
        print_error_index(im_, NX*NY*NZ_);
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
  gmres.setMaxIterations(1000);
  gmres.setTolerance(1e-6);
  gmres.solveWithGuess(b, x_ans);
  x_ans = gmres.solve(b);
  fprintf(stderr,"# iterations     :%d\n",gmres.iterations());
  fprintf(stderr,"# estimated error:%e\n",gmres.error());
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        im  = i*NY*NZ  + j*NZ  +k;
        print_error_index(im , NX*NY*NZ );
        print_error_index(im_, NX*NY*NZ_);
        potential[im_] = x_ans(im);
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
  Conc_k2charge_field_no_surfase_charge(p, conc_k, free_charge_density, dmy_value, epsilon);
  A2a_k_out(free_charge_density, potential);
  Charge_field_k2Coulomb_potential_k_PBC(potential);
  A_k2a(potential);
  // compute the non-uniform dielectric field 'ep' and the gradient of it, 'dep'
  Make_dielectric_field(epsilon, dmy_value, eps_particle, eps_fluid);
  Charge_field2potential_dielectric(free_charge_density, potential, epsilon, external);
  fprintf(stderr,"# solved\n");
}

void Make_Maxwell_force_x_on_fluid(double **force,
                                   Particle *p,
                                   double **conc_k,
                                   double *free_charge_density, // working memory
                                   double *potential,
                                   double *epsilon, // working memory
                                   double **grad_epsilon, // working memory
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
  Conc_k2charge_field_no_surfase_charge(p, conc_k, free_charge_density, force[0], force[1]);
  // compute the non-uniform dielectric field 'ep' and the gradient of it, 'dep'
  Make_dielectric_field(epsilon, force[0], eps_particle, eps_fluid);
  // poisson solver in a non-uniform dielectric
  Charge_field2potential_dielectric(free_charge_density, potential, epsilon, external);

  A2a_k(epsilon);
  A_k2da_k(epsilon, grad_epsilon);
  /* staggered
  for(int im; im<NX*NY*NZ; im++){
    grad_epsilon[0][im] *= Shift_x[im];
    grad_epsilon[1][im] *= Shift_y[im];
    grad_epsilon[2][im] *= Shift_z[im];
  }
  */
  A_k2a(epsilon);
  U_k2u(grad_epsilon);
  // compute the electric field from the electric potential (take care about the staggered grid)
  A2a_k(potential);
  A_k2da_k(potential, force);
  /* staggered
  for(int im; im<NX*NY*NZ; im++){
    force[0][im] *= Shift_x[im];
    force[1][im] *= Shift_y[im];
    force[2][im] *= Shift_z[im];
  }
  */
  A_k2a(potential);
  U_k2u(force);

  int im;
  double electric_field_x, electric_field_y, electric_field_z;
  double electric_field_square;
#pragma omp parallel
  {
//#pragma omp parallel for private(im,electric_field)
#pragma omp  for private(im)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          print_error_index(im, NX*NY*NZ_);
          electric_field_x = -force[0][im];
          electric_field_y = -force[1][im];
          electric_field_z = -force[2][im];
          if(External_field){
            electric_field_x += external[0];
            electric_field_y += external[1];
            electric_field_z += external[2];
          }
          electric_field_square = SQ(electric_field_x)+SQ(electric_field_y)+SQ(electric_field_z);
          force[0][im] = free_charge_density[im] * electric_field_x - electric_field_square * grad_epsilon[0][im]/2.0;
          force[1][im] = free_charge_density[im] * electric_field_y - electric_field_square * grad_epsilon[1][im]/2.0;
          force[2][im] = free_charge_density[im] * electric_field_z - electric_field_square * grad_epsilon[2][im]/2.0; 
        }
      }
    }
  }
}
