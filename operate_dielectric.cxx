/*!
  \file operate_dielectric.cxx
  \author T. Oguri
  \date 2018/09/14
  \version 1
  \brief Routines to compute potential distribution and forces in a non-uniform dielectric fiels (header file)  
*/
#include "operate_dielectric.h"

double *Potential;

void Mem_alloc_potential(void){
  Potential = alloc_1d_double(NX*NY*NZ_);
}

void Conc_k2charge_field_no_surfase_charge(Particle *p,
                                           double **conc_k,
                                           double *charge_density,
                                           double *phi, // working memory
                                           double *dmy_value){ // working memory
	int im;
	double dmy_conc;

  for(int n=0;n< N_spec;n++){
    A_k2a_out(conc_k[n], dmy_value);
#pragma omp parallel for private(im,dmy_phi,dmy_conc)
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
  for(int i = 0; i < NX; i++){
    for(int j = 0; j < NY; j++){
      for(int k = 0; k < NZ; k++){
        int im  = i*NY*NZ + j*NZ + k;
        eps[im] = eps_f*Dielectric_cst + (eps_p-eps_f)*Dielectric_cst*phi_p[im];
      }
    }
  }
}

void insertCoefficient_x(int id1, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, double *eps, double external_e_field[]){
  int i_star = (i_+NX)%NX;
  int id2    = i_star*NY*NZ + j_*NZ + k_;
  b[id1]    += sign * eps[id2]*external_e_field[0]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps[id1]+eps[id2])/(2*DX*DX)));             
  coeffs.push_back(T(id1, id1, -(eps[id1]+eps[id2])/(2*DX*DX))); 
}
void insertCoefficient_y(int id1, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, double *eps, double external_e_field[]){
  int j_star = (j_+NY)%NY;
  int id2    = i_*NY*NZ + j_star*NZ + k_;
  b[id1]    += sign * eps[id2]*external_e_field[1]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps[id1]+eps[id2])/(2*DX*DX)));
  coeffs.push_back(T(id1, id1, -(eps[id1]+eps[id2])/(2*DX*DX))); 
}
void insertCoefficient_z(int id1, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, double *eps, double external_e_field[]){
  int k_star = (k_+NZ)%NZ;
  int id2    = i_*NY*NZ + j_*NZ + k_star;
  b[id1]    += sign * eps[id2]*external_e_field[2]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps[id1]+eps[id2])/(2*DX*DX)));
  coeffs.push_back(T(id1, id1, -(eps[id1]+eps[id2])/(2*DX*DX))); 
}
void buildProblem(std::vector<T>& coefficients, 
                  Eigen::VectorXd& b, 
                  double *eps, 
                  double external_e_field[],
                  const int m){
  for(int i = 0; i < NX; i++){
    for(int j = 0; j < NY; j++){
      for(int k = 0; k < NZ; k++){
        int im = i*NY*NZ + j*NZ + k;
        insertCoefficient_x(im, i+1, j  , k  , +1.0, b, coefficients, eps, external_e_field);
        insertCoefficient_x(im, i-1, j  , k  , -1.0, b, coefficients, eps, external_e_field);
        insertCoefficient_y(im, i  , j+1, k  , +1.0, b, coefficients, eps, external_e_field);
        insertCoefficient_y(im, i  , j-1, k  , -1.0, b, coefficients, eps, external_e_field);
        insertCoefficient_z(im, i  , j  , k+1, +1.0, b, coefficients, eps, external_e_field);
        insertCoefficient_z(im, i  , j  , k-1, -1.0, b, coefficients, eps, external_e_field);
      }
    }
  }
}

void Charge_field2potential_dielectric(double *free_charge_density,
                                       double *potential,
                                       double *eps,
                                       double external_e_field[]){
  // Assembly:
  int m = NX*NY*NZ;
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  Eigen::VectorXd x_ans(m);
  for(int im=0; im<m; im++){
    b[im] = -free_charge_density[im];
    x_ans[im] = potential[im];
  }
  buildProblem(coefficients, b, eps, external_e_field, m);
  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());
  // solve
  Eigen::GMRES<SpMat> gmres(A);
  gmres.setMaxIterations(2000);
  gmres.setTolerance(1e-6);
  gmres.solveWithGuess(b, x_ans);
  x_ans = gmres.solve(b);
  //std::cout << "#iterations:     " << gmres.iterations() << std::endl;
  //std::cout << "estimated error: " << gmres.error()      << std::endl;
  for(int im=0; im<m; im++)
    potential[im] = x_ans[im];
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
  for(int im; im<NX*NY*NZ; im++){
    potential[im] = 0.0;
  }
  // compute free charge density, assuming particles do not have initial charges
  Conc_k2charge_field(p, conc_k, free_charge_density, dmy_value, epsilon);
  // compute the non-uniform dielectric field 'ep' and the gradient of it, 'dep'
  Make_dielectric_field(epsilon, dmy_value, eps_particle, eps_fluid);
  Charge_field2potential_dielectric(free_charge_density, potential, epsilon, external);
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
  Conc_k2charge_field(p, conc_k, free_charge_density, force[0], force[1]);
  // compute the non-uniform dielectric field 'ep' and the gradient of it, 'dep'
  Make_dielectric_field(epsilon, force[0], eps_particle, eps_fluid);
  //A2da(epsilon, grad_epsilon);
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
  
  // poisson solver in a non-uniform dielectric
  Charge_field2potential_dielectric(free_charge_density, potential, epsilon, external);
  // compute the electric field from the electric potential (take care about the staggered grid)
  //A2da(potential, force);
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
