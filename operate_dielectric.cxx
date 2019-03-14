/*!
  \file operate_dielectric.cxx
  \author T. Oguri
  \date 2018/12/12
  \version 1
  \brief Routines to compute potential distribution and forces in a non-uniform dielectric fiels (header file)  
*/
#include "operate_dielectric.h"

double *Potential;
double *epsilon;
double **grad_epsilon; 
int have_surface_charge;
MatrixReplacement A_LO; 

// You can choose Finite Defference Method or Pseudo Spectral Method
// Please set it in the function 'Init_potential_dielectric'
void (*Charge_field2potential_dielectric)(double *free_charge_density,
                                          double *potential,
                                          double *eps,
                                          double *rhs, //working memory
                                          double external_e_field[]);

void Mem_alloc_potential(void){
  Potential    = alloc_1d_double(NX*NY*NZ_);
  epsilon      = alloc_1d_double(NX*NY*NZ_);
  grad_epsilon = (double **) malloc(sizeof(double *) * DIM);
  for(int d=0; d<DIM; d++) grad_epsilon[d] = alloc_1d_double(NX*NY*NZ_);
  A_LO.attachMatrix(NX*NY*NZ, &Ax);
}

void Interpolate_vec_on_normal_grid(double **vec){
  int im;
  double vec_x_fw, vec_y_fw, vec_z_fw; 
  double vec_x_bw, vec_y_bw, vec_z_bw; 
  Reset_u(work_v3);
  Copy_v3(work_v3, vec);
#pragma omp parallel for private(im, vec_x_fw, vec_y_fw, vec_z_fw, vec_x_bw, vec_y_bw, vec_z_bw) 
  for(int i=0; i<NX; i++){
    for(int j=0; j<NY; j++){
      for(int k=0; k<NZ; k++){
        im=(i*NY*NZ_)+(j*NZ_)+k;
        vec_x_fw = 0.5*work_v3[0][im];
        vec_x_bw = 0.5*work_v3[0][extract_id(i-1,j,k)];
        vec_y_fw = 0.5*work_v3[1][im];
        vec_y_bw = 0.5*work_v3[1][extract_id(i,j-1,k)];
        vec_z_fw = 0.5*work_v3[2][im];
        vec_z_bw = 0.5*work_v3[2][extract_id(i,j,k-1)];
        vec[0][im] = (vec_x_fw + vec_x_bw);
        vec[1][im] = (vec_y_fw + vec_y_bw);
        vec[2][im] = (vec_z_fw + vec_z_bw);
      }
    }
  }
}

void Interpolate_scalar_on_staggered_grid(double **vec, double *scalar, double factor[]){
  int im;
  double scalar_x_fw, scalar_y_fw, scalar_z_fw, scalar_newtral; 
  Reset_u(vec);
#pragma omp parallel for private(im, scalar_x_fw, scalar_y_fw, scalar_z_fw, scalar_newtral) 
  for(int i=0; i<NX; i++){
    for(int j=0; j<NY; j++){
      for(int k=0; k<NZ; k++){
        im=(i*NY*NZ_)+(j*NZ_)+k;
        scalar_newtral = 0.5*scalar[im];                         //(i    ,j    ,k    )
        scalar_x_fw    = 0.5*scalar[extract_id(i+1,j,k)];        //(i+1  ,j    ,k    )
        scalar_y_fw    = 0.5*scalar[extract_id(i,j+1,k)];        //(i    ,j+1  ,k    )
        scalar_z_fw    = 0.5*scalar[extract_id(i,j,k+1)];        //(i    ,j    ,k+1  )
        vec[0][im] = (scalar_x_fw + scalar_newtral)*factor[0];   //(i+1/2,j    ,k    )
        vec[1][im] = (scalar_y_fw + scalar_newtral)*factor[1];   //(i    ,j+1/2,k    )
        vec[2][im] = (scalar_z_fw + scalar_newtral)*factor[2];   //(i    ,j    ,k+1/2)
      }
    }
  }
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
                                           double *phi_p, // working memory
                                           double *dmy_value){ // working memory
	int im;
  double dmy_phi;
	double dmy_conc;
  //Reset_phi(phi);
  Reset_phi(charge_density);
  //Make_phi_particle(phi, p);
  for(int n=0;n< N_spec;n++){
    A_k2a_out(conc_k[n], dmy_value);
#pragma omp parallel for private(im,dmy_phi,dmy_conc)
    for(int i=0;i<NX;i++){
	    for(int j=0;j<NY;j++){
	      for(int k=0;k<NZ;k++){
		      im=(i*NY*NZ_)+(j*NZ_)+k;
          dmy_phi    = phi_p[im];
	        dmy_conc   = dmy_value[im];
	        charge_density[im] += Valency_e[n] * dmy_conc * (1.-dmy_phi);
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
    double sharpness = 1/DX;
    int im;

#pragma omp parallel for \
  private(xp,x_int,residue,sw_in_cell,r_mesh,r,x,janus_normal,janus_r,dmyR,dmy_phi,dmy_tanh,im) 
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
        phi_p[im] += dmy_phi;

        dmy_tanh  = std::tanh(sharpness*janus_r);
#pragma omp atomic
        eps[im]  += dmy_phi*(eps_ave + eps_delta*dmy_tanh);
        
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
  b[id1]    += sign * eps_eigen[id2]*external_e_field[0]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps_eigen[id1]+eps_eigen[id2])/(2*DX*DX)));             
  coeffs.push_back(T(id1, id1, -(eps_eigen[id1]+eps_eigen[id2])/(2*DX*DX))); 
}
void insertCoefficient_y(int id1, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, Eigen::VectorXd& eps_eigen, double external_e_field[]){
  int j_star = (j_+NY)%NY;
  int id2    = i_*NY*NZ + j_star*NZ + k_;
  b[id1]    += sign * eps_eigen[id2]*external_e_field[1]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps_eigen[id1]+eps_eigen[id2])/(2*DX*DX)));
  coeffs.push_back(T(id1, id1, -(eps_eigen[id1]+eps_eigen[id2])/(2*DX*DX))); 
}
void insertCoefficient_z(int id1, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, Eigen::VectorXd& eps_eigen, double external_e_field[]){
  int k_star = (k_+NZ)%NZ;
  int id2    = i_*NY*NZ + j_*NZ + k_star;
  b[id1]    += sign * eps_eigen[id2]*external_e_field[2]/(2*DX);
  coeffs.push_back(T(id1, id2,  (eps_eigen[id1]+eps_eigen[id2])/(2*DX*DX)));
  coeffs.push_back(T(id1, id1, -(eps_eigen[id1]+eps_eigen[id2])/(2*DX*DX))); 
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

void test_Ax_FD(double *x,
                double *Ax,
                double *rhs_b,
                double *free_charge_density,
                double *eps,
                double external_e_field[]){
  int m = NX*NY*NZ;
  int im, im_;
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(NX*NY*NZ);
  Eigen::VectorXd dmy_eps(NX*NY*NZ);
  Eigen::VectorXd x_guess(NX*NY*NZ);
  Eigen::VectorXd x_ans(NX*NY*NZ);
#pragma omp parallel for private(im, im_)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im  = i*NY*NZ  + j*NZ  +k;
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        b[im]       = -free_charge_density[im_];
        dmy_eps[im] = eps[im_];
        x_guess[im] = x[im_];
      }
    }
  }
  buildProblem(coefficients, b, dmy_eps, external_e_field, m);
  SpMat A_FD(m,m);
  A_FD.setFromTriplets(coefficients.begin(), coefficients.end());
  x_ans = A_FD*x_guess;
#pragma omp parallel for private(im_, im)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        im  = i*NY*NZ  + j*NZ  +k;
        Ax[im_] = x_ans[im];
        rhs_b[im_] = b[im];
      }
    }
  }
}

void Charge_field2potential_dielectric_FD(double *free_charge_density,
                                          double *potential,
                                          double *eps,
                                          double *rhs, //working memory
                                          double external_e_field[]){
  int m = NX*NY*NZ;
  int im, im_;
  std::vector<T> coefficients; 
  Eigen::VectorXd b(NX*NY*NZ);
  Eigen::VectorXd dmy_eps(NX*NY*NZ);
  Eigen::VectorXd x_guess(NX*NY*NZ);
  Eigen::VectorXd x_ans(NX*NY*NZ);
#pragma omp parallel for private(im, im_)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im  = i*NY*NZ  + j*NZ  +k;
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        b[im]       = -free_charge_density[im_];
        dmy_eps[im] = eps[im_];
        x_guess[im] = potential[im_];
      }
    }
  }
  buildProblem(coefficients, b, dmy_eps, external_e_field, m);
  SpMat A_FD(m,m);
  A_FD.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::BiCGSTAB<SpMat> solver(A_FD);
  //Eigen::GMRES<SpMat> solver(A_FD);
  solver.setMaxIterations(MaxIter_potential);
  solver.setTolerance(Tol_potential);
  x_ans = solver.solveWithGuess(b, x_guess);
  if(solver.iterations()==MaxIter_potential){
    //fprintf(stderr,"# iterations     :%d\n",gmres.iterations());
    fprintf(stderr,"# estimated error:%e\n",solver.error());
    fprintf(stderr,"# solve again without initial guess\n");
    x_ans = solver.solve(b);
    fprintf(stderr,"# iterations     :%d\n",solver.iterations());
    fprintf(stderr,"# estimated error:%e\n",solver.error());
  }
#pragma omp parallel for private(im_, im)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im_ =(i*NY*NZ_)+(j*NZ_)+k;
        im  = i*NY*NZ  + j*NZ  +k;
        potential[im_] = x_ans[im];
      }
    }
  }
}

void test_ax(double *x, double *Ax, double *eps){
  Reset_phi_u(work_v1, work_v3);
  Copy_v1(work_v1, x);
  A2a_k(work_v1);
  A_k2da_k(work_v1, work_v3);
  Shift_vec_fw_imag(work_v3);
  U_k2u(work_v3);
  int im, im_;
  double eps_neutral, eps_x_fw, eps_y_fw, eps_z_fw;
#pragma omp parallel for private(im, eps_neutral, eps_x_fw, eps_y_fw, eps_z_fw)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im=(i*NY*NZ_)+(j*NZ_)+k;
        eps_neutral = epsilon[im];
        eps_x_fw    = epsilon[extract_id(i+1,j,k)];
        eps_y_fw    = epsilon[extract_id(i,j+1,k)];
        eps_z_fw    = epsilon[extract_id(i,j,k+1)];
        work_v3[0][im] *= 0.5*(eps_x_fw+eps_neutral);
        work_v3[1][im] *= 0.5*(eps_y_fw+eps_neutral);
        work_v3[2][im] *= 0.5*(eps_z_fw+eps_neutral);
      }
    }
  }
  U2u_k(work_v3);
  U_k2divergence_k_shift(work_v3, work_v1);
  A_k2a(work_v1);
#pragma omp parallel for private(im, im_)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im =(i*NY*NZ )+(j*NZ )+k;
        im_=(i*NY*NZ_)+(j*NZ_)+k;
        Ax[im_] = work_v1[im_];
      }
    }
  }
}

void Ax(const double *x, double*Ax){
  Reset_phi_u(work_v1, work_v3);
  Copy_eigenvec_to_vec_1d(x, work_v1);
  A2a_k(work_v1);
  A_k2da_k(work_v1, work_v3);
  Shift_vec_fw_imag(work_v3);
  U_k2u(work_v3);
  int im, im_;
  double eps_neutral, eps_x_fw, eps_y_fw, eps_z_fw;
#pragma omp parallel for private(im, eps_neutral, eps_x_fw, eps_y_fw, eps_z_fw)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im=(i*NY*NZ_)+(j*NZ_)+k;
        eps_neutral = epsilon[im];
        eps_x_fw    = epsilon[extract_id(i+1,j,k)];
        eps_y_fw    = epsilon[extract_id(i,j+1,k)];
        eps_z_fw    = epsilon[extract_id(i,j,k+1)];
        work_v3[0][im] *= 0.5*(eps_x_fw+eps_neutral);
        work_v3[1][im] *= 0.5*(eps_y_fw+eps_neutral);
        work_v3[2][im] *= 0.5*(eps_z_fw+eps_neutral);
      }
    }
  }
  U2u_k(work_v3);
  U_k2divergence_k_shift(work_v3, work_v1);
  A_k2a(work_v1);
#pragma omp parallel for private(im, im_)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im =(i*NY*NZ )+(j*NZ )+k;
        im_=(i*NY*NZ_)+(j*NZ_)+k;
        Ax[im] = work_v1[im_];
      }
    }
  }
}

void Build_rhs(double *rhs, double *free_charge_density, double *eps, double external_e_field[]){
  int im;
  Reset_phi(rhs);
  Interpolate_scalar_on_staggered_grid(work_v3, eps, external_e_field);
  U2u_k(work_v3);
  U_k2divergence_k_shift(work_v3, rhs);
  A_k2a(rhs);
#pragma omp parallel for private(im)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im = (i*NY*NZ_)+(j*NZ_)+k;
          rhs[im] -= free_charge_density[im];
        }
      }
    }
}

void Charge_field2potential_dielectric_LO(double *free_charge_density,
                                          double *potential,
                                          double *eps,
                                          double *rhs, //working memory
                                          double external_e_field[]){
  Eigen::VectorXd b        = Eigen::VectorXd::Zero(NX*NY*NZ);
  Eigen::VectorXd x_guess  = Eigen::VectorXd::Zero(NX*NY*NZ);
  Eigen::VectorXd x_ans    = Eigen::VectorXd::Zero(NX*NY*NZ);
  Reset_phi(rhs);
  Build_rhs(rhs, free_charge_density, eps, external_e_field);
  Copy_vec_to_eigenvec_1d(rhs, b);
  Copy_vec_to_eigenvec_1d(potential, x_guess);
  int im, im_;
  //Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> solver;
  Eigen::BiCGSTAB<MatrixReplacement, Eigen::IdentityPreconditioner> solver;
  //Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner > solver;
  //MatrixReplacement A; A.attachMatrix(NX*NY*NZ, &Ax);
  solver.compute(A_LO);
  solver.setTolerance(Tol_potential); 
  solver.setMaxIterations(MaxIter_potential);
  x_ans = solver.solveWithGuess(b, x_guess);
  if(solver.iterations()==MaxIter_potential){
    //fprintf(stderr,"# iterations     :%d\n",solver.iterations());
    fprintf(stderr,"# estimated error:%e\n",solver.error());
    fprintf(stderr,"# solve again without initial guess\n");
    x_ans = solver.solve(b);
    fprintf(stderr,"# iterations     :%d\n",solver.iterations());
    fprintf(stderr,"# estimated error:%e\n",solver.error());
  }
  Copy_eigenvec_to_vec_1d(x_ans, potential);
}

void Init_potential_dielectric(Particle *p,
                               double *dmy_value1, //working memory
                               double *dmy_value2, //working memory
                               double **conc_k,
                               double *free_charge_density, // working memory
                               double *potential,
                               double *epsilon, 
                               const CTime &jikan){ 
  double external[DIM];
  for(int d=0;d<DIM;d++){
    external[d] = E_ext[d];
    if(AC){
      double time = jikan.time;
      external[d] *= sin(Angular_Frequency * time);
    }
  }

  // choose which function to use for potential calculation
  Charge_field2potential_dielectric = Charge_field2potential_dielectric_LO;
  
  fprintf(stderr,"# calculate an initial value of Potential\n");
  fprintf(stderr,"# set max iterations : %d\n", MaxIter_potential);
  fprintf(stderr,"# set tolerance : %e\n", Tol_potential);
  if(have_surface_charge){
    Make_dielectric_field(p, epsilon, dmy_value1, eps_particle_top, eps_particle_bottom, eps_fluid);
    Conc_k2charge_field(p, conc_k, free_charge_density, dmy_value1, dmy_value2);
    Charge_field2potential_dielectric(free_charge_density, potential, epsilon, dmy_value1, external);
  }else{
    Make_dielectric_field(p, epsilon, dmy_value1, eps_particle_top, eps_particle_bottom, eps_fluid);
    Conc_k2charge_field_no_surfase_charge(p, conc_k, free_charge_density, dmy_value1, dmy_value2);
    Charge_field2potential_dielectric(free_charge_density, potential, epsilon, dmy_value1, external);
  }

  fprintf(stderr,"# solved\n");
  fprintf(stderr,"############################\n");
}

void Make_Maxwell_force_x_on_fluid(double **force,
                                   Particle *p,
                                   double **conc_k,
                                   double *free_charge_density, // working memory
                                   double *potential,
                                   double *epsilon, 
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
    Make_dielectric_field(p, epsilon, force[0], eps_particle_top, eps_particle_bottom, eps_fluid);
    Conc_k2charge_field(p, conc_k, free_charge_density, force[0], force[1]);
    Charge_field2potential_dielectric(free_charge_density, potential, epsilon, force[2], external);
  }else{
    // compute the non-uniform dielectric field 'ep' and the gradient of it, 'dep'
    Make_dielectric_field(p, epsilon, force[0], eps_particle_top, eps_particle_bottom, eps_fluid);
    Conc_k2charge_field_no_surfase_charge(p, conc_k, free_charge_density, force[0], force[1]);
    // poisson solver in a non-uniform dielectric
    Charge_field2potential_dielectric(free_charge_density, potential, epsilon, force[2], external);
  }

  // compute the gradient of the permittivity field on the staggered grid
  {
    A2a_k(epsilon);
    A_k2da_k(epsilon, grad_epsilon);
    Shift_vec_fw_imag(grad_epsilon); // shift to staggered
    A_k2a(epsilon);
    U_k2u(grad_epsilon);
  }
  // compute the electric field from the electric potential on the staggered grid
  {
    A2a_k(potential);
    A_k2da_k(potential, force);
    Shift_vec_fw_imag(force); // shift to staggered
    A_k2a(potential);
    U_k2u(force);
  }
  Reset_phi(work_v1);

  int im;
  double electric_field_x_fw, electric_field_y_fw, electric_field_z_fw;
  double electric_field_x_bw, electric_field_y_bw, electric_field_z_bw;
  double electric_field_square;
  double rho_free_fw, rho_free_bw, rho_free;
  double E2_fw, E2_bw, E2;

#pragma omp parallel for private(im, electric_field_x_fw, electric_field_y_fw, electric_field_z_fw, \
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
        electric_field_square  = SQ((electric_field_x_fw + electric_field_x_bw)/2.0) + SQ((electric_field_y_fw + electric_field_y_bw)/2.0) + SQ((electric_field_z_fw + electric_field_z_bw)/2.0);
        //electric_field_square  = (SQ(electric_field_x_fw) + SQ(electric_field_x_bw))/2.0
        //                        +(SQ(electric_field_y_fw) + SQ(electric_field_y_bw))/2.0
        //                        +(SQ(electric_field_z_fw) + SQ(electric_field_z_bw))/2.0;
        work_v1[im] = electric_field_square;
      }
    }
  }

#pragma omp parallel
  {
#pragma omp for private(im, electric_field_x_fw, rho_free_fw, rho_free_bw, rho_free, E2_fw, E2_bw, E2)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          electric_field_x_fw = -force[0][im];
          if(External_field){
            electric_field_x_fw += external[0];
          }
          rho_free_fw = 0.5*free_charge_density[extract_id(i+1,j,k)];
          rho_free_bw = 0.5*free_charge_density[im];
          rho_free    = (rho_free_bw + rho_free_fw);
          E2_fw = 0.5*work_v1[extract_id(i+1,j,k)];
          E2_bw = 0.5*work_v1[im];
          E2    = E2_bw + E2_fw;
          force[0][im] = rho_free * electric_field_x_fw - E2 * grad_epsilon[0][im]/2.0;
        }
      }
    }
#pragma omp for private(im, electric_field_y_fw, rho_free_fw, rho_free_bw, rho_free, E2_fw, E2_bw, E2)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          electric_field_y_fw = -force[1][im];
          if(External_field){
            electric_field_y_fw += external[1];
          }
          rho_free_fw = 0.5*free_charge_density[extract_id(i,j+1,k)];
          rho_free_bw = 0.5*free_charge_density[im];
          rho_free    = (rho_free_bw + rho_free_fw);
          E2_fw = 0.5*work_v1[extract_id(i,j+1,k)];
          E2_bw = 0.5*work_v1[im];
          E2    = E2_bw + E2_fw;
          force[1][im] = rho_free * electric_field_y_fw - E2 * grad_epsilon[1][im]/2.0;
        }
      }
    }
#pragma omp for private(im, electric_field_z_fw, rho_free_fw, rho_free_bw, rho_free, E2_fw, E2_bw, E2)
    for(int i=0;i<NX;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NZ;k++){
          im=(i*NY*NZ_)+(j*NZ_)+k;
          electric_field_z_fw = -force[2][im];
          if(External_field){
            electric_field_z_fw += external[2];
          }
          rho_free_fw = 0.5*free_charge_density[extract_id(i,j,k+1)];
          rho_free_bw = 0.5*free_charge_density[im];
          rho_free    = (rho_free_bw + rho_free_fw);
          E2_fw = 0.5*work_v1[extract_id(i,j,k+1)];
          E2_bw = 0.5*work_v1[im];
          E2    = E2_bw + E2_fw;
          force[2][im] = rho_free * electric_field_z_fw - E2 * grad_epsilon[2][im]/2.0;
        }
      }
    }
  }
  //recover
  Interpolate_vec_on_normal_grid(force);
  Reset_u(grad_epsilon);
  Copy_v3(grad_epsilon, force);
  //Copy_v1(grad_epsilon[0], free_charge_density);
}
