/*!
  \file operate_dielectric.h
  \author T. Oguri
  \date 2018/09/14
  \version 1
  \brief distribution to compute potential distribution in non-uniform dielectric and forces (header file)  
*/
#ifndef OPERATE_DIELECTRIC_H
#define OPERATE_DIELECTRIC_H

#include "fluid_solver.h"
#include "solute_rhs.h"
#include "operate_electrolyte.h"
#include "LinearOperator.h"

#include <math.h>
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> _VectorReplacement;
typedef Eigen::Map<_VectorReplacement> VectorReplacement;

extern double *Potential; 
extern double *epsilon;
extern double **grad_epsilon; 
extern int have_surface_charge;

inline void rm_external_electric_field_x(double *potential
            ,const CTime &jikan
            ){
  double external[DIM];
  for(int d=0;d<DIM;d++){
    external[d] = E_ext[d];
    if(AC){
      double time = jikan.time;
      external[d] *= sin( Angular_Frequency * time);
    }
  }

  int im;
#pragma omp parallel for private(im)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
    im=(i*NY*NZ_)+(j*NZ_)+k;
  potential[im] += (
             (external[0] * (double)i 
        + external[1] * (double)j 
        + external[2] * (double)k
        ) 
             * DX);
      }
    }
  }
}

void Mem_alloc_potential(void);
void Interpolate_vec_on_normal_grid(double **vec);
void Interpolate_scalar_on_staggered_grid(double **vec, double *scalar);

void Make_phi_sin_primitive(Particle *p, 
                            double *phi_p,
                            const int &SW_UP,
                            const double &dx,
                            const int &np_domain,
                            int **sekibun_cell,
                            const int Nlattice[DIM],
                            const double radius = RADIUS);

void Make_phi_sin(double *phi_sin,
                  Particle *p,
                  const double radius = RADIUS);

/*!
  Compute the solute source term appearing in the rhs of the Navier-Stokes equation (including possible contributions from the external field) in the case of non-uniform dielectric field
  \details
  \f[
  \rho_{\text{free}}\vec{E} - \frac{1}{2}|\vec{E}|^2\nabla\varepsilon
  \f]
  \param[in] force Maxwell force exerterd on the fluid due to the presence of the charged solute and dielectric object
  \param[in] p particle data
  \param[in] conc_k solute concentration field (reciprocal space)
  \param[out] charge_density auxiliary field to compute total charge density
  \param[out] jikan time data
 */
void Make_Maxwell_force_x_on_fluid(double **force,
								                   Particle *p,
								                   double **conc_k,
								                   double *free_charge_density,
                                   double *potential,
							                	   const CTime &jikan);

/*!
  \brief Computes the total charge density field (real space)
  \details Computes the total charge density field  by adding the colloid surface charge density (for the current particle positions) to the solute charge density, where the total charge density is given by
  \f[
  \rho_e = (1-\phi)\sum_\alpha Z_\alpha e C_\alpha 
  \f]
  in the case that particles do not have initial charges
  \param[in] p particle data
  \param[in] conc_k solute concentration field (reciprocal space) 
  \param[out] charge_density total charge density (real space)
  \param[in,out] phi auxiliary field to compute the smooth profile
  \param[in,out] dmy_value auxiliary field to compute the concentration of a single solute species
 */
void Conc_k2charge_field_no_surfase_charge(Particle *p,
                                           double **conc_k,
                                           double *charge_density,
                                           double *phi,
                                           double *dmy_value); // working memory

void Conc_k2charge_field_surface_charge(Particle *p, 
                                        double *charge_density,
                                        double *phi_p); // working memory

void Make_janus_normal_dielectric(double r[DIM], const Particle &p);

void Make_janus_permittivity_dielectric(Particle *p, 
                                        const double eps_p_top,
                                        const double eps_p_bottom,
                                        double *eps,
                                        double *phi_p);

void Make_dielectric_field(Particle *p, 
                           double *eps, 
                           double *phi_p, 
                           const double eps_p_top, 
                           const double eps_p_bottom, 
                           const double eps_f);

void insertCoefficient_x(int id, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, Eigen::VectorXd& eps_eigen, double external_e_field[]);
void insertCoefficient_y(int id, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, Eigen::VectorXd& eps_eigen, double external_e_field[]);
void insertCoefficient_z(int id, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, Eigen::VectorXd& eps_eigen, double external_e_field[]);

/*!
  \brief Constract A and b in Ax=b, with finite deference method
  \details Construct 2D m*m sparse matrix A and 1D m vector b, in Ax=b
  \f[
  \vec{x}=A^{-1}\vec{b}
  \f]
  \param[in] p particle data
  \param[in] conc_k solute concentration field (reciprocal space) 
  \param[out] charge_density total charge density (real space)
  \param[in,out] phi auxiliary field to compute the smooth profile
  \param[in,out] dmy_value auxiliary field to compute the concentration of a single solute species
 */
void buildProblem(std::vector<T>& coefficients, 
                  Eigen::VectorXd& b, 
                  Eigen::VectorXd& eps_eigen, 
                  double external_e_field[],
                  const int m);

/*!
  \brief Solve linear equation Ax=b (real space, staggered grid)
  \details Compute the electric potential by solving linear equation Ax=b from Poisson eq. 
  \f[
  \vec{x}=A^{-1}\vec{b}
  \f]
  \param[in] p particle data
  \param[in] conc_k solute concentration field (reciprocal space) 
  \param[out] charge_density total charge density (real space)
  \param[in,out] phi auxiliary field to compute the smooth profile
  \param[in,out] dmy_value auxiliary field to compute the concentration of a single solute species
 */
void Charge_field2potential_dielectric(double *free_charge_density,
                                       double *potential,
                                       double *eps,
                                       double external_e_field[]);

void test_Ax_FD(double *x,
                double *Ax,
                double *rhs_b,
                double *free_charge_density,
                double *eps,
                double external_e_field[]);
void test_ax(double *x, double *Ax, double *eps);

/*!
  \brief Constract A in Ax=b, with pseudo spectral method
  \details Construct 2D m*m sparse matrix A and 1D m vector b, in Ax=b
  \f[
  \vec{x}=A^{-1}\vec{b}
  \f]
  \param[in] p particle data
  \param[in] conc_k solute concentration field (reciprocal space) 
  \param[out] charge_density total charge density (real space)
  \param[in,out] phi auxiliary field to compute the smooth profile
  \param[in,out] dmy_value auxiliary field to compute the concentration of a single solute species
 */
void Ax(const double *x, double*Ax);

/*!
  \brief Constract b in Ax=b, with pseudo spectral method
  \details Construct 2D m*m sparse matrix A and 1D m vector b, in Ax=b
  \f[
  \vec{x}=A^{-1}\vec{b}
  \f]
  \param[in] p particle data
  \param[in] conc_k solute concentration field (reciprocal space) 
  \param[out] charge_density total charge density (real space)
  \param[in,out] phi auxiliary field to compute the smooth profile
  \param[in,out] dmy_value auxiliary field to compute the concentration of a single solute species
 */
void Build_rhs(double *rhs, double *free_charge_density, double *eps, double external_e_field[]);

/*!
  \brief Solve linear equation Ax=b (real space, staggered grid), with a linear operator
  \details Compute the electric potential by solving linear equation Ax=b from Poisson eq. 
  \f[
  \vec{x}=A^{-1}\vec{b}
  \f]
  \param[in] p particle data
  \param[in] conc_k solute concentration field (reciprocal space) 
  \param[out] charge_density total charge density (real space)
  \param[in,out] phi auxiliary field to compute the smooth profile
  \param[in,out] dmy_value auxiliary field to compute the concentration of a single solute species
 */
void Charge_field2potential_dielectric_LO(double *free_charge_density,
                                          double *potential,
                                          double *eps,
                                          double *rhs, //working memory
                                          double external_e_field[]);

void Init_potential_dielectric(Particle *p,
                               double *dmy_value1, //working memory
                               double *dmy_value2, //working memory
                               double **conc_k,
                               double *free_charge_density, // working memory
                               double *potential,
                               double *epsilon, 
                               const CTime &jikan);

void Make_Maxwell_force_x_on_fluid(double **force,
                                   Particle *p,
                                   double **conc_k,
                                   double *free_charge_density, // working memory
                                   double *potential,
                                   double *epsilon, 
                                   double **grad_epsilon, 
                                   const CTime &jikan);

template<typename T>inline void Copy_vec_to_eigenvec_1d(const double *vec, T& eigenvec){
  int im, im_;
#pragma omp parallel for private(im_,im)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im_ = (i*NY*NZ_)+(j*NZ_)+k;
        im  = (i*NY*NZ )+(j*NZ )+k;
        eigenvec[im] = vec[im_];
      }
    }
  }
}
template<typename T>inline void Copy_eigenvec_to_vec_1d(const T eigenvec, double *vec){
  int im, im_;
#pragma omp parallel for private(im_,im)
  for(int i=0;i<NX;i++){
    for(int j=0;j<NY;j++){
      for(int k=0;k<NZ;k++){
        im_ = (i*NY*NZ_)+(j*NZ_)+k;
        im  = (i*NY*NZ )+(j*NZ )+k;
        vec[im_] = eigenvec[im];
      }
    }
  }
}

#endif
