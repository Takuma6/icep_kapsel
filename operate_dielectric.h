/*!
  \file icep.h
  \author T. Oguri
  \date 2018/09/14
  \version 1
  \brief distribution to compute potential distribution in non-uniform dielectric and forces (header file)  
*/
#ifndef OPERATE_DIELECTRIC_H
#define OPERATE_DIELECTRIC_H

#include "fluid_solver.h"
#include "solute_rhs.h"

// include files (add later)
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

//it takes long time to compute Ax=b in an iterative way like GMRES
//so the potential (or electric field) should be kept till the next step, computing advection-diffusion
extern double *Potential; 
extern int Dielectric;
extern double eps_fluid;
extern double eps_particle;



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
							                	   const CTime &jikan,
						                 		   );

void Mem_alloc_potential(void)

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
                                           double *dmy_value, // working memory
                                           );

void Make_dielectric_field(double *eps, double *phi, double eps_p, double eps_f)
void insertCoefficient_x(int id, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, double *eps, double external_e_field);
void insertCoefficient_y(int id, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, double *eps, double external_e_field);
void insertCoefficient_z(int id, int i_, int j_, int k_, double sign, Eigen::VectorXd& b, std::vector<T>& coeffs, double *eps, double external_e_field);

/*!
  \brief Constract A and b in Ax=b
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
                  double *eps, 
                  double external_e_field,
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
                                       double external_e_field);

void Init_potential_dielectric(double **conc_k,
                               double *free_charge_density, // working memory
                               double **potential,
                               double *epsilon)

void Make_Maxwell_force_x_on_fluid(double **force,
                                   Particle *p,
                                   double **conc_k,
                                   double *free_charge_density, // working memory
                                   double *potential,
                                   double *epsilon,
                                   double **grad_epsilon,
                                   const CTime &jikan)
