/*!
  \file avs_output.h
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
  \brief Output routines for field data in AVS/Express format (header file)
 */
#ifndef AVS_OUTPUT_H
#define AVS_OUTPUT_H

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "variable.h"
#include "macro.h"
#include "input.h"
#include "fluid_solver.h"

enum AVS_Field {
    irregular, uniform, rectilinear
};
//////////////////////////////
typedef struct AVS_parameters{
  int nx;
  int ny;
  int nz;
  char out_fld[128];
  char out_cod[128];
  char out_pfx[128];
  char fld_file[128];
  char cod_file[128];

  char data_file[128];

  char out_pfld[128];
  char out_ppfx[128];
  char pfld_file[128];
  int istart;
  int iend;
  int jstart;
  int jend;
  int kstart;
  int kend;
  int nstep;
} AVS_parameters;

extern AVS_parameters Avs_parameters;

/*!
  \brief Print avs parameters to stderr
 */
void Show_avs_parameter();

/*!
  \brief Write AVS .fld definition files
 */
void Init_avs(const AVS_parameters &Avs_parameters);

/*!
  \brief Set global AVS parameters
 */
void Set_avs_parameters(AVS_parameters &Avs_parameters);

/*!
  \brief Output field data for current configuration in AVS format
 */
void Output_avs(AVS_parameters &Avs_parameters, 
		double **u, 
		double *phi, 
		double *Pressure,
		double **strain,
		const CTime &time);


/*!
  \brief Output field data of charged system for current configuration
  in AVS format
 */
void Output_avs_charge(AVS_parameters &Avs_parameters, 
		       double** u, 
		       double* phi, 
		       double* colloid_charge, 
		       double* solute_charge_total,
		       double* potential,
		       const CTime &time);

#endif
