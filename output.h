/*!
  \file resume.h
  \brief Routines to write output file (header file)
  \author J. Molina
  \date 2014/05/21
  \version 1.0
 */
#ifndef OUTPUT_H
#define OUTPUT_H

#include "input.h"
#include "variable.h"
#include "rigid.h"
#include "fft_wrapper.h"
#include "avs_output.h"
#include "avs_output_p.h"
#ifdef WITH_EXTOUT
#include "output_writer.h"
extern output_writer *writer;
extern hdf5_writer   *h5writer;
#endif

/*!
  \brief Initialize proper output writer
 */
void Init_output(Particle* p);

/*!
  \brief Free any allocated memory for selected writer
 */
void Free_output();

/*!
  \brief Print output configuration to stderr
 */
void Show_output_parameter();

/*!
  \brief Open new time step frame in output
 */
void Output_open_frame();

/*!
  \brief Close currently open time step frame 
 */
void Output_close_frame();

/*!
  \brief Write field data to currently open time step frame
 */
void Output_field_data(double** zeta,
		       double* uk_dc,
		       Particle* p,
		       const CTime &time);

/*!
  \brief Write field data for charged systems to currently open time step frame
 */
void Output_charge_field_data(double** zeta,
			      double* uk_dc,
			      double** Concentration,
			      Particle* p,
			      const CTime &time);

/*!
  \brief Write particle data to currently open time step frame
 */
void Output_particle_data(Particle*p, 
			 const CTime &time);

/*!
  \brief Output particle data for current configuration in UDF format
*/
void Output_udf(UDFManager *ufout,
		  double **zeta,
		  double *uk_dc,
		  const Particle *p,
		  const CTime &time);

#endif
