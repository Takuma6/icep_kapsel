#ifndef OUTPUT_WRITER_BASE_H
#define OUTPUT_WRITER_BASE_H
#include "variable.h"
class output_writer {
 public:

  //destructor of derived classes should free all allocated memory
  virtual ~output_writer(){};

  //virtual function to initialize writing one timestep
  virtual void write_start() = 0;

  //virtual function to finilize writing one timestep
  virtual void write_end() = 0;

  //virtual function to write field data
  virtual void write_field_data(double**u, double*phi, double*pressure, double**tau) = 0;

  //virtual function to write field data for charged systems
  virtual void write_charge_field_data(double**u, double*phi, double*colloid_charge, double*solute_charge, double*potential) = 0;

  //virtual function to write particle data
  virtual void write_particle_data(Particle *p) = 0;

  //virtual function to write obstacle data
  virtual void write_obstacle_data(Particle *p) = 0;

  //virtual function to write output parameters to stderr
  virtual void show_parameter() = 0;
};
#endif
