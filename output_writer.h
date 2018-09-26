#ifndef OUTPUT_WRITER_H
#define OUTPUT_WRITER_H

#include <stdio.h>
#include <string.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <assert.h>
#include <vector>
#include "macro.h"
#include "parameter_define.h"
#include "variable.h"
#include "alloc.h"
#include "output_writer_base.h"

class hdf5_writer : public output_writer {
 public:
  /*!
    \brief Class constructor for hdf5 writer class
    \param[in] _NX points in x
    \param[in] _NY points in y
    \param[in] _NZ points in z
    \param[in] _NZ_ total points in z, including ghost points used in fft computations
    \param[in] _DX sclae for grid spacing
    \param[in] _nump total number of particles
    \param[in] _dt time step between frames (assumes printing at regular intervals)
    \param[in] _out_dir output directory for trajectory files
    \param[in] _out_name name of trajectory files
    \param[in] _field_crop hyperslab selection parameters
    \param[in] _print_field parameters to control how to print field data
   */
  hdf5_writer(const int&     _NX, 
	      const int&     _NY,
	      const int&     _NZ,
	      const int&     _NZ_,
	      const double&  _DX,
	      const int&     _nump,
	      const double&  _dt,
	      const char*    _out_dir,
	      const char*    _out_name,
	      const Field_crop& _field_crop,
	      const Field_output& _print_field,
	      std::vector<int> &_print_particle_list,
	      std::vector<int> &_print_obstacle_list,
	      Particle* p
	      );

  /*!
    \brief Overloaded class destroyer
    \details Make sure to release all allocated memory
   */
  ~hdf5_writer();

  //
  // Functions inherited from virtual base class
  //

  /*! 
    \brief Initialization for writing one frame to disk
  */
  void write_start();

  /*!
    \brief Finish writing current frame to disk
  */
  void write_end();

  /*!
    \brief Write float attribute data to current time frame attributes
   */
  inline void write_frame_attributes(const char* name, const float* attr, 
				     const int& rank){
    herr_t status;
    status = H5LTset_attribute_float(gid_time, "./", name, attr, rank);
    h5_check_err(status);
  }

  /*!
    \brief Write field data to open frame
   */
  void write_field_data(double**u, double* phi, double* pressure,
			double**tau);
  /*!
    \brief Write charged field data to open frame
   */
  void write_charge_field_data(double** u, double*phi, double* surface_charge, 
			       double* solute_charge, double* potential);

  /*!
    \brief Writes particle data to open frame
   */
  void write_particle_data(Particle *p);

  /*!
    \brief Writes obstacle data to open frame
   */
  void write_obstacle_data(Particle *p);

  /*!
    \brief Write output parameters to stderr
  */
  void show_parameter();

 private:
  /*
    Field Data
  */
  int NX;
  int NY;
  int NZ;
  int NZ_;
  float DX;
  float Origin[DIM];

  /*
    Particle Data
  */
  int nump;
  int print_particle_num;
  int print_obstacle_num;
  int* print_particle_list;
  int* print_obstacle_list;


  /*
    Time Data
   */
  int    ts;
  float  dt;

  /*
    Output Options
   */
  Field_crop   crop_field;
  Field_output print_field;
  
  /*
    Output Param
  */
  static const char* f_axis_name[];
  static const char* f_vel_name[];
  static const char* f_tau_name[];
  static const char* f_phi_name;
  static const char* f_pressure_name;
  static const char* f_surface_charge_name;
  static const char* f_solute_charge_name;
  static const char* f_potential_charge_name;
  static const char* p_id_name;
  static const char* p_spec_name;
  static const char* p_pos_name;
  static const char* p_pos_raw_name;
  static const char* p_vel_name;
  static const char* p_omega_name;
  static const char* p_force_h_name;
  static const char* p_force_r_name;
  static const char* p_torque_h_name;
  static const char* p_torque_r_name;
  static const char* p_QR_name;
  char out_path[128];
  char out_name[128];

  /*
    HDF5 Parameters
   */
  char   gid_time_name[128];            //time frame group name
  static const char* gid_sys_data_name; //system data group name
  static const char* gid_trj_data_name; //trajectory data group name
  static const char* gid_field_name;    //field group name
  static const char* gid_part_name;     //particle group name
  static const char* gid_pobs_name;     //obstacle group name

  static const hid_t hid_null;
  hid_t  fid;          //file id
  hid_t  gid_sys_data; //group id for system data
  hid_t  gid_trj_data; //group id for trajectory data
  hid_t  gid_time;     //time frame id
  hid_t  gid_field;    //group id for field data
  hid_t  gid_part;     //group id for particle data
  hid_t  gid_pobs;     //group id for obstacle data

  //Field Output
  static const hsize_t mem_rank_field = 1;
  hsize_t mem_dims_field[mem_rank_field];
  hid_t mem_dataspace_field;

  static const hsize_t out_rank_field = DIM;
  hsize_t out_dims_field[out_rank_field];
  hid_t out_dataspace_field;

  //
  //Private Functions
  //

  /*!
    \brief check h5 error code
   */
  inline void h5_check_err(const herr_t &err){
    if(err < 0){
      fprintf(stderr, "# HDF5 Error:\n");
      H5Eprint(H5E_DEFAULT, stderr);
      exit_job(EXIT_FAILURE);
    }
  }

  /*!
    \brief Writes buffer array to disk
    \param[in] loc_id location id to write to
    \param[in] buffer array to write
    \param[in] name name of new dataset to create
    \param[in] mem_dtype data type in memory
    \param[in] mem_dataspace dataspace to use when reading buffer from memory
    \param[in] out_dtype data type on disk
    \param[in] out_dataspace dataspace to use when writing buffer to disk
   */
  inline void write_data(hid_t loc_id, const void* buffer, const char* name,
			 const hid_t mem_dtype, hid_t &mem_dataspace,
			 const hid_t out_dtype, hid_t &out_dataspace){

    hid_t status;
    hid_t out_dataset = H5Dcreate(loc_id, name, out_dtype, out_dataspace,
				  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check_err(out_dataset);

    status = H5Dwrite(out_dataset, mem_dtype, mem_dataspace, 
		      H5S_ALL, H5P_DEFAULT, buffer);
    h5_check_err(status);

    status = H5Dclose(out_dataset);
    h5_check_err(status);
  }
  /*!
    \brief Writes buffer array to disk
    \param[in] lod_id location id to write to
    \param[in] buffer array to write
    \param[in] name name of new dataset to create
    \param[in] mem_dtype data type in memory
    \param[in] out_dtype data type on disk
    \param[in] out_rank rank of array on disk
    \param[in] out_dim size of each dimension on disk
   */
  inline void write_data(hid_t loc_id, const void* buffer, const char* name,
			 const hid_t mem_dtype, const hid_t out_dtype,
			 const int &out_rank, const hsize_t* out_dim
			 ){
    hid_t status;
    hid_t out_dataspace, out_dataset;

    out_dataspace = H5Screate_simple(out_rank, out_dim, NULL);
    h5_check_err(out_dataspace);

    out_dataset   = H5Dcreate(loc_id, name, out_dtype, out_dataspace, 
			      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check_err(out_dataset);

    status = H5Dwrite(out_dataset, mem_dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    h5_check_err(status);

    status = H5Dclose(out_dataset);
    h5_check_err(status);

    status = H5Sclose(out_dataspace);
    h5_check_err(status);
  }
  
  /*!
    \brief Write scalar field data to current frame
   */
  void write_field_scalar(const double* phi, const char* name, hid_t _loc);

  /*!
    \brief Write scalar particle data to current frame
   */
  void write_particle_scalar(const int* data, const char* name, hid_t loc);

  /*!
    \brief Write vector particle data to current frame
   */
  void write_particle_vectorn(const double* data, const int& dim, const char* name, hid_t loc);

  /*!
    \brief Write 3x3 square matrix particle data to current frame
   */
  void write_particle_matrix3(const double* data, const char* name, hid_t loc);

  /*!
    \brief Write scalar obstacle data to current frame
  */
  void write_obstacle_scalar(const int* data, const char* name, hid_t loc);

  /*!
    \brief Write vector obstacle data to current frame
   */
  void write_obstacle_vectorn(const double* data, const int& dim, const char* name, hid_t loc);

  /*!
    \brief Write 3x3 square matrix obstacle data to current frame
   */
  void write_obstacle_matrix3(const double* data, const char* name, hid_t loc);

  /*!
    \brief Write hdf5/xdmf configuration file
    \details List all data required to build xdmf xml file
   */
  void write_configure_file();
  
  /*!
    \brief Print x,y,z coordiantes for ALL points 
    \details Required to obtain correct visualization using xdmf
   */
  void write_field_info();

  /*!
    \brief Write static particle properties
   */
  void write_particle_info(Particle *p);

  /*!
    \brief Write static obstacle properties
   */
  void write_obstacle_info(Particle *p);
};

#endif
