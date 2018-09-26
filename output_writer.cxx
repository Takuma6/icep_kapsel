#include "output_writer.h"

//fluid properties
const char* hdf5_writer::f_axis_name[]  = {"x", "y", "z"};
const char* hdf5_writer::f_vel_name[] = {"u_x", "u_y", "u_z", "u"};
const char* hdf5_writer::f_tau_name[] = {"tau_xx", "tau_xy", "tau_xz", \
					 "tau_yy", "tau_yz", "tau_zz", "tau"};
const char* hdf5_writer::f_phi_name = "phi";
const char* hdf5_writer::f_pressure_name = "pressure";
const char* hdf5_writer::f_surface_charge_name = "surface_charge";
const char* hdf5_writer::f_solute_charge_name = "solute_charge";
const char* hdf5_writer::f_potential_charge_name= "e_potential";
//particle properties
const char* hdf5_writer::p_id_name = "id";
const char* hdf5_writer::p_spec_name = "spec";
const char* hdf5_writer::p_pos_name = "pos";
const char* hdf5_writer::p_pos_raw_name = "pos_raw";
const char* hdf5_writer::p_vel_name = "vel";
const char* hdf5_writer::p_omega_name = "omega";
const char* hdf5_writer::p_force_h_name = "force_h";
const char* hdf5_writer::p_force_r_name = "force_r";
const char* hdf5_writer::p_torque_h_name= "torque_h";
const char* hdf5_writer::p_torque_r_name= "torque_r";
const char* hdf5_writer::p_QR_name ="QR";

//hdf5 group names
const char* hdf5_writer::gid_sys_data_name="system_data";
const char* hdf5_writer::gid_trj_data_name="trajectory_data";
const char* hdf5_writer::gid_field_name="./field";
const char* hdf5_writer::gid_part_name="./particle";
const char* hdf5_writer::gid_pobs_name="./obstacle";

const hid_t hdf5_writer::hid_null=static_cast<hid_t>(-1);

template<hsize_t, hsize_t> hsize_t* alloc_1d(hsize_t n1);
template<hsize_t> void free_1d(hsize_t* i);

//hdf5 writer Constructor / Destroyer
hdf5_writer::hdf5_writer(const int&    _NX,
			 const int&    _NY,
			 const int&    _NZ,
			 const int&    _NZ_,
			 const double& _DX,
			 const int&    _nump,
			 const double& _dt,
			 const char*   _out_dir,
			 const char*   _out_name,
			 const Field_crop& _crop_field,
			 const Field_output& _print_field,
			 std::vector<int> & _print_particle_list,
			 std::vector<int> & _print_obstacle_list,
			 Particle *p
			 )
{
  //FIELD
  {
    NX   = _NX;
    NY   = _NY;
    NZ   = _NZ;
    NZ_  = _NZ_;
    DX   = static_cast<float>(_DX);
    Origin[0] = _crop_field.start[0]*DX;
    Origin[1] = _crop_field.start[1]*DX;
    Origin[2] = _crop_field.start[2]*DX;
  }
  
  //PARTICLE
  {
    nump               = _nump;
    print_particle_num = _print_particle_list.size();
    print_obstacle_num = _print_obstacle_list.size();
    print_particle_list = (print_particle_num > 0 ? alloc_1d_int(print_particle_num) : NULL);
    print_obstacle_list = (print_obstacle_num > 0 ? alloc_1d_int(print_obstacle_num) : NULL);
    for(int i = 0; i < print_particle_num; i++) print_particle_list[i] = _print_particle_list[i];
    for(int i = 0; i < print_obstacle_num; i++) print_obstacle_list[i] = _print_obstacle_list[i];
  }
  
  //TIME
  {
    ts   = 0;
    dt   = static_cast<float>(_dt);
  }
  
  //OUTPUT OPTIONS
  {
    dircheckmake(_out_dir);
    sprintf(out_name, "%s", _out_name);
    sprintf(out_path, "%s/%s", _out_dir, _out_name);
    crop_field  = _crop_field;
    print_field = _print_field;
  }

  // HDF5 PARAMETERS
  {
    fid = gid_sys_data = gid_trj_data = gid_time = gid_field = gid_part = gid_pobs = -1;
    herr_t status;
    
    //Initialize Memory Dataspace
    {
      mem_dims_field[0] = NX*NY*NZ_;
      mem_dataspace_field = H5Screate_simple(mem_rank_field, mem_dims_field, NULL);
      h5_check_err(mem_dataspace_field);
      status = H5Sselect_none(mem_dataspace_field);
      h5_check_err(status);
      
      int i, j, k;
      hsize_t  im = 0;
      hsize_t* coords = alloc_1d<hsize_t, size_t>(static_cast<hsize_t>(mem_dims_field[0]));
      for(int ii = 0; ii < crop_field.count[0]; ii++){
	i = crop_field.start[0] + ii*crop_field.stride[0];
	
	for(int jj = 0; jj < crop_field.count[1]; jj++){
	  j = crop_field.start[1] + jj*crop_field.stride[1];
	  
	  for(int kk = 0; kk < crop_field.count[2]; kk++){
	    k = crop_field.start[2] + kk*crop_field.stride[2];
	    
	    coords[im++] = (i*NY*NZ_) + (j*NZ_) + k;
	    //status = H5Sselect_elements(mem_dataspace_field, H5S_SELECT_APPEND, 1, &im);
	    //h5_check_err(status);
	  }
	}
      }
      status = H5Sselect_elements(mem_dataspace_field, H5S_SELECT_APPEND, im, coords);
      h5_check_err(status);
      free_1d(coords);
    }

    //Initialize Disk Dataspace
    {
      out_dims_field[0] = crop_field.count[0];
      out_dims_field[1] = crop_field.count[1];
      out_dims_field[2] = crop_field.count[2];
      
      //Disk Acess
      out_dataspace_field = H5Screate_simple(out_rank_field, out_dims_field, NULL);
      h5_check_err(out_dataspace_field);
    }

    //Check consistency between memory and disk spaces
    {
      hsize_t dmy_npoints = H5Sget_select_npoints(mem_dataspace_field);
      assert(dmy_npoints == out_dims_field[0]*out_dims_field[1]*out_dims_field[2]);
    }
  }

  {//Initialize output file
    hid_t status;
    char fname[128];
    sprintf(fname, "%s.h5", out_path);
    fid = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    h5_check_err(fid);
    
    //set global attributes
    {
      status = H5LTset_attribute_float(fid, "/", "dx", &DX, 1);
      h5_check_err(status);

      //dimensions of output field data
      const long long dmy_dims[DIM] = {static_cast<long long>(out_dims_field[0]),
				     static_cast<long long>(out_dims_field[1]),
				     static_cast<long long>(out_dims_field[2])};
      status = H5LTset_attribute_long_long(fid, "/", "nxnynz", dmy_dims, DIM);
      h5_check_err(status);

      //grid origin
      status = H5LTset_attribute_float(fid, "/", "origin", Origin, DIM);
      h5_check_err(status);

      //number of particles in trajectory
      status = H5LTset_attribute_int(fid, "/", "nump", &print_particle_num, 1);
      h5_check_err(status);

      //number of obstacle particles
      status = H5LTset_attribute_int(fid, "/", "nump_obs", &print_obstacle_num, 1);
      h5_check_err(status);

      //total number of particles in system
      status = H5LTset_attribute_int(fid, "/", "nump_total", &nump, 1);
      h5_check_err(status);

      //git veresion information
      status = H5LTset_attribute_string(fid, "/", "git_version", GIT_VERSION);
      h5_check_err(status);

      status = H5LTset_attribute_string(fid, "/", "git_ref", GIT_REF);
      h5_check_err(status);
    }

    //create main groups
    {
      gid_sys_data = H5Gcreate(fid, gid_sys_data_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      h5_check_err(gid_sys_data);

      gid_trj_data = H5Gcreate(fid, gid_trj_data_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      h5_check_err(gid_trj_data);
    }

    {
      status = H5Fflush(fid, H5F_SCOPE_LOCAL);
      h5_check_err(status);
    }
  }

  //print static field properties (grid info required for xdmf format)
  this -> write_field_info();

  //print static particle properties
  this -> write_particle_info(p);

  //print static obstacle properties
  this -> write_obstacle_info(p);

  //print configure file
  this -> write_configure_file();
}

hdf5_writer::~hdf5_writer(){
  //print updated configure file (with time step info)
  this -> write_configure_file();

  herr_t status;
  status = H5Sclose(mem_dataspace_field);
  h5_check_err(status);

  status = H5Sclose(out_dataspace_field);
  h5_check_err(status);

  status = H5Gclose(gid_sys_data);
  h5_check_err(status);

  status = H5Gclose(gid_trj_data);
  h5_check_err(status);

  status = H5Fclose(fid);
  h5_check_err(status);

  free_1d_int(print_particle_list);
  free_1d_int(print_obstacle_list);
}

//
//Inherited functions
//
void hdf5_writer::write_start(){
  //open file & groups
  {
    sprintf(gid_time_name, "./frame_%d", ts);
    gid_time  = H5Gcreate(gid_trj_data, gid_time_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check_err(gid_time);

    //create field group
    if(!print_field.none){
      gid_field = H5Gcreate(gid_time, gid_field_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      h5_check_err(gid_field);
    }

    //create particle group
    if(print_particle_num != 0){
      gid_part  = H5Gcreate(gid_time, gid_part_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      h5_check_err(gid_part);
    }

    //create obstacle group
    if(print_obstacle_num != 0){
      gid_pobs  = H5Gcreate(gid_time, gid_pobs_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      h5_check_err(gid_pobs);
    }
  }

  //add attributes

  {
    float dmy_float = static_cast<float>(ts)*dt;
    this->write_frame_attributes("time", &dmy_float, 1);
  }
}
void hdf5_writer::write_end(){
  herr_t status;

  if(print_obstacle_num != 0){
    status = H5Gclose(gid_pobs);
    h5_check_err(status);
  }

  if(print_particle_num != 0){
    status = H5Gclose(gid_part);
    h5_check_err(status);
  }

  if(!print_field.none){
    status = H5Gclose(gid_field);
    h5_check_err(status);
  }

  status = H5Fflush(gid_time, H5F_SCOPE_LOCAL);
  h5_check_err(status);
  
  status = H5Gclose(gid_time);
  h5_check_err(status);

  ts += 1;
}

//writer for field data
inline void hdf5_writer::write_field_scalar(const double* phi, const char* name,
					    hid_t _loc=hid_null){
  hid_t loc = (_loc == hid_null ? gid_field : _loc);
  write_data(loc, phi, name, H5T_NATIVE_DOUBLE, mem_dataspace_field, H5T_NATIVE_FLOAT, out_dataspace_field);
}

//writer for particle data
inline void hdf5_writer::write_particle_scalar(const int* data, const char* name,
					       hid_t _loc=hid_null){
  hid_t loc = (_loc == hid_null ? gid_part : _loc);
  hsize_t dmy_dims[1] = {static_cast<hsize_t>(print_particle_num)};
  write_data(loc, data, name, H5T_NATIVE_INT, H5T_NATIVE_INT, 1, dmy_dims);
}
inline void hdf5_writer::write_particle_vectorn(const double* data, const int& dim, const char* name,
						hid_t _loc=hid_null){
  hid_t loc = (_loc == hid_null ? gid_part : _loc);
  hsize_t dmy_dims[2] = {static_cast<hsize_t>(print_particle_num), DIM};
  write_data(loc, data, name, H5T_NATIVE_DOUBLE, H5T_NATIVE_FLOAT, 2, dmy_dims);
}
inline void hdf5_writer::write_particle_matrix3(const double* data, const char* name,
						hid_t _loc=hid_null){
  hid_t loc = (_loc == hid_null ? gid_part : _loc);
  hsize_t dmy_dims[3] = {static_cast<hsize_t>(print_particle_num), DIM, DIM};
  write_data(loc, data, name, H5T_NATIVE_DOUBLE, H5T_NATIVE_FLOAT, 3, dmy_dims);
}

//writer for obstacle particle data
inline void hdf5_writer::write_obstacle_scalar(const int* data, const char* name,
					       hid_t _loc=hid_null){
  hid_t loc = (_loc == hid_null ? gid_pobs : _loc);
  hsize_t dmy_dims[1] = {static_cast<hsize_t>(print_obstacle_num)};
  write_data(loc, data, name, H5T_NATIVE_INT, H5T_NATIVE_INT, 1, dmy_dims);
}
inline void hdf5_writer::write_obstacle_vectorn(const double* data, const int& dim, const char* name,
						hid_t _loc=hid_null){
  hid_t loc = (_loc == hid_null ? gid_pobs : _loc);
  hsize_t dmy_dims[2] = {static_cast<hsize_t>(print_obstacle_num), DIM};
  write_data(loc, data, name, H5T_NATIVE_DOUBLE, H5T_NATIVE_FLOAT, 2, dmy_dims);
}
inline void hdf5_writer::write_obstacle_matrix3(const double* data, const char* name,
						hid_t _loc=hid_null){
  hid_t loc = (_loc == hid_null ? gid_pobs : _loc);
  hsize_t dmy_dims[3] = {static_cast<hsize_t>(print_obstacle_num), DIM, DIM};
  write_data(loc, data, name, H5T_NATIVE_DOUBLE, H5T_NATIVE_FLOAT, 3, dmy_dims);
}

void hdf5_writer::write_field_data(double** u, double* phi, double* pressure, double** tau){
  if(print_field.none) return;

  if(print_field.vel){
    this -> write_field_scalar(u[0], f_vel_name[0]);
    this -> write_field_scalar(u[1], f_vel_name[1]);
    this -> write_field_scalar(u[2], f_vel_name[2]);
  }
  if(print_field.phi){
    this -> write_field_scalar(phi, f_phi_name);
  }
  if(print_field.pressure){
    this -> write_field_scalar(pressure, f_pressure_name);
  }
  if(print_field.tau){
    this -> write_field_scalar(tau[0], f_tau_name[0]);
    this -> write_field_scalar(tau[1], f_tau_name[1]);
    this -> write_field_scalar(tau[2], f_tau_name[2]);
    this -> write_field_scalar(tau[3], f_tau_name[3]);
    this -> write_field_scalar(tau[4], f_tau_name[4]);
  }
}
void hdf5_writer::write_charge_field_data(double** u, double* phi, 
					  double* surface_charge, double* solute_charge, double* potential){
  if(print_field.none) return;

  if(print_field.vel){
    this -> write_field_scalar(u[0], f_vel_name[0]);
    this -> write_field_scalar(u[1], f_vel_name[1]);
    this -> write_field_scalar(u[2], f_vel_name[2]);
  }
  if(print_field.phi){
    this -> write_field_scalar(phi, f_phi_name);
  }
  if(print_field.charge){
    this -> write_field_scalar(surface_charge, f_surface_charge_name);
    this -> write_field_scalar(solute_charge,  f_solute_charge_name);
    this -> write_field_scalar(potential, f_potential_charge_name);
  }
}

//Todo: figure out how to print structures using hdf5 interface (avoid mem copies)!
void hdf5_writer::write_particle_data(Particle *p){
  if(print_particle_num == 0) return;

  const int* plist     = print_particle_list;
  const int nump_print = print_particle_num;
  const int nump_print3= 3*nump_print;
  const int nump_print6= 6*nump_print;

  double* pdata = alloc_1d_double(nump_print3*3); 
  
  int i, jj;
#pragma omp parallel for private(i, jj)
  for(int j = 0; j < nump_print; j++){
    i  = plist[j];
    jj = 3*j;

    pdata[jj]  = p[i].x[0];
    pdata[jj+1]= p[i].x[1];
    pdata[jj+2]= p[i].x[2];
    
    pdata[nump_print3+jj]  = p[i].x_nopbc[0];
    pdata[nump_print3+jj+1]= p[i].x_nopbc[1];
    pdata[nump_print3+jj+2]= p[i].x_nopbc[2];
    
    pdata[nump_print6+jj]  = p[i].v[0];
    pdata[nump_print6+jj+1]= p[i].v[1];
    pdata[nump_print6+jj+2]= p[i].v[2];
  }
  this->write_particle_vectorn(&pdata[0], DIM, p_pos_name);
  this->write_particle_vectorn(&pdata[nump_print3], DIM, p_pos_raw_name);
  this->write_particle_vectorn(&pdata[nump_print6], DIM, p_vel_name);

#pragma omp parallel for private(i, jj)
  for(int j = 0; j < nump_print; j++){
    i  = plist[j];
    jj = 3*j;

    //total hydrodynamic force
    pdata[jj]  = p[i].f_hydro_previous[0] + p[i].f_slip_previous[0];
    pdata[jj+1]= p[i].f_hydro_previous[1] + p[i].f_slip_previous[1];
    pdata[jj+2]= p[i].f_hydro_previous[2] + p[i].f_slip_previous[2];

    //other forces (LJ, external, etc.)
    pdata[nump_print3+jj]  = p[i].fr_previous[0];
    pdata[nump_print3+jj+1]= p[i].fr_previous[1];
    pdata[nump_print3+jj+2]= p[i].fr_previous[2];
  }
  this->write_particle_vectorn(&pdata[0], DIM, p_force_h_name);
  this->write_particle_vectorn(&pdata[nump_print3], DIM, p_force_r_name);

#pragma omp parallel for private(i, jj)
  for(int j = 0; j < nump_print; j++){
    i  = plist[j];
    jj = 3*j;

    pdata[jj]  = p[i].omega[0];
    pdata[jj+1]= p[i].omega[1];
    pdata[jj+2]= p[i].omega[2];
    
    //total hydrodynamic torque
    pdata[nump_print3+jj]  = p[i].torque_hydro_previous[0] + p[i].torque_slip_previous[0];
    pdata[nump_print3+jj+1]= p[i].torque_hydro_previous[1] + p[i].torque_slip_previous[1];
    pdata[nump_print3+jj+2]= p[i].torque_hydro_previous[2] + p[i].torque_slip_previous[2];

    pdata[nump_print6+jj]  = p[i].torque_r_previous[0];
    pdata[nump_print6+jj+1]= p[i].torque_r_previous[1];
    pdata[nump_print6+jj+2]= p[i].torque_r_previous[2];
  }
  this->write_particle_vectorn(&pdata[0], DIM, p_omega_name);
  this->write_particle_vectorn(&pdata[nump_print3], DIM, p_torque_h_name);
  this->write_particle_vectorn(&pdata[nump_print6], DIM, p_torque_r_name);
  
#pragma omp parallel for private(i, jj)
  for(int j = 0; j < nump_print; j++){
    i  = plist[j];
    jj = 9*j;
    
    qtn_normalize(p[i].q);
    rqtn_rm(p[i].QR, p[i].q);
    //x axis
    pdata[jj]  = p[i].QR[0][0]; 
    pdata[jj+1]= p[i].QR[1][0]; 
    pdata[jj+2]= p[i].QR[2][0];
    //y axis
    pdata[jj+3]= p[i].QR[0][1];
    pdata[jj+4]= p[i].QR[1][1]; 
    pdata[jj+5]= p[i].QR[2][1];
    //z axis
    pdata[jj+6]= p[i].QR[0][2]; 
    pdata[jj+7]= p[i].QR[1][2];
    pdata[jj+8]= p[i].QR[2][2];
  }
  this->write_particle_matrix3(&pdata[0], p_QR_name);

  free_1d_double(pdata);
}

void hdf5_writer::write_obstacle_data(Particle *p){
  if(print_obstacle_num == 0) return;

  const int* plist     = print_obstacle_list;
  const int nump_print = print_obstacle_num;
  const int nump_print3= 3*nump_print;

  double* pdata = alloc_1d_double(nump_print3*2);

  int i, jj;
#pragma omp parallel for private(i, jj)
  for(int j = 0; j < nump_print; j++){
    i  = plist[j];
    jj = 3*j;

    //total hydrodynamic force
    pdata[jj]  = p[i].f_hydro_previous[0] + p[i].f_slip_previous[0];
    pdata[jj+1]= p[i].f_hydro_previous[1] + p[i].f_slip_previous[1];
    pdata[jj+2]= p[i].f_hydro_previous[2] + p[i].f_slip_previous[2];

    //other forces (LJ, external, etc.)
    pdata[nump_print3+jj]  = p[i].fr_previous[0];
    pdata[nump_print3+jj+1]= p[i].fr_previous[1];
    pdata[nump_print3+jj+2]= p[i].fr_previous[2];
  }
  this->write_obstacle_vectorn(&pdata[0], DIM, p_force_h_name);
  this->write_obstacle_vectorn(&pdata[nump_print3], DIM, p_force_r_name);

  free_1d_double(pdata);
}

void hdf5_writer::show_parameter() {
  fprintf(stderr, "# *** HDF5 Parameters ***\n");
  fprintf(stderr, "# Output Name          : %s.h5\n", out_path);
  fprintf(stderr, "# Mem Layout           : (%d, %d, %d, %d)\n", NX, NY, NZ, NZ_);
  fprintf(stderr, "# Grid spacing         : %.3g\n", DX);
  fprintf(stderr, "# Time between frames  : %.3g\n", dt);
  if(!print_field.none){
    for(int d = 0; d < DIM; d++){
      fprintf(stderr, "# %s-axis slicing       : start = %4d, stride = %4d, count = %4d\n",
	      f_axis_name[d], crop_field.start[d], crop_field.stride[d], crop_field.count[d]
	      );
    }
    fprintf(stderr, "# Output Origin        : (%.3g, %.3g, %.3g)\n", Origin[0], Origin[1], Origin[2]);
    fprintf(stderr, "# Print velocity field : %s\n", (print_field.vel ? "YES" : "NO"));
    fprintf(stderr, "# Print phi field      : %s\n", (print_field.phi ? "YES" : "NO"));
    fprintf(stderr, "# Print charge field   : %s\n", (print_field.charge  ? "YES" : "NO"));
    fprintf(stderr, "# Print Pressure field : %s\n", (print_field.pressure ? "YES" : "NO"));
    fprintf(stderr, "# Print Stress field   : %s\n", (print_field.tau ? "YES" : "NO"));
  }else{
    fprintf(stderr, "# Field Data is being suppressed\n");
  }
  fprintf(stderr, "# ***********************\n");
}

//
//New functions
//
void hdf5_writer::write_configure_file(){
  
  FILE* conf;
  char confname[128];
  sprintf(confname, "%s.conf", out_path);
  conf = filecheckopen(confname, "w");
  //Parameters
  fprintf(conf, "#Configure file needed to generate xdmf trajectory\n");
  fprintf(conf, "#v1.1\n");
  fprintf(conf, "[Filename]\n %s\n", out_name);
  fprintf(conf, "[Grid Dimensions]\n %lld %lld %lld\n", 
	  out_dims_field[0], out_dims_field[1], out_dims_field[2]);
  fprintf(conf, "[Time Series]\n 0.0 %.6f %d\n", dt, ts);
  fprintf(conf, "[Particle Number]\n %d\n", print_particle_num);
  fprintf(conf, "[Obstacle Number]\n %d\n", print_obstacle_num);
  fprintf(conf, "\n");

  //Particle Data
  if(print_particle_num > 0){
    //Particle Grid
    fprintf(conf, "[Particle Grid]\n %s\n", p_pos_name);
    fprintf(conf, "#[Particle Grid]\n#%s\n", p_pos_raw_name);

    //Scalar Particle Quantities
    fprintf(conf, "[Particle Scalar Const]\n %s\n", p_id_name);
    fprintf(conf, "[Particle Scalar Const]\n %s\n", p_spec_name);

    //Vector Particle Quantities
    fprintf(conf, "[Particle Vector]\n %s\n", p_vel_name);
    fprintf(conf, "[Particle Vector]\n %s\n", p_omega_name);
    fprintf(conf, "[Particle Vector]\n %s\n", p_force_h_name);
    fprintf(conf, "[Particle Vector]\n %s\n", p_force_r_name);
    fprintf(conf, "[Particle Vector]\n %s\n", p_torque_h_name);
    fprintf(conf, "[Particle Vector]\n %s\n", p_torque_r_name);

    //Tensor Particle Quantities
    fprintf(conf, "[Particle Tensor]\n %s\n", p_QR_name);

    fprintf(conf, "\n");
  }

  //Obstacle Data
  if(print_obstacle_num > 0){
    //Particle Grid
    fprintf(conf, "[Obstacle Grid Const]\n %s\n", p_pos_name);

    //Scalar Particle Quantities
    fprintf(conf, "[Obstacle Scalar Const]\n %s\n", p_id_name);
    fprintf(conf, "[Obstacle Scalar Const]\n %s\n", p_spec_name);

    //Vector Particle Quantities
    fprintf(conf, "[Obstacle Vector]\n %s\n", p_force_h_name);
    fprintf(conf, "[Obstacle Vector]\n %s\n", p_force_r_name);

    fprintf(conf, "\n");
  }

  // Field Data
  if(!print_field.none){
    fprintf(conf, "[Field GridX Const]\n %s\n", f_axis_name[0]);
    fprintf(conf, "[Field GridY Const]\n %s\n", f_axis_name[1]);
    fprintf(conf, "[Field GridZ Const]\n %s\n", f_axis_name[2]);

    if(print_field.vel)
      fprintf(conf, "[Field Vector]\n %s %s %s %s\n", 
	      f_vel_name[0], f_vel_name[1], f_vel_name[2], f_vel_name[3]);
    if(print_field.phi)
      fprintf(conf, "[Field Scalar]\n %s\n", f_phi_name);
    if(print_field.pressure)
      fprintf(conf, "[Field Scalar]\n %s\n", f_pressure_name);
    if(print_field.tau){
      fprintf(conf, "[Field Scalar]\n %s\n", f_tau_name[0]);
      fprintf(conf, "[Field Scalar]\n %s\n", f_tau_name[1]);
      fprintf(conf, "[Field Scalar]\n %s\n", f_tau_name[2]);
      fprintf(conf, "[Field Scalar]\n %s\n", f_tau_name[3]);
      fprintf(conf, "[Field Scalar]\n %s\n", f_tau_name[4]);
    }
    if(print_field.charge){
      fprintf(conf, "[Field Scalar]\n %s\n", f_surface_charge_name);
      fprintf(conf, "[Field Scalar]\n %s\n", f_solute_charge_name);
      fprintf(conf, "[Field Scalar]\n %s\n", f_potential_charge_name);
    }

    fprintf(conf, "\n");    
  }
  fclose(conf);
}

void hdf5_writer::write_field_info(){
  if(print_field.none) return;
  double* work_v3[DIM];
  for(int d = 0; d < DIM; d++) 
    work_v3[d] = alloc_1d_double(NX*NY*NZ_);

  double ddx = static_cast<double>(DX);
  for(int i = 0; i < NX; i++){
    for(int j = 0; j < NY; j++){
      for(int k = 0; k < NZ; k++){
	int im = (i*NY*NZ_) + (j*NZ_) + k;
	work_v3[0][im] = static_cast<double>(i)*ddx;
	work_v3[1][im] = static_cast<double>(j)*ddx;
	work_v3[2][im] = static_cast<double>(k)*ddx;
      }
    }
  }
  hid_t gid_dmy = H5Gcreate(gid_sys_data, gid_field_name, 
			      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  h5_check_err(gid_dmy);
  
  this->write_field_scalar(work_v3[0], f_axis_name[0], gid_dmy);
  this->write_field_scalar(work_v3[1], f_axis_name[1], gid_dmy);
  this->write_field_scalar(work_v3[2], f_axis_name[2], gid_dmy);
  
  herr_t status = H5Gclose(gid_dmy);
  h5_check_err(status);
  for(int d = 0; d < DIM; d++)
    free_1d_double(work_v3[d]);
}

void hdf5_writer::write_particle_info(Particle* p){
  if(print_particle_num == 0) return;

  const int* plist     = print_particle_list;
  const int nump_print = print_particle_num;

  int*    pid   = alloc_1d_int(nump_print);
  int*    psp   = alloc_1d_int(nump_print);
  
  int i;
#pragma omp parallel for private(i)
  for(int j = 0; j < nump_print; j++){
    i  = plist[j];
    pid[j] = i;
    psp[j] = p[i].spec;
  }

  hid_t gid_dmy = H5Gcreate(gid_sys_data, gid_part_name,
			    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  h5_check_err(gid_dmy);

  this-> write_particle_scalar(pid, p_id_name, gid_dmy);
  this-> write_particle_scalar(psp, p_spec_name, gid_dmy);

  herr_t status = H5Gclose(gid_dmy);
  h5_check_err(status);
  
  free_1d_int(pid);
  free_1d_int(psp);
}

void hdf5_writer::write_obstacle_info(Particle *p){
  if(print_obstacle_num == 0) return;

  const int* plist     = print_obstacle_list;
  const int nump_print = print_obstacle_num;
  const int nump_print3= 3*nump_print;

  int*    pid   = alloc_1d_int(nump_print);
  int*    psp   = alloc_1d_int(nump_print);
  double* pdata = alloc_1d_double(nump_print3);

  int i, jj;
#pragma omp parallel for private(i, jj)
  for(int j = 0; j < nump_print; j++){
    jj = 3*j;

    i = plist[j];
    pid[j] = i;
    psp[j] = p[i].spec;

    pdata[jj]   = p[i].x[0];
    pdata[jj+1] = p[i].x[1];
    pdata[jj+2] = p[i].x[2];
  }


  hid_t gid_dmy = H5Gcreate(gid_sys_data, gid_pobs_name,
			    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  h5_check_err(gid_dmy);

  this -> write_obstacle_scalar(pid, p_id_name, gid_dmy);
  this -> write_obstacle_scalar(psp, p_spec_name, gid_dmy);
  this -> write_obstacle_vectorn(&pdata[0], DIM, p_pos_name, gid_dmy);

  herr_t status = H5Gclose(gid_dmy);
  h5_check_err(status);

  free_1d_int(pid);
  free_1d_int(psp);
  free_1d_double(pdata);
}
//end hdf5 writer
