/*!
  \file avs_output.cxx
  \author Y. Nakayama
  \date 2006/06/27
  \version 1.1
  \brief Output routines for field data in AVS/Express format
 */

#include "avs_output.h"

const int Veclen = 5+5; 
const char *Label="ux uy uz phi pressure tau_xy tau_yz tau_zx tau_xx tau_yy"; // avs 出力ラベル用
const int Veclen_charge = 4+3; 
const char *Label_charge="ux uy uz phi surface_charge rho e_potential"; // avs 出力ラベル用

//const AVS_Field Field = irregular;
const AVS_Field Field = uniform;
AVS_parameters Avs_parameters;

void Show_avs_parameter(){
  if(SW_OUTFORMAT == OUT_AVS_BINARY){
    fprintf(stderr, "#for AVS (filetype is binary)\n");
  }else if(SW_OUTFORMAT == OUT_AVS_ASCII){
    fprintf(stderr, "#for AVS (filetype is ascii)\n");
  }else{
    fprintf(stderr, "# Uknown AVS FORMAT\n");
    exit_job(EXIT_FAILURE);
  }
  fprintf(stderr, "#directory:%s\n", Out_dir);
  fprintf(stderr, "# (mesh data)->\t{%s, %s, %s*.dat}\n"
	  ,Avs_parameters.out_fld
	  ,Avs_parameters.out_cod
	  ,Avs_parameters.out_pfx);
  if(Particle_Number > 0){
    fprintf(stderr, "# (particle data)->\t{%s, %s*.cod, %s*.dat}\n"
	    ,Avs_parameters.out_pfld
	    ,Avs_parameters.out_ppfx
	    ,Avs_parameters.out_ppfx
	    );
  }
}

void Init_avs(const AVS_parameters &Avs_parameters){
  {
    char dmy_dir[256];
    sprintf(dmy_dir, "%s/avs", Out_dir);
    dircheckmake(dmy_dir);
  }
  FILE *fout;
  fout=filecheckopen(Avs_parameters.fld_file,"w");
  fprintf(fout,"# AVS field file\n");
  fprintf(fout,"ndim=%d\n",DIM);
  fprintf(fout,"dim1=%d\n", Avs_parameters.nx);
  fprintf(fout,"dim2=%d\n", Avs_parameters.ny);
  fprintf(fout,"dim3=%d\n", Avs_parameters.nz);
  fprintf(fout,"nspace=%d\n", DIM);
  if(SW_EQ == Navier_Stokes 
     || SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards
     ){
    fprintf(fout,"veclen=%d\n", Veclen);
  }else if(SW_EQ==Electrolyte){
    fprintf(fout,"veclen=%d\n", Veclen_charge);
  }
  fprintf(fout,"data=float\n");
  if(Field == irregular ){
    fprintf(fout,"field=irregular\n");
  }else if(Field == uniform ){
    fprintf(fout,"field=uniform\n");
  }else {
    fprintf(stderr, "invalid Field\n"); 
    exit_job(EXIT_FAILURE);
  }
  fprintf(fout,"nstep=%d\n",Avs_parameters.nstep);
  if(SW_EQ == Navier_Stokes 
     || SW_EQ == Shear_Navier_Stokes || SW_EQ == Shear_Navier_Stokes_Lees_Edwards
     ){
    fprintf(fout,"label = %s\n", Label);
  }else if(SW_EQ==Electrolyte){
    fprintf(fout,"label = %s\n", Label_charge);
  }
  
  fclose(fout);
  
  if(Field == irregular){
    fout=filecheckopen(Avs_parameters.cod_file,"wb");

    if(SW_OUTFORMAT == OUT_AVS_BINARY){
      for(int i=Avs_parameters.istart; i<=Avs_parameters.iend; i++){
	for(int j=Avs_parameters.jstart; j<= Avs_parameters.jend ; j++){
	  for(int k=Avs_parameters.kstart; k<= Avs_parameters.kend ; k++){
	    float dmy= (float)(i*DX);
	    fwrite(&dmy,sizeof(float),1,fout);
	  }
	}
      }
      for(int i=Avs_parameters.istart; i<=Avs_parameters.iend; i++){
	for(int j=Avs_parameters.jstart; j<= Avs_parameters.jend ; j++){
	  for(int k=Avs_parameters.kstart; k<= Avs_parameters.kend ; k++){
	    float dmy= (float)(j*DX);
	    fwrite(&dmy,sizeof(float),1,fout);
	  }
	}
      }
      for(int i=Avs_parameters.istart; i<=Avs_parameters.iend; i++){
	for(int j=Avs_parameters.jstart; j<= Avs_parameters.jend ; j++){
	  for(int k=Avs_parameters.kstart; k<= Avs_parameters.kend ; k++){
	    float dmy= (float)(k*DX);
	    fwrite(&dmy,sizeof(float),1,fout);
	  }
	}
      }
    }else if(SW_OUTFORMAT == OUT_AVS_ASCII){
      fprintf(fout,"X Y Z\n");
      for(int i=Avs_parameters.istart; i<=Avs_parameters.iend; i++){
	for(int j=Avs_parameters.jstart; j<= Avs_parameters.jend ; j++){
	  for(int k=Avs_parameters.kstart; k<= Avs_parameters.kend ; k++){
	  fprintf(fout,"%g %g %g\n",
		  (double)i*DX, (double)j*DX, (double)k*DX);
	  }
	}
      }
    }else{
      fprintf(stderr, "# Uknown AVS FORMAT\n");
      exit_job(EXIT_FAILURE);
    }
    fclose(fout);
  }else if(Field == uniform){
    fout=filecheckopen(Avs_parameters.cod_file,"wb");
    {
      fprintf(fout,"%g\n%g\n%g\n%g\n%g\n%g\n"
	      ,(double)(Avs_parameters.istart)*DX
	      ,(double)(Avs_parameters.iend)*DX
	      ,(double)(Avs_parameters.jstart)*DX
	      ,(double)(Avs_parameters.jend)*DX
	      ,(double)(Avs_parameters.kstart)*DX
	      ,(double)(Avs_parameters.kend)*DX
	      );
    }
    fclose(fout);
  }else {
    fprintf(stderr, "invalid Field\n"); 
    exit_job(EXIT_FAILURE);
  }
}


void Set_avs_parameters(AVS_parameters &Avs_parameters){
  {
    Avs_parameters.nx=NX;
    Avs_parameters.ny=NY;
    Avs_parameters.nz=NZ;
  }
  
  {
    Avs_parameters.istart = 0;
    Avs_parameters.iend = NX-1;
    Avs_parameters.jstart = 0;
    Avs_parameters.jend = NY-1;
    Avs_parameters.kstart = 0;
    Avs_parameters.kend = NZ-1;
  }

  {
    sprintf(Avs_parameters.out_fld, "%s.fld", Out_name);
    {
      sprintf(Avs_parameters.out_cod, "%s.cod", Out_name);
      sprintf(Avs_parameters.cod_file, "%s/%s",
	      Out_dir, Avs_parameters.out_cod);
    }
    {
      sprintf(Avs_parameters.out_pfx, "avs/%s_", Out_name);
      sprintf(Avs_parameters.fld_file, "%s/%s",
	      Out_dir, Avs_parameters.out_fld);
    }

    sprintf(Avs_parameters.out_pfld, "%sp.fld", Out_name);
    //sprintf(Avs_parameters.out_pcod, "%sp.cod", Out_name);
    sprintf(Avs_parameters.out_ppfx, "avs/%sp_", Out_name);

    sprintf(Avs_parameters.pfld_file, "%s/%s",
	    Out_dir, Avs_parameters.out_pfld);
    //sprintf(Avs_parameters.pcod_file, "%s/%s",
    //    Out_dir, Avs_parameters.out_pcod);
  }
  {
    Avs_parameters.nstep=(Num_snap+1);
  }

}

inline void Binary_write(FILE *fout
			 ,AVS_parameters &Avs_parameters
			 ,const double *a
			 ){
  int im;
  for(int k=Avs_parameters.kstart; k<= Avs_parameters.kend; k++){
    for(int j=Avs_parameters.jstart; j<= Avs_parameters.jend; j++){
      for(int i=Avs_parameters.istart; i<=Avs_parameters.iend; i++){
	im = (i * NY * NZ_) + (j * NZ_) + k;
	float dmy= (float)a[im];
	fwrite(&dmy,sizeof(float),1,fout);
      }
    }
  }
}
inline void Add_field_description(AVS_parameters &Avs_parameters
				  ,const CTime &time
				  ,const int &veclen
				  ){
  FILE *fout;
  fout=filecheckopen(Avs_parameters.fld_file,"a");
  fprintf(fout,"time value = \"step%dtime%g\"\n"
	  ,time.ts, time.time);
  if(SW_OUTFORMAT == OUT_AVS_BINARY){
    static const int data_size=sizeof(float)*
      Avs_parameters.nx * Avs_parameters.ny * Avs_parameters.nz;
    if(Field == irregular){
      for(int n=0; n < DIM; n++){
	fprintf(fout,
		"coord %d file = %s filetype = binary skip = %d\n",
		n+1,
		Avs_parameters.out_cod, n*data_size);
      }
    }else if(Field == uniform){
      {
	for(int n=0; n < DIM; n++){
	  fprintf(fout,
		  "coord %d file = %s filetype = ascii skip = %d\n",
		  n+1, Avs_parameters.out_cod, n*2);
	}
      }
    }
    for(int n=0; n < veclen; n++){
      fprintf(fout,
	      "variable %d file = %s%d.dat filetype = binary skip = %d\n",
	      n+1,
	      Avs_parameters.out_pfx, time.ts, n * data_size);
    }
  }else if(SW_OUTFORMAT == OUT_AVS_ASCII){
    if(Field == irregular){
      for(int n=0; n < DIM; n++){
	fprintf(fout,
		"coord %d file = %s filetype = ascii skip = 1 offset = %d stride =%d\n",
		n+1, Avs_parameters.out_cod, n, DIM);
      }
    }else if(Field == uniform){
      for(int n=0; n < DIM; n++){
	fprintf(fout,
		"coord %d file = %s filetype = ascii skip = %d\n",
		n+1, Avs_parameters.out_cod, n*2);
      }
    }else {
      fprintf(stderr, "invalid Field\n"); 
      exit_job(EXIT_FAILURE);
    }
    for(int n=0; n < veclen; n++){
      fprintf(fout,
	      "variable %d file = %s%d.dat filetype = ascii skip = 2 offset = %d stride = %d\n",
	      n+1, Avs_parameters.out_pfx, time.ts, n, veclen);
    }
  }else{
    fprintf(stderr, "# Uknown AVS FORMAT\n");
    exit_job(EXIT_FAILURE);
  }
  fprintf(fout,"EOT\n");
  fclose(fout);
}

void Output_avs(AVS_parameters &Avs_parameters
		,double **u
		,double *phi
		,double *Pressure
		,double **strain
		,const CTime &time){

  Add_field_description(Avs_parameters,time, Veclen);

  FILE *fout;
  char line[512];
  sprintf(line,"timesteps=%d time=%f\n%s",time.ts,time.time, Label);
  sprintf(Avs_parameters.data_file,"%s/%s%d.dat",
	  Out_dir, Avs_parameters.out_pfx, time.ts);
  fout=filecheckopen(Avs_parameters.data_file,"wb");
  
  if(SW_OUTFORMAT == OUT_AVS_BINARY){
    Binary_write(fout, Avs_parameters, u[0]);
    Binary_write(fout, Avs_parameters, u[1]);
    Binary_write(fout, Avs_parameters, u[2]);
    Binary_write(fout, Avs_parameters, phi);
    Binary_write(fout, Avs_parameters, Pressure);
    {

      Binary_write(fout, Avs_parameters, strain[1]); // 12
      Binary_write(fout, Avs_parameters, strain[4]); // 23
      Binary_write(fout, Avs_parameters, strain[2]); // 13

      Binary_write(fout, Avs_parameters, strain[0]); // 11
      Binary_write(fout, Avs_parameters, strain[3]); // 22

    }
  }else if(SW_OUTFORMAT == OUT_AVS_ASCII){
    fprintf(fout,"%s\n", line);
    for(int k=Avs_parameters.kstart; k<= Avs_parameters.kend; k++){
      for(int j=Avs_parameters.jstart; j<= Avs_parameters.jend; j++){
	for(int i=Avs_parameters.istart; i<=Avs_parameters.iend; i++){
		int im=(i*NY*NZ_)+(j*NZ_)+k;
	  fprintf(fout,"%.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g\n"
		  ,u[0][im]
		  ,u[1][im]
		  ,u[2][im]
		  ,phi[im]
		  ,Pressure[im]
		  ,strain[1][im]
		  ,strain[4][im]
		  ,strain[2][im]
		  ,strain[0][im]
		  ,strain[3][im]
		  );
	}
      }
    }
  }else{
    fprintf(stderr, "# Uknown AVS FORMAT\n");
    exit_job(EXIT_FAILURE);
  }

  fclose(fout);

}


void Output_avs_charge(AVS_parameters &Avs_parameters
		       ,double** u
		       ,double* phi
		       ,double* colloid_charge
		       ,double* solute_charge_total
		       ,double* potential
		       ,const CTime &time
		       ){
  Add_field_description(Avs_parameters,time, Veclen_charge);

  FILE *fout;
  char line[512];
  sprintf(line,"timesteps=%d time=%f\n%s",time.ts,time.time, Label_charge);
  sprintf(Avs_parameters.data_file,"%s/%s%d.dat",
	  Out_dir, Avs_parameters.out_pfx, time.ts);
  fout=filecheckopen(Avs_parameters.data_file,"wb");
  
  if(SW_OUTFORMAT == OUT_AVS_BINARY){
    Binary_write(fout, Avs_parameters, u[0]);
    Binary_write(fout, Avs_parameters, u[1]);
    Binary_write(fout, Avs_parameters, u[2]);
    Binary_write(fout, Avs_parameters, phi);
    Binary_write(fout, Avs_parameters, colloid_charge);
    Binary_write(fout, Avs_parameters, solute_charge_total);
    Binary_write(fout, Avs_parameters, potential);
  }else if(SW_OUTFORMAT == OUT_AVS_ASCII){ // OUT_AVS_ASCII
    fprintf(fout,"%s\n", line);
    for(int k=Avs_parameters.kstart; k<= Avs_parameters.kend; k++){
      for(int j=Avs_parameters.jstart; j<= Avs_parameters.jend; j++){
	for(int i=Avs_parameters.istart; i<=Avs_parameters.iend; i++){
	  int im = (i*NY*NZ_) + (j*NZ_) + k;
	  fprintf(fout,"%.3g %.3g %.3g %.3g %.3g %.3g %.3g\n"
		  ,u[0][im]
		  ,u[1][im]
		  ,u[2][im]
		  ,phi[im]
		  ,colloid_charge[im]
		  ,solute_charge_total[im]
		  ,potential[im]
		  );
	}
      }
    }
  }else{
    fprintf(stderr, "# Uknown AVS FORMAT\n");
    exit_job(EXIT_FAILURE);
  }
  fclose(fout);
}

