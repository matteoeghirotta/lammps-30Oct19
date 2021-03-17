/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   -------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------
   Contributing author: Matteo Ricci (...)
   -------------------------------------------------------------------------*/

#include "update.h"
#include "math.h"
#include "math_extra.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_coul_cut_offcentre.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairCoulCutOffcentre::PairCoulCutOffcentre(LAMMPS *lmp) : Pair(lmp) 
{
  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec)
    error->all(FLERR,"Pair Coul Cut Offcenter requires atom style ellipsoid");

  single_enable = 1;
}

/* ---------------------------------------------------------------------- */

PairCoulCutOffcentre::~PairCoulCutOffcentre()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);

    //molFrameCharge

    delete[] nsites;
  }
}

/* ---------------------------------------------------------------------- */

void PairCoulCutOffcentre::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double q1, q2,ecoul,fpair;
  double rsq,r2inv,rinv,forcecoul,factor_coul;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double fforce[3],ttor[3],r12[3];
  double *iquat,*jquat;

  ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

// debug!!!!!!!!!!!!!!!!!!!!
  // for (ii = 0; ii < nlocal; ii++) {
  //   i = ilist[ii];
  //   itype = type[i];

  //   // rotate site1 in lab frame
  //   double rotMat1[3][3];
  //   if (nsites[itype] > 0) {
  //     iquat = bonus[ellipsoid[i]].quat;
  //     MathExtra::quat_to_mat(iquat, rotMat1);
  //   }
    
  //   //std::cout << "[PairCoulCutOffcentre] SITES " << nsites[itype] << " " << nsites[jtype] << " " << itype << " " << jtype << std::endl;  

  //   for (int s1 = 1; s1 <= nsites[itype]; ++s1) {
  //       q1 = molFrameCharge[itype][s1];
	  
  //       double labFrameSite1[3] = {0.0, 0.0, 0.0};
  //       //double labframeNormalized1[3] = {0.0, 0.0, 0.0};
  //       if (molFrameSite[itype][s1][0] != 0.0 ||
  //           molFrameSite[itype][s1][1] != 0.0 ||
  //           molFrameSite[itype][s1][2] != 0.0)
  //         {
  //           double ms1[3] = {
  //             molFrameSite[itype][s1][0],
  //             molFrameSite[itype][s1][1],
  //             molFrameSite[itype][s1][2]
  //           };

  //           MathExtra::matvec(rotMat1, ms1, labFrameSite1);
  //           // double labFrameSite1norm =
  //           //   sqrt(labFrameSite1[0]*labFrameSite1[0]+
  //           // 	   labFrameSite1[1]*labFrameSite1[1]+
  //           // 	   labFrameSite1[2]*labFrameSite1[2]);

  //           // labframeNormalized1[0] = labFrameSite1[0]/labFrameSite1norm;
  //           // labframeNormalized1[1] = labFrameSite1[1]/labFrameSite1norm;
  //           // labframeNormalized1[2] = labFrameSite1[2]/labFrameSite1norm;
  //         }	

  //       double rsite1[3] = {
  //         labFrameSite1[0]+x[i][0],
  //         labFrameSite1[1]+x[i][1],
  //         labFrameSite1[2]+x[i][2]
  //       };        

  //   std::cout << "[PairCoulCutOffcentre]DEBUG "
  // 	      << update->ntimestep << " " << itype << " " << s1 << " " 
  // 	      << rsite1[0] << " " << rsite1[1] << " " << rsite1[2] << " "
  // 	      << q1
  // 	      << std::endl;
  //   }
  // }
// debug!!!!!!!!!!!!!!!!!!!!
  
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // rotate site1 in lab frame
    double rotMat1[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat1);
    }
    
    //std::cout << "[PairCoulCutOffcentre] SITES " << nsites[itype] << " " << nsites[jtype] << " " << itype << " " << jtype << std::endl;  

    for (int s1 = 1; s1 <= nsites[itype]; ++s1) {
      q1 = molFrameCharge[itype][s1];
	  
      double labFrameSite1[3] = {0.0, 0.0, 0.0};
      //double labframeNormalized1[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s1][0] != 0.0 ||
	  molFrameSite[itype][s1][1] != 0.0 ||
	  molFrameSite[itype][s1][2] != 0.0)
	{
	  double ms1[3] = {
	    molFrameSite[itype][s1][0],
	    molFrameSite[itype][s1][1],
	    molFrameSite[itype][s1][2]
	  };

	  MathExtra::matvec(rotMat1, ms1, labFrameSite1);
	  // double labFrameSite1norm =
	  //   sqrt(labFrameSite1[0]*labFrameSite1[0]+
	  // 	   labFrameSite1[1]*labFrameSite1[1]+
	  // 	   labFrameSite1[2]*labFrameSite1[2]);

	  // labframeNormalized1[0] = labFrameSite1[0]/labFrameSite1norm;
	  // labframeNormalized1[1] = labFrameSite1[1]/labFrameSite1norm;
	  // labframeNormalized1[2] = labFrameSite1[2]/labFrameSite1norm;
	}	

      double rsite1[3] = {
	labFrameSite1[0]+x[i][0],
	labFrameSite1[1]+x[i][1],
	labFrameSite1[2]+x[i][2]
      };

      for (jj = 0; jj < jnum; jj++) {
	j = jlist[jj];
	factor_coul = special_coul[sbmask(j)];
	j &= NEIGHMASK;
	jtype = type[j];

	// rotate site2 in lab frame
	double rotMat2[3][3];
	if (nsites[jtype] > 0) {
	  jquat = bonus[ellipsoid[j]].quat;
	  MathExtra::quat_to_mat(jquat, rotMat2);
	  //this was originally here but why not quat_to_mat as for site 1? 
	  //MathExtra::quat_to_mat_trans(jquat, rotMat2);
	}

	for (int s2 = 1; s2 <= nsites[jtype]; ++s2) {
	  //std::cout << "[PairCoulCutOffcentre] s1 " << s2 << "/" << nsites[jtype] << std::endl;  
	      	  
	  double labFrameSite2[3] = {0.0, 0.0, 0.0};
	  //double labframeNormalized2[3] = {0.0, 0.0, 0.0};
	  if (molFrameSite[jtype][s2][0] != 0.0 ||
	      molFrameSite[jtype][s2][1] != 0.0 ||
	      molFrameSite[jtype][s2][2] != 0.0)
	    {
	      double ms2[3] = {
		molFrameSite[jtype][s2][0],
		molFrameSite[jtype][s2][1],
		molFrameSite[jtype][s2][2]
	      };

	      MathExtra::matvec(rotMat2, ms2, labFrameSite2);
	      // double labFrameSite2norm =
	      //   sqrt(labFrameSite2[0]*labFrameSite2[0]+
	      // 	 labFrameSite2[1]*labFrameSite2[1]+
	      // 	 labFrameSite2[2]*labFrameSite2[2]);

	      // labframeNormalized2[0] = labFrameSite2[0]/labFrameSite2norm;
	      // labframeNormalized2[1] = labFrameSite2[1]/labFrameSite2norm;
	      // labframeNormalized2[2] = labFrameSite2[2]/labFrameSite2norm;
	    }
		
	  double rsite2[3] = {
	    labFrameSite2[0]+x[j][0],
	    labFrameSite2[1]+x[j][1],
	    labFrameSite2[2]+x[j][2]
	  };
		
	  // r12 = site center to site center vector
	  r12[0] = rsite1[0]-rsite2[0];
	  r12[1] = rsite1[1]-rsite2[1];
	  r12[2] = rsite1[2]-rsite2[2];

	  //domain->minimum_image(r12[0], r12[1], r12[2]);

	  rsq = MathExtra::dot3(r12, r12);
	      
	  // std::cout << "[PairCoulCutOffcentre] r " << rsq << " " <<  cutsq[itype][jtype] <<std::endl;
	  // std::cout << "[PairCoulCutOffcentre] site "
	  // 		<< rsite1[0] << " " << rsite2[0] << " "
	  // 		<< rsite1[1] << " " << rsite2[1] << " "
	  // 		<< rsite1[2] << " " << rsite2[2] << " "
	  // 		<< std::endl;
	  // std::cout << "[PairCoulCutOffcentre] frame "
	  // 		<< labFrameSite1[0] << " " << x[i][0] << " "
	  // 		<< labFrameSite1[1] << " " << x[i][1] << " "
	  // 		<< labFrameSite1[2] << " " << x[i][2] << " "
	  // 		<< std::endl;

	  if (rsq < cutsq[itype][jtype]) {
	    r2inv = 1.0/rsq;
	    rinv = sqrt(r2inv);

	    q2 = molFrameCharge[jtype][s2];

	    forcecoul = qqrd2e * q1*q2*rinv;
	    if (eflag) 
	      ecoul = factor_coul * forcecoul;

	    // std::cout << "[PairCoulCutOffcentre] "
	    // 	      << itype << " " << jtype << " *"
	    // 	      << factor_coul << " " << sbmask(j) << " " << j
	    // 	      << s1 << " " << s2 << " "
	    // 	      << q1 << " " << q2 << " "
	    // 	      << rsite1[0] << " " << rsite1[1] << " " << rsite1[2] << " "
	    // 	      << rsite2[0] << " " << rsite2[1] << " " << rsite2[2] << " " 
	    // 	      << sqrt(rsq) << " " << factor_coul * forcecoul
	    // 	      << std::endl;

	    fpair = factor_coul*forcecoul*r2inv;

	    if (evflag) ev_tally(i,j,nlocal,newton_pair,
				 0.0,ecoul,fpair,r12[0],r12[1],r12[2]);

	    //if (eflag) 
	    //		  std::cout << "[PairCoulCutOffcentre] e " << ecoul << " " << rinv << " cut " << cutsq[itype][jtype] << std::endl;  

	    // if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
	    // 			 evdwl,ecoul,fforce[0],fforce[1],fforce[2],
	    // 			 -r12[0],-r12[1],-r12[2]);
		
	    fforce[0] = r12[0]*fpair; ///r12n;
	    fforce[1] = r12[1]*fpair; ///r12n;
	    fforce[2] = r12[2]*fpair; ///r12n;

	    // F_parallel = F_tot . r_normalized		
	    f[i][0] += fforce[0];
	    f[i][1] += fforce[1];
	    f[i][2] += fforce[2];
	    // f[i][0] += labframeNormalized1[0]*fforce[0];
	    // f[i][1] += labframeNormalized1[1]*fforce[1];
	    // f[i][2] += labframeNormalized1[2]*fforce[2];

	    // Torque = r x F_tot
	    // torque on 1 = -pos1 x grad_pos1
	    MathExtra::cross3(labFrameSite1, fforce, ttor);
		
	    // std::cout << "[PairCoulCutOffcentre] tor 1 "
	    //  	  << ttor[0] << " " 
	    //  	  << ttor[1] << " "  
	    //  	  << ttor[2] << " "  
	    //  	  << std::endl;

	    // std::cout << "[PairCoulCutOffcentre] for 1 "
	    //  	  << f[i][0] << " " 
	    //  	  << f[i][1] << " "  
	    //  	  << f[i][2] << " "  
	    //  	  << labframeNormalized1[0]*fforce[0] << " " 
	    //  	  << labframeNormalized1[1]*fforce[1] << " "  
	    //  	  << labframeNormalized1[2]*fforce[2] << " "  
	    // 	  << fpair << " "
	    //  	  << labframeNormalized1[0] << " " 
	    //  	  << labframeNormalized1[1] << " "  
	    //  	  << labframeNormalized1[2] << " "  
	    //  	  << r12[0] << " " 
	    //  	  << r12[1] << " "  
	    //  	  << r12[2] << " "  
	    //  	  << fforce[0] << " " 
	    //  	  << fforce[1] << " "  
	    //  	  << fforce[2] << " "  
	    //  	  << std::endl;

	    tor[i][0] += ttor[0];
	    tor[i][1] += ttor[1];
	    tor[i][2] += ttor[2];

	    // std::cout << "[PairCoulCutOffcentre] gtor 1 "
	    //  	  << tor[i][0] << " " 
	    //  	  << tor[i][1] << " "  
	    //  	  << tor[i][2] << " "  
	    //  	  << std::endl;

	    if (newton_pair || j < nlocal) {
	      // F_parallel = F_tot . r_normalized
	      f[j][0] -= fforce[0]; ///r12n;
	      f[j][1] -= fforce[1]; ///r12n;
	      f[j][2] -= fforce[2]; ///r12n;

	      // f[j][0] -= labframeNormalized2[0]*fforce[0]; ///r12n;
	      // f[j][1] -= labframeNormalized2[1]*fforce[1]; ///r12n;
	      // f[j][2] -= labframeNormalized2[2]*fforce[2]; ///r12n;

	      //std::cout << "[PairCoulCutOffcentre] distance " << sqrt(rsq) << " " << cutsq[itype][jtype] << " " << ecoul<< std::endl;  
	      MathExtra::cross3(labFrameSite2, fforce, ttor);
		    
	      // std::cout << "[PairCoulCutOffcentre] tor 2 "
	      // 	      << ttor[0] << " " 
	      // 	      << ttor[1] << " "  
	      // 	      << ttor[2] << " "  
	      // 	      << std::endl;

	      // std::cout << "[PairCoulCutOffcentre] for 2 "
	      // 	      << labframeNormalized2[0]*fforce[0] << " " 
	      // 	      << labframeNormalized2[1]*fforce[1] << " "  
	      // 	      << labframeNormalized2[2]*fforce[2] << " "  
	      // 	      << std::endl;

	      tor[j][0] -= ttor[0];
	      tor[j][1] -= ttor[1];
	      tor[j][2] -= ttor[2];

	      // std::cout << "[PairCoulCutOffcentre] gtor 2 "
	      //  	  << tor[j][0] << " " 
	      //  	  << tor[j][1] << " "  
	      //  	  << tor[j][2] << " "  
	      //  	  << std::endl;
	    }
	  }
	}
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
   ------------------------------------------------------------------------- */

void PairCoulCutOffcentre::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  setflag = memory->create(setflag, n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  cutsq = memory->create(cutsq, n+1,n+1,"pair:cutsq");

  cut = memory->create(cut, n+1,n+1,"pair:cut");
}

/* ----------------------------------------------------------------------
   global settings
   ----------------------------------------------------------------------- */

void PairCoulCutOffcentre::settings(int narg, char **arg)
{
  if (narg < 7) error->all(FLERR,"Illegal pair_style command");

  cut_coul = force->numeric(FLERR, arg[0]);

  int nCoulSites = force->inumeric(FLERR, arg[1]);
  unsigned start_sitesspec_argcount = 2;

  // std::cout << "[PairCoulCutOffcentre] coul sites " <<
  //   nCoulSites << " " << std::endl;
  // std::cout << "[PairCoulCutOffcentre] atom types " <<
  //   atom->ntypes << " " << std::endl;
  // std::cout << "[PairCoulCutOffcentre] cutoff " <<
  //   cut_coul << " " << std::endl;

  int atomType[nCoulSites+1];

  nsites = new int[atom->ntypes+1];
  for (int t = 1; t <= atom->ntypes; ++t)
    nsites[t] = 0;

  molFrameSite = new double**[atom->ntypes+1];
  molFrameCharge = new double*[atom->ntypes+1];

  int totsites = 0;
  unsigned argcount = start_sitesspec_argcount;
  for (int t = 1; t <= nCoulSites; ++t)
    {
      // std::cout << "[PairCoulCutOffcentre] HERE " 
      // 		<< t << " / " << nCoulSites
      // 		<< arg[argcount]
      // 		<< std::endl;

      atomType[t] = force->inumeric(FLERR, arg[argcount++]);
      ++nsites[atomType[t]];

      argcount += 4;
      // std::cout << "[PairCoulCutOffcentre] coulsite " << t
      //           << " atype " << atomType[t]
      //           << " ncoulsites " << nsites[atomType[t]]<< std::endl;
    }

  argcount = start_sitesspec_argcount;
  for (int type = 1; type <= atom->ntypes; ++type)
    {
      //int type = atomType[t];

      molFrameSite[type] = new double*[nsites[type]+1];
      molFrameCharge[type] = new double[nsites[type]+1];

      for (int site = 1; site <= nsites[type]; ++site)
        {
          ++argcount;
          // std::cout << "[PairCoulCutOffcentre] site " <<
          //   site << "/" << nsites[type] << " type "<< type << std::endl;

          molFrameSite[type][site] = new double[3];
          molFrameSite[type][site][0] = force->numeric(FLERR,arg[argcount++]);
          molFrameSite[type][site][1] = force->numeric(FLERR,arg[argcount++]);
          molFrameSite[type][site][2] = force->numeric(FLERR,arg[argcount++]);
	  
          molFrameCharge[type][site] = force->numeric(FLERR,arg[argcount++]);

          // std::cout << "[PairCoulCutOffcentre] type " <<
          //   type << " " << " site " << site
          //           << " pos "
          //           << " " << molFrameSite[type][site][0] 
          //           << " " << molFrameSite[type][site][1] 
          //           << " " << molFrameSite[type][site][2] 
          //           << " charge "
          //           << " " << molFrameCharge[type][site]
          //           << " arg " << argcount
          //           << std::endl;	  
        }

      totsites += nsites[type];
    }

  allocate();

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   ------------------------------------------------------------------------- */

void PairCoulCutOffcentre::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double cut_one = cut_global;
  cut_one = force->numeric(FLERR, arg[2]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {

    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
   ------------------------------------------------------------------------- */

void PairCoulCutOffcentre::init_style()
{
  // if (!atom->q_flag)
  //   error->all(FLERR,"Pair style coul/cut/offcengtre requires atom attribute q");

  neighbor->request(this);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   ------------------------------------------------------------------------- */

double PairCoulCutOffcentre::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairCoulCutOffcentre::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) fwrite(&cut[i][j],sizeof(double),1,fp);
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairCoulCutOffcentre::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) fread(&cut[i][j],sizeof(double),1,fp);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairCoulCutOffcentre::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairCoulCutOffcentre::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairCoulCutOffcentre::single(int i, int j, int itype, int jtype,
                                    double rsq,
                                    double factor_coul, double factor_lj,
                                    double &fpair)
{
  double r2inv,rinv,forcecoul;
  double q1, q2;
  double *iquat,*jquat;
  int *type = atom->type;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double r12[3];
  double qqrd2e = force->qqrd2e;

  itype = type[i];

  // accumulated...
  double phicoul = 0.0;
  fpair = 0.0;
  
  // rotate site1 in lab frame
  double rotMat1[3][3];
  if (nsites[itype] > 0) {
    iquat = bonus[ellipsoid[i]].quat;
    MathExtra::quat_to_mat(iquat, rotMat1);
  }

  for (int s1 = 1; s1 <= nsites[itype]; ++s1) {
    q1 = molFrameCharge[itype][s1];
	  
    double labFrameSite1[3] = {0.0, 0.0, 0.0};
    if (molFrameSite[itype][s1][0] != 0.0 ||
	molFrameSite[itype][s1][1] != 0.0 ||
	molFrameSite[itype][s1][2] != 0.0)
      {
	double ms1[3] = {
	  molFrameSite[itype][s1][0],
	  molFrameSite[itype][s1][1],
	  molFrameSite[itype][s1][2]
	};

	MathExtra::matvec(rotMat1, ms1, labFrameSite1);
      }	

    double rsite1[3] = {
      labFrameSite1[0]+x[i][0],
      labFrameSite1[1]+x[i][1],
      labFrameSite1[2]+x[i][2]
    };

    // rotate site2 in lab frame
    double rotMat2[3][3];
    if (nsites[jtype] > 0) {
      jquat = bonus[ellipsoid[j]].quat;
      MathExtra::quat_to_mat(jquat, rotMat2);
    }

    for (int s2 = 1; s2 <= nsites[jtype]; ++s2) {
      double labFrameSite2[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[jtype][s2][0] != 0.0 ||
	  molFrameSite[jtype][s2][1] != 0.0 ||
	  molFrameSite[jtype][s2][2] != 0.0)
	{
	  double ms2[3] = {
	    molFrameSite[jtype][s2][0],
	    molFrameSite[jtype][s2][1],
	    molFrameSite[jtype][s2][2]
	  };

	  MathExtra::matvec(rotMat2, ms2, labFrameSite2);
	}
		
      double rsite2[3] = {
	labFrameSite2[0]+x[j][0],
	labFrameSite2[1]+x[j][1],
	labFrameSite2[2]+x[j][2]
      };
		
      // r12 = site center to site center vector
      r12[0] = rsite1[0]-rsite2[0];
      r12[1] = rsite1[1]-rsite2[1];
      r12[2] = rsite1[2]-rsite2[2];

      //domain->minimum_image(r12[0], r12[1], r12[2]);
      rsq = MathExtra::dot3(r12, r12);

      if (rsq < cutsq[itype][jtype]) {
	r2inv = 1.0/rsq;
	rinv = sqrt(r2inv);

	q2 = molFrameCharge[jtype][s2];

	forcecoul = qqrd2e * q1*q2*rinv;

	phicoul += factor_coul * forcecoul;
	fpair += factor_coul*forcecoul*r2inv;
      }
    }
  }

  return factor_coul*phicoul;
}

/* ---------------------------------------------------------------------- */

// unsure about that
// should give hort range cutoff but cut_global is long?
void *PairCoulCutOffcentre::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    //printf("PairCoulCutOffcentre called %g\n", cut_coul);
    return (void *) &cut_coul;
  }
  return NULL;
}
