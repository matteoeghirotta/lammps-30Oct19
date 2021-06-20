/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
http://lammps.sandia.gov, Sandia National Laboratories
Steve Plimpton, sjplimp@sandia.gov

Copyright (2003) Sandia Corporation.  Under the terms of Contract
DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
certain rights in this software.  This software is distributed under
the GNU General Public License.

See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Matteo Ricci
   Soft-core version: Agilio Padua (Univ Blaise Pascal & CNRS)
   ------------------------------------------------------------------------- */

#include "pair_coul_long_offcentre_soft.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "kspace.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "utils.h"
#include "math.h"
#include "math_extra.h"
#include "atom_vec_ellipsoid.h"

using namespace LAMMPS_NS;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairCoulLongOffcentreSoft::PairCoulLongOffcentreSoft(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  qdist = 0.0;

  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec)
    error->all(FLERR,"Pair Coul Cut Offcenter requires atom style ellipsoid");

  single_enable = 1;
}

/* ---------------------------------------------------------------------- */

PairCoulLongOffcentreSoft::~PairCoulLongOffcentreSoft()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(scale);

    memory->destroy(lambda);
    memory->destroy(lam1);
    memory->destroy(lam2);

    delete[] nsites;
  }
}

/* ---------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::compute_pair(int i, int j, int eflag, int vflag, int evflag)
{
  double ecoul = 0.0;

  bool sameAtom = i == j;

  tagint *tag = atom->tag;
  int *type = atom->type;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int itype = type[i];
  int jnum = numneigh[i];

  double *iquat,*jquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  double fforce[3],ttor[3],r12[3];
  double forcecoul;
  double grij,expm2,prefactor,t,erfc;
  double fraction,table;
  int itable;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  double *special_coul = force->special_coul;
  double fpair = 0.0;

  double factor_coul = special_coul[sbmask(j)];
  factor_coul = sameAtom ? 0.0 : factor_coul;
  j &= NEIGHMASK;
  int jtype = type[j];

  // rotate site1 in lab frame
  double rotMat1[3][3];
  if (nsites[itype] > 0) {
    iquat = bonus[ellipsoid[i]].quat;
    MathExtra::quat_to_mat(iquat, rotMat1);
  }

  // rotate site2 in lab frame
  double rotMat2[3][3];
  if (nsites[jtype] > 0) {
    jquat = bonus[ellipsoid[j]].quat;
    MathExtra::quat_to_mat(jquat, rotMat2);
  }

  for (int s1 = 1; s1 <= nsites[itype]; ++s1) {
    double q1 = molFrameCharge[itype][s1];
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

      double rsq = MathExtra::dot3(r12, r12);

      if ((rsq < cut_coulsq) && ((!sameAtom) || (s1 < s2))) {
        double r2inv = 1.0/rsq;
        double q2 = molFrameCharge[jtype][s2];

        double r = sqrt(rsq);
        grij = g_ewald * r;
        expm2 = exp(-grij*grij);
        t = 1.0 / (1.0 + EWALD_P*grij);
        erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;

        double denc = sqrt(lam2[itype][jtype] + rsq);
        prefactor = qqrd2e * lam1[itype][jtype] * q1*q2 / (denc*denc*denc);

        forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
        if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;

        fpair = forcecoul;

        if (!sameAtom) {
          fpair = forcecoul * r2inv;

          fforce[0] = r12[0]*fpair;
          fforce[1] = r12[1]*fpair;
          fforce[2] = r12[2]*fpair;

          // F_parallel = F_tot . r_normalized
          f[i][0] += fforce[0];
          f[i][1] += fforce[1];
          f[i][2] += fforce[2];

          // Torque = r x F_tot
          // torque on 1 = -pos1 x grad_pos1
          MathExtra::cross3(labFrameSite1, fforce, ttor);
          tor[i][0] += ttor[0];
          tor[i][1] += ttor[1];
          tor[i][2] += ttor[2];

          if (newton_pair || j < nlocal) {
            // F_parallel = F_tot . r_normalized
            f[j][0] -= fforce[0];
            f[j][1] -= fforce[1];
            f[j][2] -= fforce[2];
            MathExtra::cross3(labFrameSite2, fforce, ttor);

            tor[j][0] -= ttor[0];
            tor[j][1] -= ttor[1];
            tor[j][2] -= ttor[2];
          }
        }

        if (eflag) {
          prefactor = qqrd2e * lam1[itype][jtype] * q1*q2 / denc;
          ecoul = prefactor*erfc;
          if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
        }

        if (evflag) {
          // charges inside same bead should not contribute to virial
          if (sameAtom) fpair = 0.0;

          ev_tally(i,j,nlocal,newton_pair,
                   0.0,ecoul,fpair,r12[0],r12[1],r12[2]);
        }
      }
    }
  }
}

void PairCoulLongOffcentreSoft::compute(int eflag, int vflag)
{
  int i,j,ii,jj,itable,itype,jtype;
  double q1, q2;
  double ecoul,fpair;
  double fraction,table;
  double r,r2inv,forcecoul,factor_coul;
  double grij,expm2,prefactor,t,erfc;
  double rsq;
  double *iquat,*jquat;
  double fforce[3],ttor[3],r12[3];

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

  int **firstneigh = list->firstneigh;
  int *numneigh = list->numneigh;

  int inum = list->inum;
  int *ilist = list->ilist;

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    // loop over neighbors
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      compute_pair(i, j, eflag, vflag, evflag);
    }
    // long range self interactions of own charges
    compute_pair(i, i, eflag, vflag, evflag);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(scale,n+1,n+1,"pair:scale");

  memory->create(lambda,n+1,n+1,"pair:lambda");
  memory->create(lam1,n+1,n+1,"pair:lam1");
  memory->create(lam2,n+1,n+1,"pair:lam2");
}

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::settings(int narg, char **arg)
{
  if (narg < 3) error->all(FLERR,"Illegal pair_style command");

  nlambda = force->numeric(FLERR,arg[0]);
  alphac  = force->numeric(FLERR,arg[1]);

  cut_coul = force->numeric(FLERR,arg[2]);

  int nCoulSites = force->inumeric(FLERR, arg[3]);
  unsigned start_sitesspec_argcount = 4;

  // std::cout << "[PairCoulLongOffcentre] coul sites " << nCoulSites<<std::endl;
  // std::cout << "[PairCoulLongOffcentre] atom types "<<atom->ntypes<<std::endl;
  // std::cout << "[PairCoulLongOffcentre] cutoff " <<cut_coul<< std::endl;

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
    atomType[t] = force->inumeric(FLERR, arg[argcount++]);
    ++nsites[atomType[t]];

    argcount += 4;

    // std::cout << "[PairCoulLongOffcentre] coulsite " << t
    // 					<< " atype " << atomType[t]
    // 					<< " ncoulsites " << nsites[atomType[t]]<< std::endl;
  }

  //std::cout << "[PairCoulLongOffcentre] HERE" << std::endl;
  argcount = start_sitesspec_argcount;
  for (int type = 1; type <= atom->ntypes; ++type)
  {
    //int type = atomType[t];

    molFrameSite[type] = new double*[nsites[type]+1];
    molFrameCharge[type] = new double[nsites[type]+1];

    for (int site = 1; site <= nsites[type]; ++site)
    {
      ++argcount;
      // std::cout << "[PairCoulLongOffcentre] site " <<
      // 	site << "/" << nsites[type] << " atype "<< type<< std::endl;

      molFrameSite[type][site] = new double[3];
      molFrameSite[type][site][0] = force->numeric(FLERR, arg[argcount++]);
      molFrameSite[type][site][1] = force->numeric(FLERR, arg[argcount++]);
      molFrameSite[type][site][2] = force->numeric(FLERR, arg[argcount++]);

      molFrameCharge[type][site] = force->numeric(FLERR, arg[argcount++]);

      // std::cout << "[PairCoulLongOffcentre] type " <<
      // 	type << " " << " site " << site
      // 					<< " pos "
      // 					<< " " << molFrameSite[type][site][0]
      // 					<< " " << molFrameSite[type][site][1]
      // 					<< " " << molFrameSite[type][site][2]
      // 					<< " charge "
      // 					<< " " << molFrameCharge[type][site]
      // 					<< " arg " << argcount
      // 					<< std::endl;
    }

    totsites += nsites[type];
  }

  // check
  if (narg > argcount) {
    fprintf(stderr, "number of specified charges exceeds the declared %i\n",
            nCoulSites);
    exit(1);
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::coeff(int narg, char **arg)
{
  if (narg != 3)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double lambda_one = force->numeric(FLERR,arg[2]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      lambda[i][j] = lambda_one;
      scale[i][j] = 1.0;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::init_style()
{
  // if (!atom->q_flag)
  //   error->all(FLERR,"Pair style lj/cut/coul/long requires atom attribute q");

  neighbor->request(this,instance_me);

  cut_coulsq = cut_coul * cut_coul;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   ------------------------------------------------------------------------- */

double PairCoulLongOffcentreSoft::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    if (lambda[i][i] != lambda[j][j])
      error->all(FLERR,"Pair coul/cut/soft different lambda values in mix");
    lambda[i][j] = lambda[i][i];
  }

  lam1[i][j] = pow(lambda[i][j], nlambda);
  lam2[i][j] = alphac * (1.0 - lambda[i][j])*(1.0 - lambda[i][j]);

  scale[j][i] = scale[i][j];
  lambda[j][i] = lambda[i][j];
  lam1[j][i] = lam1[i][j];
  lam2[j][i] = lam2[i][j];

  return cut_coul+2.0*qdist;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j])
        fwrite(&lambda[i][j],sizeof(double),1,fp);
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,NULL,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0)
          utils::sfread(FLERR,&lambda[i][j],sizeof(double),1,fp,NULL,error);
        MPI_Bcast(&lambda[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::write_restart_settings(FILE *fp)
{
  fwrite(&nlambda,sizeof(double),1,fp);
  fwrite(&alphac,sizeof(double),1,fp);

  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentreSoft::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&nlambda,sizeof(double),1,fp,NULL,error);
    utils::sfread(FLERR,&alphac,sizeof(double),1,fp,NULL,error);

    utils::sfread(FLERR,&cut_coul,sizeof(double),1,fp,NULL,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,NULL,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,NULL,error);
  }
  MPI_Bcast(&nlambda,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&alphac,1,MPI_DOUBLE,0,world);

  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairCoulLongOffcentreSoft::single(int i, int j, int itype, int jtype,
                                         double rsq,
                                         double factor_coul, double /*factor_lj*/,
                                         double &fforce)
{
  double *iquat,*jquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double r12[3];

  double r,grij,expm2,t,erfc,prefactor;
  double forcecoul,phicoul = 0.0;
  double denc;

  // rotate site1 in lab frame
  double rotMat1[3][3];
  if (nsites[itype] > 0) {
    iquat = bonus[ellipsoid[i]].quat;
    MathExtra::quat_to_mat(iquat, rotMat1);
  }

  // rotate site2 in lab frame
  double rotMat2[3][3];
  if (nsites[jtype] > 0) {
    jquat = bonus[ellipsoid[j]].quat;
    MathExtra::quat_to_mat(jquat, rotMat2);
  }

  for (int s1 = 1; s1 <= nsites[itype]; ++s1) {
    double q1 = molFrameCharge[itype][s1];
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

    for (int s2 = 1; s2 <= nsites[jtype]; ++s2) {
      double q2 = molFrameCharge[jtype][s2];
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

      double rsq = MathExtra::dot3(r12, r12);

      if (rsq < cut_coulsq) {
        r = sqrt(rsq);
        grij = g_ewald * r;
        expm2 = exp(-grij*grij);
        t = 1.0 / (1.0 + EWALD_P*grij);
        erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;

        denc = sqrt(lam2[itype][jtype] + rsq);
        prefactor = force->qqrd2e * lam1[itype][jtype] * q1*q2 /
          (denc*denc*denc);

        forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
        if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
      } else forcecoul = 0.0;

      fforce += forcecoul;

      if (rsq < cut_coulsq) {
        prefactor = force->qqrd2e * lam1[itype][jtype] * q1*q2 / denc;
        phicoul += prefactor*erfc;
        if (factor_coul < 1.0) phicoul -= (1.0-factor_coul)*prefactor;
      }
    }
  }

  return phicoul;
}

/* ---------------------------------------------------------------------- */

void *PairCoulLongOffcentreSoft::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  dim = 2;
  if (strcmp(str,"scale") == 0) return (void *) scale;
  if (strcmp(str,"lambda") == 0) return (void *) lambda;

  return NULL;
}
