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
   Contributing author: Matteo Ricci
   -------------------------------------------------------------------------*/

#include "math.h"
#include "math_extra.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_coul_long_offcentre.h"
#include "atom_vec_ellipsoid.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "kspace.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include <iostream>

using namespace LAMMPS_NS;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairCoulLongOffcentre::PairCoulLongOffcentre(LAMMPS *lmp) : Pair(lmp)
{
  ftable = NULL;
  ewaldflag = pppmflag = 1;

  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec)
    error->all(FLERR,"Pair Coul Cut Offcenter requires atom style ellipsoid");

  single_enable = 1;
}

/* ---------------------------------------------------------------------- */

PairCoulLongOffcentre::~PairCoulLongOffcentre()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(scale);

    delete[] nsites;
  }
  if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */

void PairCoulLongOffcentre::compute_pair(int i, int j, int eflag, int vflag)
{
  double ecoul = 0.0;

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  bool sameAtom = i == j;

  // tagint *tag = atom->tag;
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

      //domain->minimum_image(r12[0], r12[1], r12[2]);
      double rsq = MathExtra::dot3(r12, r12);

      if ((rsq < cut_coulsq) && ((!sameAtom) || (sameAtom && (s1 < s2)))) {
        double r2inv = 1.0/rsq;
        double q2 = molFrameCharge[jtype][s2];

        if (!ncoultablebits || rsq <= tabinnersq) {
          double r = sqrt(rsq);
          double grij = g_ewald * r;
          double expm2 = exp(-grij*grij);
          double t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
          prefactor = qqrd2e * scale[itype][jtype] * q1*q2/r;
          forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
          if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else {
          union_int_float_t rsq_lookup;
          rsq_lookup.f = rsq;
          itable = rsq_lookup.i & ncoulmask;
          itable >>= ncoulshiftbits;
          fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
          table = ftable[itable] + fraction*dftable[itable];
          forcecoul = scale[itype][jtype] * q1*q2 * table;
          if (factor_coul < 1.0) {
            table = ctable[itable] + fraction*dctable[itable];
            prefactor = scale[itype][jtype] * q1*q2 * table;
            forcecoul -= (1.0-factor_coul)*prefactor;
          }
        }

        double fpair = forcecoul * r2inv;

        if (!sameAtom) {
          fforce[0] = r12[0]*fpair; ///r12n;
          fforce[1] = r12[1]*fpair; ///r12n;
          fforce[2] = r12[2]*fpair; ///r12n;

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
            f[j][0] -= fforce[0]; ///r12n;
            f[j][1] -= fforce[1]; ///r12n;
            f[j][2] -= fforce[2]; ///r12n;
            MathExtra::cross3(labFrameSite2, fforce, ttor);

            tor[j][0] -= ttor[0];
            tor[j][1] -= ttor[1];
            tor[j][2] -= ttor[2];
          }
        }

        if (eflag) {
          if (!ncoultablebits || rsq <= tabinnersq)
            ecoul = prefactor*erfc;
          else {
            table = etable[itable] + fraction*detable[itable];
            ecoul = scale[itype][jtype] * q1*q2 * table;
          }
          if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
        }

        if (evflag) {
          // printf("EVFLAG %f %i %i\n", fpair, (tag[i]-1)*nsites[itype]+s1, (tag[j]-1)*nsites[jtype]+s2);

          ev_tally(i,j,nlocal,newton_pair,
                   0.0,ecoul,fpair,r12[0],r12[1],r12[2]);
          // double *virial = force->pair->virial;
          // printf("virial %f %f %f %f %f\n", virial[0], virial[1], virial[2], virial[3], virial[4], virial[5]);
        }
      }
    }
  }
}

void PairCoulLongOffcentre::compute(int eflag, int vflag)
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
      compute_pair(i, j, eflag, vflag);
    }
    // long range self interactions of own charges
    compute_pair(i, i, eflag, vflag);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(scale,n+1,n+1,"pair:scale");
}

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal pair_style command");

  cut_coul = force->numeric(FLERR, arg[0]);

  int nCoulSites = force->inumeric(FLERR, arg[1]);
  unsigned start_sitesspec_argcount = 2;

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
   -------------------------------------------------------------------------*/

void PairCoulLongOffcentre::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      scale[i][j] = 1.0;
      setflag[i][j] = 1;
      count++;
    }
  }

  // std::cout << "coeff() " << ilo << " " << jlo
  //           << " " << ihi << " " << jhi << " " << count
  //           << " args " << arg[0] << " " << arg[1] << std::endl;

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::init_style()
{
  // if (!atom->q_flag)
  //   error->all(FLERR,"Pair style lj/cut/coul/long requires atom attribute q");

  neighbor->request(this);

  cut_coulsq = cut_coul * cut_coul;

  // set & error check interior rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0) {
    cut_respa = ((Respa *) update->integrate)->cutoff;
    if (cut_coul < cut_respa[3])
      error->all(FLERR,"Pair cutoff < Respa interior cutoff");
  } else cut_respa = NULL;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all(FLERR,"Pair style is incompatible with KSpace style");
  g_ewald = force->kspace->g_ewald;

  // setup force tables

  if (ncoultablebits) init_tables();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   ------------------------------------------------------------------------- */

double PairCoulLongOffcentre::init_one(int i, int j)
{
  scale[j][i] = scale[i][j];

  return cut_coul;
}

/* ----------------------------------------------------------------------
   setup force tables used in compute routines
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::init_tables()
{
  int masklo,maskhi;
  double r,grij,expm2,derfc,rsw;
  double qqrd2e = force->qqrd2e;

  tabinnersq = tabinner*tabinner;
  init_bitmap(tabinner,cut_coul,ncoultablebits,
              masklo,maskhi,ncoulmask,ncoulshiftbits);

  int ntable = 1;
  for (int i = 0; i < ncoultablebits; i++) ntable *= 2;

  // linear lookup tables of length N = 2^ncoultablebits
  // stored value = value at lower edge of bin
  // d values = delta from lower edge to upper edge of bin

  if (ftable) free_tables();

  memory->create(rtable,ntable,"pair:rtable");
  memory->create(ftable,ntable,"pair:ftable");
  memory->create(ctable,ntable,"pair:ctable");
  memory->create(etable,ntable,"pair:etable");
  memory->create(drtable,ntable,"pair:drtable");
  memory->create(dftable,ntable,"pair:dftable");
  memory->create(dctable,ntable,"pair:dctable");
  memory->create(detable,ntable,"pair:detable");

  if (cut_respa == NULL) {
    vtable = ptable = dvtable = dptable = NULL;
  } else {
    memory->create(vtable,ntable,"pair:vtable");
    memory->create(ptable,ntable,"pair:ptable");
    memory->create(dvtable,ntable,"pair:dvtable");
    memory->create(dptable,ntable,"pair:dptable");
  }

  union_int_float_t rsq_lookup;
  union_int_float_t minrsq_lookup;
  int itablemin;
  minrsq_lookup.i = 0 << ncoulshiftbits;
  minrsq_lookup.i |= maskhi;

  for (int i = 0; i < ntable; i++) {
    rsq_lookup.i = i << ncoulshiftbits;
    rsq_lookup.i |= masklo;
    if (rsq_lookup.f < tabinnersq) {
      rsq_lookup.i = i << ncoulshiftbits;
      rsq_lookup.i |= maskhi;
    }
    r = sqrtf(rsq_lookup.f);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    derfc = erfc(grij);
    if (cut_respa == NULL) {
      rtable[i] = rsq_lookup.f;
      ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      ctable[i] = qqrd2e/r;
      etable[i] = qqrd2e/r * derfc;
    } else {
      rtable[i] = rsq_lookup.f;
      ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2 - 1.0);
      ctable[i] = 0.0;
      etable[i] = qqrd2e/r * derfc;
      ptable[i] = qqrd2e/r;
      vtable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      if (rsq_lookup.f > cut_respa[2]*cut_respa[2]) {
        if (rsq_lookup.f < cut_respa[3]*cut_respa[3]) {
          rsw = (r - cut_respa[2])/(cut_respa[3] - cut_respa[2]);
          ftable[i] += qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
          ctable[i] = qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
        } else {
          ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
          ctable[i] = qqrd2e/r;
        }
      }
    }
    minrsq_lookup.f = MIN(minrsq_lookup.f,rsq_lookup.f);
  }

  tabinnersq = minrsq_lookup.f;

  int ntablem1 = ntable - 1;

  for (int i = 0; i < ntablem1; i++) {
    drtable[i] = 1.0/(rtable[i+1] - rtable[i]);
    dftable[i] = ftable[i+1] - ftable[i];
    dctable[i] = ctable[i+1] - ctable[i];
    detable[i] = etable[i+1] - etable[i];
  }

  if (cut_respa) {
    for (int i = 0; i < ntablem1; i++) {
      dvtable[i] = vtable[i+1] - vtable[i];
      dptable[i] = ptable[i+1] - ptable[i];
    }
  }

  // get the delta values for the last table entries
  // tables are connected periodically between 0 and ntablem1

  drtable[ntablem1] = 1.0/(rtable[0] - rtable[ntablem1]);
  dftable[ntablem1] = ftable[0] - ftable[ntablem1];
  dctable[ntablem1] = ctable[0] - ctable[ntablem1];
  detable[ntablem1] = etable[0] - etable[ntablem1];
  if (cut_respa) {
    dvtable[ntablem1] = vtable[0] - vtable[ntablem1];
    dptable[ntablem1] = ptable[0] - ptable[ntablem1];
  }

  // get the correct delta values at itablemax
  // smallest r is in bin itablemin
  // largest r is in bin itablemax, which is itablemin-1,
  //   or ntablem1 if itablemin=0
  // deltas at itablemax only needed if corresponding rsq < cut*cut
  // if so, compute deltas between rsq and cut*cut

  double f_tmp,c_tmp,e_tmp,p_tmp,v_tmp;
  itablemin = minrsq_lookup.i & ncoulmask;
  itablemin >>= ncoulshiftbits;
  int itablemax = itablemin - 1;
  if (itablemin == 0) itablemax = ntablem1;
  rsq_lookup.i = itablemax << ncoulshiftbits;
  rsq_lookup.i |= maskhi;

  if (rsq_lookup.f < cut_coulsq) {
    rsq_lookup.f = cut_coulsq;
    r = sqrtf(rsq_lookup.f);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    derfc = erfc(grij);

    if (cut_respa == NULL) {
      f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      c_tmp = qqrd2e/r;
      e_tmp = qqrd2e/r * derfc;
    } else {
      f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2 - 1.0);
      c_tmp = 0.0;
      e_tmp = qqrd2e/r * derfc;
      p_tmp = qqrd2e/r;
      v_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      if (rsq_lookup.f > cut_respa[2]*cut_respa[2]) {
        if (rsq_lookup.f < cut_respa[3]*cut_respa[3]) {
          rsw = (r - cut_respa[2])/(cut_respa[3] - cut_respa[2]);
          f_tmp += qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
          c_tmp = qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
        } else {
          f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
          c_tmp = qqrd2e/r;
        }
      }
    }

    drtable[itablemax] = 1.0/(rsq_lookup.f - rtable[itablemax]);
    dftable[itablemax] = f_tmp - ftable[itablemax];
    dctable[itablemax] = c_tmp - ctable[itablemax];
    detable[itablemax] = e_tmp - etable[itablemax];
    if (cut_respa) {
      dvtable[itablemax] = v_tmp - vtable[itablemax];
      dptable[itablemax] = p_tmp - ptable[itablemax];
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::write_restart(FILE *fp)
{
  write_restart_settings(fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::write_restart_settings(FILE *fp)
{
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_coul,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   free memory for tables used in pair computations
   ------------------------------------------------------------------------- */

void PairCoulLongOffcentre::free_tables()
{
  memory->destroy(rtable);
  memory->destroy(drtable);
  memory->destroy(ftable);
  memory->destroy(dftable);
  memory->destroy(ctable);
  memory->destroy(dctable);
  memory->destroy(etable);
  memory->destroy(detable);
  memory->destroy(vtable);
  memory->destroy(dvtable);
  memory->destroy(ptable);
  memory->destroy(dptable);
}

/* ---------------------------------------------------------------------- */

double PairCoulLongOffcentre::single(int i, int j, int itype, int jtype,
                                     double rsq,
                                     double factor_coul, double factor_lj,
                                     double &fpair)
{
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;

  int ii,jj,inum,jnum;
  double q1, q2;
  double rinv;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double fforce[3],ttor[3],r12[3];
  double *iquat,*jquat;

  double r2inv,r,grij,expm2,t,erfc,prefactor;
  double fraction,table,forcecoul;
  int itable;
  double qqrd2e = force->qqrd2e;

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
      //this was originally here but why not quat_to_mat as for site 1?
      //MathExtra::quat_to_mat_trans(jquat, rotMat2);
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

      if (rsq < cut_coulsq) {
        r2inv = 1.0/rsq;
        q2 = molFrameCharge[jtype][s2];

        if (!ncoultablebits || rsq <= tabinnersq) {
          r = sqrt(rsq);
          grij = g_ewald * r;
          expm2 = exp(-grij*grij);
          t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
          prefactor = qqrd2e * scale[itype][jtype] * q1*q2/r;
          forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
          if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else {
          union_int_float_t rsq_lookup;
          rsq_lookup.f = rsq;
          itable = rsq_lookup.i & ncoulmask;
          itable >>= ncoulshiftbits;
          fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
          table = ftable[itable] + fraction*dftable[itable];
          forcecoul = scale[itype][jtype] * q1*q2 * table;
          if (factor_coul < 1.0) {
            table = ctable[itable] + fraction*dctable[itable];
            prefactor = scale[itype][jtype] * q1*q2 * table;
            forcecoul -= (1.0-factor_coul)*prefactor;
          }
        }

        fpair += forcecoul * r2inv;

        if (!ncoultablebits || rsq <= tabinnersq)
          phicoul += prefactor*erfc;
        else {
          table = etable[itable] + fraction*detable[itable];
          phicoul += scale[itype][jtype] * q1*q2 * table;
        }
        if (factor_coul < 1.0) phicoul -= (1.0-factor_coul)*prefactor;
      }
    }
  }

  return phicoul;
}

/* ---------------------------------------------------------------------- */

void *PairCoulLongOffcentre::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cut_coul;
  }
  if (strcmp(str,"scale") == 0) {
    dim = 2;
    return (void *) scale;
  }
  return NULL;
}
