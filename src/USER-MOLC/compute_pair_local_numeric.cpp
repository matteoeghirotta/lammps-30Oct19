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

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "compute_pair_local_numeric.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "group.h"
#include "memory.h"
#include "error.h"
#include "atom_vec_ellipsoid.h"
#include "math_extra.h"

using namespace LAMMPS_NS;

#define DELTA 10000

enum{DIST,ENG,FX,FY,FZ,PN,TX,TY,TZ};

/* ---------------------------------------------------------------------- */

ComputePairLocalNumeric::ComputePairLocalNumeric(LAMMPS *lmp,
                                                 int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 5)
    error->all(FLERR,"Illegal compute pair/local/numeric command");

  compute_force = false;
  compute_torque = false;

  local_flag = 1;
  nvalues = narg - 4;
  if (nvalues == 1) size_local_cols = 0;
  else size_local_cols = nvalues;

  pstyle = new int[nvalues];
  pindex = new int[nvalues];

  delta_move = atof(arg[3]);

  nvalues = 0;
  for (int iarg = 4; iarg < narg; iarg++) {
    if (strcmp(arg[iarg],"dist") == 0) pstyle[nvalues++] = DIST;
    else if (strcmp(arg[iarg],"eng") == 0) pstyle[nvalues++] = ENG;
    else if (strcmp(arg[iarg],"fx") == 0) { pstyle[nvalues++] = FX;
      compute_force = true; }
    else if (strcmp(arg[iarg],"fy") == 0) { pstyle[nvalues++] = FY;
      compute_force = true; }
    else if (strcmp(arg[iarg],"fz") == 0) { pstyle[nvalues++] = FZ;
      compute_force = true; }
    else if (strcmp(arg[iarg],"tx") == 0) { pstyle[nvalues++] = TX;
      compute_torque = true; }
    else if (strcmp(arg[iarg],"ty") == 0) { pstyle[nvalues++] = TY;
      compute_torque = true; }
    else if (strcmp(arg[iarg],"tz") == 0) { pstyle[nvalues++] = TZ;
      compute_torque = true; }
    else if (arg[iarg][0] == 'p') {
      int n = atoi(&arg[iarg][1]);
      fprintf(stderr, "%s: ", arg[iarg]);
      if (n <= 0)
        error->all(FLERR,
                   "Invalid keyword in compute pair/local/numeric command");
      pstyle[nvalues] = PN;
      pindex[nvalues++] = n-1;
    } else {
      //   fprintf(stderr, "%s: ", arg[iarg]);
      error->all(FLERR,
                 "Invalid keyword in compute pair/local/numeric command");
    }
  }

  // set singleflag if need to call pair->single()
  // set singleflag if need to call pair->single()

  singleflag = 0;
  for (int i = 0; i < nvalues; i++)
    if (pstyle[i] != DIST) singleflag = 1;

  nmax = 0;
  vector = NULL;
  array = NULL;

  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec) error->all(FLERR,
                        "compute pair/local/numeric requires atom style "
                        "ellipsoid");
}

/* ---------------------------------------------------------------------- */

ComputePairLocalNumeric::~ComputePairLocalNumeric()
{
  memory->destroy(vector);
  memory->destroy(array);
  delete [] pstyle;
  delete [] pindex;
}

/* ---------------------------------------------------------------------- */

void ComputePairLocalNumeric::init()
{
  if (singleflag && force->pair == NULL)
    error->all(FLERR,
               "No pair style is defined for compute pair/local/numeric");
  if (singleflag && force->pair->single_enable == 0)
    error->all(FLERR,
               "Pair style does not support compute pair/local/numeric");

  for (int i = 0; i < nvalues; i++)
    if (pstyle[i] == PN && pindex[i] >= force->pair->single_extra)
      error->all(FLERR,"Pair style does not have extra field"
                 " requested by compute pair/local/numeric");

  // need an occasional half neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void ComputePairLocalNumeric::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputePairLocalNumeric::compute_local()
{
  invoked_local = update->ntimestep;

  // count local entries and compute pair info

  ncount = compute_pairs(0);
  if (ncount > nmax) reallocate(ncount);
  size_local_rows = ncount;
  compute_pairs(1);
}

/* ----------------------------------------------------------------------
   count pairs and compute pair info on this proc
   only count pair once if newton_pair is off
   both atom I,J must be in group
   if flag is set, compute requested info about pair
   -------------------------------------------------------------------------*/

int ComputePairLocalNumeric::compute_pairs(int flag)
{
  int i,j,m,n,ii,jj,inum,jnum,itype,jtype;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,eng,fpair,factor_coul,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *ptr;
  double runit[3];
  double angle = delta_move;

  double **x = atom->x;
  tagint *tag = atom->tag;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  // invoke half neighbor list (will copy or build if necessary)

  if (flag == 0) neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // skip if I or J are not in group
  // for newton = 0 and J = ghost atom,
  //   need to insure I,J pair is only output by one proc
  //   use same itag,jtag logic as in Neighbor::neigh_half_nsq()
  // for flag = 0, just count pair interactions within force cutoff
  // for flag = 1, calculate requested output fields

  Pair *pair = force->pair;
  double **cutsq = force->pair->cutsq;

  m = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    double *quat = bonus[ellipsoid[i]].quat;
    double qtmp[4] = {quat[0], quat[1], quat[2], quat[3]};

    itag = tag[i];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      if (!(mask[j] & groupbit)) continue;

      // itag = jtag is possible for long cutoffs that include images of self

      if (newton_pair == 0 && j >= nlocal) {
        jtag = tag[j];
        if (itag > jtag) {
          if ((itag+jtag) % 2 == 0) continue;
        } else if (itag < jtag) {
          if ((itag+jtag) % 2 == 1) continue;
        } else {
          if (x[j][2] < ztmp) continue;
          if (x[j][2] == ztmp) {
            if (x[j][1] < ytmp) continue;
            if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
          }
        }
      }

      jtype = type[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq >= cutsq[itype][jtype]) continue;

      double ene_comp_trasl[3][2] = { { 0.0, 0.0 },
                                      { 0.0, 0.0 },
                                      { 0.0, 0.0 } };

      double ene_comp_rot[3][2] = { { 0.0, 0.0 },
                                    { 0.0, 0.0 },
                                    { 0.0, 0.0 } };

      if (flag) {
        for (int comp = 0; comp < 3; comp++) {
          if (compute_force) {
            for (int delta_sign = -1; delta_sign < 2; delta_sign += 2) {
              double delta = delta_sign*delta_move;

              x[i][0] = xtmp + delta*(comp == 0 ? 1.0 : 0);
              x[i][1] = ytmp + delta*(comp == 1 ? 1.0 : 0);
              x[i][2] = ztmp + delta*(comp == 2 ? 1.0 : 0);

              if (singleflag)
                eng = pair->single(i,j,itype,jtype,rsq,
                                   factor_coul,factor_lj,fpair);
              else eng = fpair = 0.0;

              if (delta_sign == -1)
                ene_comp_trasl[comp][0] = eng;
              else if (delta_sign == 1)
                ene_comp_trasl[comp][1] = eng;
            }

            x[i][0] = xtmp;
            x[i][1] = ytmp;
            x[i][2] = ztmp;
          }

          if (compute_torque) {
            for (int delta_sign = -1; delta_sign < 2; delta_sign += 2) {
              double delta_angle = delta_sign*delta_move;

              runit[0] = comp == 0 ? 1.0 : 0.0;
              runit[1] = comp == 1 ? 1.0 : 0.0;
              runit[2] = comp == 2 ? 1.0 : 0.0;

              double rotated[4] = {0.0, 0.0, 0.0, 0.0};
              double q[4] = {0.0, 0.0, 0.0, 0.0};

              MathExtra::axisangle_to_quat(runit, delta_angle, q);
              MathExtra::quatquat(q, qtmp, quat);
              MathExtra::qnormalize(quat);

              if (singleflag)
                eng = pair->single(i,j,itype,jtype,rsq,
                                   factor_coul,factor_lj,fpair);
              else eng = fpair = 0.0;
              
              if (delta_sign == -1)
                ene_comp_rot[comp][0] = eng;
              else if (delta_sign == 1)
                ene_comp_rot[comp][1] = eng;
            }

            quat[0] = qtmp[0];
            quat[1] = qtmp[1];
            quat[2] = qtmp[2];
            quat[3] = qtmp[3];
          }
        }

        if (singleflag)
          eng = pair->single(i,j,itype,jtype,rsq,
                             factor_coul,factor_lj,fpair);
        else eng = fpair = 0.0;
      }
      
      if (flag) {
        if (nvalues == 1) ptr = &vector[m];
        else ptr = array[m];

        for (n = 0; n < nvalues; n++) {
          switch (pstyle[n]) {
          case DIST:
            ptr[n] = sqrt(rsq);
            break;
          case ENG:
            ptr[n] = eng;
            break;
          case FX:
            ptr[n] = -(ene_comp_trasl[0][1]-ene_comp_trasl[0][0])
              /(2.0*delta_move);
            break;
          case FY:
            ptr[n] = -(ene_comp_trasl[1][1]-ene_comp_trasl[1][0])
              /(2.0*delta_move);
            break;
          case FZ:
            ptr[n] = -(ene_comp_trasl[2][1]-ene_comp_trasl[2][0])
              /(2.0*delta_move);
            break;
          case PN:
            ptr[n] = pair->svector[pindex[n]];
            break;
          case TX:
            ptr[n] = -(ene_comp_rot[0][1]-ene_comp_rot[0][0])
              /(2.0*delta_move);
            break;
          case TY:
            ptr[n] = -(ene_comp_rot[1][1]-ene_comp_rot[1][0])
              /(2.0*delta_move);
            break;
          case TZ:
            ptr[n] = -(ene_comp_rot[2][1]-ene_comp_rot[2][0])
              /(2.0*delta_move);
            break;
          }
        }
      }

      m++;
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputePairLocalNumeric::reallocate(int n)
{
  // grow vector or array and indices array

  while (nmax < n) nmax += DELTA;

  if (nvalues == 1) {
    memory->destroy(vector);
    memory->create(vector,nmax,"pair/local:vector");
    vector_local = vector;
  } else {
    memory->destroy(array);
    memory->create(array,nmax,nvalues,"pair/local:array");
    array_local = array;
  }
}

/* ----------------------------------------------------------------------
   memory usage of local data
   ------------------------------------------------------------------------- */

double ComputePairLocalNumeric::memory_usage()
{
  double bytes = nmax*nvalues * sizeof(double);
  return bytes;
}
