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

#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "math_extra.h"
#include "fix_rotate_q.h"
#include "atom.h"
#include "group.h"
#include "update.h"
#include "modify.h"
#include "force.h"
#include "domain.h"
#include "lattice.h"
#include "comm.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "atom_vec_ellipsoid.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixRotateQ::FixRotateQ(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix rotate/q command");

  // parse args

  int iarg;

  axis[0] = atof(arg[3]);
  axis[1] = atof(arg[4]);
  axis[2] = atof(arg[5]);
  period =  atof(arg[6]);

  printf("axis %f %f %f period %f\n", axis[0], axis[1], axis[2], period);

  // set omega_rotate from period
  double PI = 4.0 * atan(1.0);
  omega_rotate = 2.0*PI / period;

  if (!atom->angmom_flag && comm->me == 0)
    error->warning(FLERR,"Fix move update angular momentum");
  if (!atom->ellipsoid_flag && comm->me == 0)
    error->warning(FLERR,"Fix move update quaternions");

  double len = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  if (len == 0.0)
    error->all(FLERR,"Zero length rotation vector with fix move");

  runit[0] = axis[0]/len;
  runit[1] = axis[1]/len;
  runit[2] = axis[2]/len;

  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec) error->all(FLERR,"Pair gayberne requires atom style ellipsoid");
}

/* ---------------------------------------------------------------------- */

FixRotateQ::~FixRotateQ()
{}

/* ---------------------------------------------------------------------- */

int FixRotateQ::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ----------------------------------------------------------------------
   set x,v of particles
------------------------------------------------------------------------- */

void FixRotateQ::initial_integrate(int vflag)
{
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double angle = omega_rotate;

  double rotated[4] = {0.0, 0.0, 0.0, 0.0};
  double q[4] = {0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double *quat = bonus[ellipsoid[i]].quat;
      double qoriginal[4] = {quat[0], quat[1], quat[2], quat[3]};

      MathExtra::axisangle_to_quat(runit, angle, q);
      MathExtra::quatquat(q, qoriginal, quat);
      MathExtra::qnormalize(quat);
    }
  }
}
