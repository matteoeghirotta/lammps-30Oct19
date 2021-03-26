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

#include "compute_erotate_asphere_atom.h"
#include <mpi.h>
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "atom_vec_line.h"
#include "atom_vec_tri.h"
#include "update.h"
#include "force.h"
#include "error.h"
#include <cstring>
#include "modify.h"
#include "comm.h"
#include "memory.h"



using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeERotateAsphereAtom::
ComputeERotateAsphereAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  erot(NULL)
{
  if (narg != 3) 
    error->all(FLERR,"Illegal compute erotate/asphere//atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeERotateAsphereAtom::~ComputeERotateAsphereAtom()
{
  memory->destroy(erot);
}

/* ---------------------------------------------------------------------- */

void ComputeERotateAsphereAtom::init()
{
  // error check

  avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  avec_line = (AtomVecLine *) atom->style_match("line");
  avec_tri = (AtomVecTri *) atom->style_match("tri");
  if (!avec_ellipsoid && !avec_line && !avec_tri)
    error->all(FLERR,"Compute erotate/asphere requires "
               "atom style ellipsoid or line or tri");

  // check that all particles are finite-size
  // no point particles allowed, spherical is OK

  int *ellipsoid = atom->ellipsoid;
  int *line = atom->line;
  int *tri = atom->tri;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int flag;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      flag = 0;
      if (ellipsoid && ellipsoid[i] >= 0) flag = 1;
      if (line && line[i] >= 0) flag = 1;
      if (tri && tri[i] >= 0) flag = 1;
      if (!flag)
        error->one(FLERR,"Compute erotate/asphere requires extended particles");
    }

  pfactor = 0.5 * force->mvv2e;
}

/* ---------------------------------------------------------------------- */

void ComputeERotateAsphereAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow local energy array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(erot);
    nmax = atom->nmax;
    memory->create(erot,nmax,"erot_asphere/atom:energy");
    vector_atom = erot;
  }

  AtomVecEllipsoid::Bonus *ebonus;
  if (avec_ellipsoid) ebonus = avec_ellipsoid->bonus;
  AtomVecLine::Bonus *lbonus;
  if (avec_line) lbonus = avec_line->bonus;
  AtomVecTri::Bonus *tbonus;
  if (avec_tri) tbonus = avec_tri->bonus;
  int *ellipsoid = atom->ellipsoid;
  int *line = atom->line;
  int *tri = atom->tri;
  double **omega = atom->omega;
  double **angmom = atom->angmom;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // sum rotational energy for each particle
  // no point particles since divide by inertia

  double length;
  double *shape,*quat;
  double wbody[3],inertia[3];
  double rot[3][3];
  vector_atom = erot;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (ellipsoid && ellipsoid[i] >= 0) {
        shape = ebonus[ellipsoid[i]].shape;
        quat = ebonus[ellipsoid[i]].quat;

        // principal moments of inertia

        inertia[0] = rmass[i] * (shape[1]*shape[1]+shape[2]*shape[2]) / 5.0;
        inertia[1] = rmass[i] * (shape[0]*shape[0]+shape[2]*shape[2]) / 5.0;
        inertia[2] = rmass[i] * (shape[0]*shape[0]+shape[1]*shape[1]) / 5.0;

        // wbody = angular velocity in body frame

        MathExtra::quat_to_mat(quat,rot);
        MathExtra::transpose_matvec(rot,angmom[i],wbody);
        wbody[0] /= inertia[0];
        wbody[1] /= inertia[1];
        wbody[2] /= inertia[2];

        erot[i] = inertia[0]*wbody[0]*wbody[0] +
          inertia[1]*wbody[1]*wbody[1] + inertia[2]*wbody[2]*wbody[2];
	erot[i] *= pfactor;

      } else if (line && line[i] >= 0) {
        length = lbonus[line[i]].length;

        erot[i] = (omega[i][0]*omega[i][0] + omega[i][1]*omega[i][1] +
                    omega[i][2]*omega[i][2]) * length*length*rmass[i] / 12.0;
	erot[i] *= pfactor;

      } else if (tri && tri[i] >= 0) {

        // principal moments of inertia

        inertia[0] = tbonus[tri[i]].inertia[0];
        inertia[1] = tbonus[tri[i]].inertia[1];
        inertia[2] = tbonus[tri[i]].inertia[2];

        // wbody = angular velocity in body frame

        MathExtra::quat_to_mat(tbonus[tri[i]].quat,rot);
        MathExtra::transpose_matvec(rot,angmom[i],wbody);
        wbody[0] /= inertia[0];
        wbody[1] /= inertia[1];
        wbody[2] /= inertia[2];

        erot[i] = inertia[0]*wbody[0]*wbody[0] +
          inertia[1]*wbody[1]*wbody[1] + inertia[2]*wbody[2]*wbody[2];
	erot[i] *= pfactor;
      }
    }
}
