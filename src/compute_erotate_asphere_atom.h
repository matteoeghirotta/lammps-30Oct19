/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(erotate/asphere/atom,ComputeERotateAsphereAtom)

#else

#ifndef LMP_COMPUTE_EROTATE_ASPHERE_ATOM_H
#define LMP_COMPUTE_EROTATE_ASPHERE_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeERotateAsphereAtom : public Compute {
 public:
  ComputeERotateAsphereAtom(class LAMMPS *, int, char **);
  ~ComputeERotateAsphereAtom();
  void init();
  void compute_peratom();

 private:
  int nmax;
  double pfactor;
  class AtomVecEllipsoid *avec_ellipsoid;
  class AtomVecLine *avec_line;
  class AtomVecTri *avec_tri;
  double *erot;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute erotate/asphere/atom requires atom style ellipsoid or line or tri

Self-explanatory.

E: Compute erotate/asphere/atom requires extended particles

This compute cannot be used with point particles.

*/
