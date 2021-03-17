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

#ifdef FIX_CLASS

FixStyle(rotate/q,FixRotateQ)

#else

#ifndef LMP_FIX_ROTATE_Q_H
#define LMP_FIX_ROTATE_Q_H

#include "stdio.h"
#include "fix.h"

namespace LAMMPS_NS {

class FixRotateQ : public Fix {
 public:
  FixRotateQ(class LAMMPS *, int, char **);
  ~FixRotateQ();
  int setmask();
  void initial_integrate(int);

 private:
  int mstyle;
  double period,omega_rotate;
  double axis[3],runit[3];
  class AtomVecEllipsoid *avec;
  int maxatom;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix move cannot set linear z motion for 2d problem

Self-explanatory.

E: Fix move cannot set wiggle z motion for 2d problem

Self-explanatory.

E: Fix move cannot rotate aroung non z-axis for 2d problem

Self-explanatory.

E: Fix move cannot define z or vz variable for 2d problem

Self-explanatory.

W: Fix move does not update angular momentum

Atoms store this quantity, but fix move does not (yet) update it.

W: Fix move does not update quaternions

Atoms store this quantity, but fix move does not (yet) update it.

E: Zero length rotation vector with fix move

Self-explanatory.

E: Variable name for fix move does not exist

Self-explanatory.

E: Variable for fix move is invalid style

Only equal-style variables can be used.

E: Cannot add atoms to fix move variable

Atoms can not be added afterwards to this fix option.

E: Resetting timestep size is not allowed with fix move

This is because fix move is moving atoms based on elapsed time.

*/
