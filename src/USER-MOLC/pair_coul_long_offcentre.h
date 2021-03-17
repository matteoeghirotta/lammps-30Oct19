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

/* -------------------------------------------------------------------------
   Contributing author: Matteo Ricci <matteoeghirotta@gmail.com>
--------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(coul/long/offcentre,PairCoulLongOffcentre)

#else

#ifndef LMP_PAIR_COUL_LONG_OFFCENTRE_H
#define LMP_PAIR_COUL_LONG_OFFCENTRE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairCoulLongOffcentre : public Pair {
 public:
  PairCoulLongOffcentre(class LAMMPS *);
  ~PairCoulLongOffcentre();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 protected:
  double cut_coul,cut_coulsq;
  double *cut_respa;
  double g_ewald;
  double **scale;

  class AtomVecEllipsoid *avec;
  int* nsites;
  double ***molFrameSite;  // positions of sites
  double **molFrameCharge;  // positions of sites

  void allocate();
  void init_tables();
  void free_tables();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/cut/coul/long requires atom attribute q

The atom style defined does not have this attribute.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

E: Pair style is incompatible with KSpace style

If a pair style with a long-range Coulombic component is selected,
then a kspace style must also be used.

*/
