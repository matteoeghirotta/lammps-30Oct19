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

PairStyle(coul/cut/offcentre,PairCoulCutOffcentre)

#else

#ifndef LMP_PAIR_COUL_CUT_OFFCENTRE_H
#define LMP_PAIR_COUL_CUT_OFFCENTRE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairCoulCutOffcentre : public Pair {
 public:
  PairCoulCutOffcentre(class LAMMPS *);
  virtual ~PairCoulCutOffcentre();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 protected:
  double cut_coul, cut_global;
  double **cut;

  class AtomVecEllipsoid *avec;

  int* nsites;
  double ***molFrameSite;  // positions of sites
  double **molFrameCharge;  // positions of sites

  void allocate();
};

}

#endif
#endif
