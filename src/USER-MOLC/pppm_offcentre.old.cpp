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
------------------------------------------------------------------------- */

#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pppm_offcentre.h"
#include "atom.h"
#include "comm.h"
#include "gridcomm.h"
#include "neighbor.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "domain.h"
#include "fft3d_wrap.h"
#include "remap_wrap.h"
#include "memory.h"
#include "error.h"
#include "atom_vec_ellipsoid.h"

#include "math_const.h"
#include "math_special.h"
#include "math_extra.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define MAXORDER 7
#define OFFSET 16384
#define LARGE 10000.0
#define SMALL 0.00001
#define EPS_HOC 1.0e-7

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

/* ---------------------------------------------------------------------- */

PPPMOffcentre::PPPMOffcentre(LAMMPS *lmp) :
  KSpace(lmp),
  factors(NULL), density_brick(NULL), vdx_brick(NULL), vdy_brick(NULL),
  vdz_brick(NULL), u_brick(NULL), v0_brick(NULL), v1_brick(NULL),
  v2_brick(NULL), v3_brick(NULL), v4_brick(NULL), v5_brick(NULL),
  greensfn(NULL), vg(NULL), fkx(NULL), fky(NULL), fkz(NULL), density_fft(NULL),
  work1(NULL), work2(NULL), gf_b(NULL), rho1d(NULL), rho_coeff(NULL),
  drho1d(NULL), drho_coeff(NULL), sf_precoeff1(NULL), sf_precoeff2(NULL),
  sf_precoeff3(NULL), sf_precoeff4(NULL), sf_precoeff5(NULL),
  sf_precoeff6(NULL), acons(NULL), density_A_brick(NULL), density_B_brick(NULL),
  density_A_fft(NULL), density_B_fft(NULL), fft1(NULL), fft2(NULL), remap(NULL),
  cg(NULL), cg_peratom(NULL), part2grid(NULL), boxlo(NULL), ncharges(0) {
    peratom_allocate_flag = 0; group_allocate_flag = 0;

  pppmflag = 1;
  group_group_enable = 1;
  triclinic = domain->triclinic;

  nfactors = 3;
  factors = new int[nfactors];
  factors[0] = 2;
  factors[1] = 3;
  factors[2] = 5;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nfft_both = 0;
  nxhi_in = nxlo_in = nxhi_out = nxlo_out = 0;
  nyhi_in = nylo_in = nyhi_out = nylo_out = 0;
  nzhi_in = nzlo_in = nzhi_out = nzlo_out = 0;

  density_brick = vdx_brick = vdy_brick = vdz_brick = NULL;
  density_fft = NULL;
  u_brick = NULL;
  v0_brick = v1_brick = v2_brick = v3_brick = v4_brick = v5_brick = NULL;
  greensfn = NULL;
  work1 = work2 = NULL;
  vg = NULL;
  fkx = fky = fkz = NULL;

  sf_precoeff1 = sf_precoeff2 = sf_precoeff3 =
    sf_precoeff4 = sf_precoeff5 = sf_precoeff6 = NULL;

  density_A_brick = density_B_brick = NULL;
  density_A_fft = density_B_fft = NULL;

  gf_b = NULL;
  rho1d = rho_coeff = drho1d = drho_coeff = NULL;

  fft1 = fft2 = NULL;
  remap = NULL;
  cg = NULL;
  cg_peratom = NULL;

  nmax = 0;
  part2grid = NULL;

  // define acons coefficients for estimation of kspace errors
  // see JCP 109, pg 7698 for derivation of coefficients
  // higher order coefficients may be computed if needed

  memory->create(acons,8,7,"pppmoffcentre:acons");
  acons[1][0] = 2.0 / 3.0;
  acons[2][0] = 1.0 / 50.0;
  acons[2][1] = 5.0 / 294.0;
  acons[3][0] = 1.0 / 588.0;
  acons[3][1] = 7.0 / 1440.0;
  acons[3][2] = 21.0 / 3872.0;
  acons[4][0] = 1.0 / 4320.0;
  acons[4][1] = 3.0 / 1936.0;
  acons[4][2] = 7601.0 / 2271360.0;
  acons[4][3] = 143.0 / 28800.0;
  acons[5][0] = 1.0 / 23232.0;
  acons[5][1] = 7601.0 / 13628160.0;
  acons[5][2] = 143.0 / 69120.0;
  acons[5][3] = 517231.0 / 106536960.0;
  acons[5][4] = 106640677.0 / 11737571328.0;
  acons[6][0] = 691.0 / 68140800.0;
  acons[6][1] = 13.0 / 57600.0;
  acons[6][2] = 47021.0 / 35512320.0;
  acons[6][3] = 9694607.0 / 2095994880.0;
  acons[6][4] = 733191589.0 / 59609088000.0;
  acons[6][5] = 326190917.0 / 11700633600.0;
  acons[7][0] = 1.0 / 345600.0;
  acons[7][1] = 3617.0 / 35512320.0;
  acons[7][2] = 745739.0 / 838397952.0;
  acons[7][3] = 56399353.0 / 12773376000.0;
  acons[7][4] = 25091609.0 / 1560084480.0;
  acons[7][5] = 1755948832039.0 / 36229939200000.0;
  acons[7][6] = 4887769399.0 / 37838389248.0;

  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec)
    error->all(FLERR,"PPPM offcentre requires atom style ellipsoid");
}

/* ---------------------------------------------------------------------- */

void PPPMOffcentre::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal kspace_style pppm command");
  accuracy_relative = fabs(force->numeric(FLERR,arg[0]));

  // offcentre variant begins
  int nCoulSites = force->inumeric(FLERR,arg[1]);
  int atomType[nCoulSites+1];

  nsites = new int[atom->ntypes+1];
  for (int t = 1; t <= atom->ntypes; ++t)
    nsites[t] = 0;

  molFrameSite = new double**[atom->ntypes+1];
  molFrameCharge = new double*[atom->ntypes+1];

  max_nsites = 0;
  int totsites = 0;
  int start_sitesspec_argcount = 2;
  int argcount = start_sitesspec_argcount;
  for (int t = 1; t <= nCoulSites; ++t) {
    atomType[t] = force->inumeric(FLERR,arg[argcount++]);
    ++nsites[atomType[t]];

    max_nsites = nsites[atomType[t]] > max_nsites ?
      nsites[atomType[t]] : max_nsites;

    argcount += 4;
  }

  argcount = start_sitesspec_argcount;
  for (int type = 1; type <= atom->ntypes; ++type) {
    molFrameSite[type] = new double*[nsites[type]+1];
    molFrameCharge[type] = new double[nsites[type]+1];

    for (int site = 1; site <= nsites[type]; ++site) {
      ++argcount;

      molFrameSite[type][site] = new double[3];
      molFrameSite[type][site][0] = force->numeric(FLERR,arg[argcount++]);
      molFrameSite[type][site][1] = force->numeric(FLERR,arg[argcount++]);
      molFrameSite[type][site][2] = force->numeric(FLERR,arg[argcount++]);

      molFrameCharge[type][site] = force->numeric(FLERR,arg[argcount++]);
    }

    totsites += nsites[type];
  }
  // offcentre variant ends

  if (narg > argcount) {
    fprintf(stderr, "number of specified charges exceeds the declared %i\n",
            nCoulSites);
    exit(1);
  }
}

/* ----------------------------------------------------------------------
   free all memory
   ------------------------------------------------------------------------- */

PPPMOffcentre::~PPPMOffcentre()
{
  if (copymode) return;

  delete [] factors;
  deallocate();
  if (peratom_allocate_flag) deallocate_peratom();
  if (group_allocate_flag) deallocate_groups();
  memory->destroy(part2grid);
  memory->destroy(acons);
}

/* ----------------------------------------------------------------------
   called once before run
   ------------------------------------------------------------------------- */

void PPPMOffcentre::init()
{
  if (me == 0) {
    if (screen) fprintf(screen,"PPPMOffcentre initialization ...\n");
    if (logfile) fprintf(logfile,"PPPMOffcentre initialization ...\n");
  }

  // error check

  triclinic_check();

  if (triclinic != domain->triclinic)
    error->all(FLERR,"Must redefine kspace_style after changing to triclinic "
               "box");

  if (domain->triclinic && differentiation_flag == 1)
    error->all(FLERR,"Cannot (yet) use PPPMOffcentre with triclinic box and "
               "kspace_modify diff ad");
  if (domain->triclinic && slabflag)
    error->all(FLERR,"Cannot (yet) use PPPMOffcentre with triclinic box and "
               "slab correction");
  if (domain->dimension == 2) error->all(FLERR, "Cannot use PPPMOffcentre with"
                                         "2d simulation");
  if (comm->style != 0)
    error->universe_all(FLERR,"PPPMOffcentre can only currently be used with "
                        "comm_style brick");

  //if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute
  //q");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with PPPMOffcentre");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab PPPMOffcentre");
  }

  if (order < 2 || order > MAXORDER) {
    char str[128];
    sprintf(str,"PPPMOffcentre order cannot be < 2 or > than %d",MAXORDER);
    error->all(FLERR,str);
  }

  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  pair_check();

  int itmp = 0;
  double *p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
  if (p_cutoff == NULL)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  cutoff = *p_cutoff;

  // compute ncharges
  int *type = atom->type;
  int nlocal = atom->nlocal;

  bigint ncharges_local(0);

#if defined(_OPENMP)
  // icpc bug: error for nlocal, type
  // solved by adding firstprivate(nlocal, type) to pragma
#pragma omp parallel for default(none) firstprivate(nlocal, type)\
  reduction(+:ncharges_local)
#endif
  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    ncharges_local += nsites[itype];
  }

  MPI_Allreduce(&ncharges_local,
                &ncharges,
                1,
                MPI_LMP_BIGINT,
                MPI_SUM,
                world);

  // search for max offset due to offcentre charges useful for qdist below
  double max_charge_offset_local = 0.0;
  double scratch = 0.0;

  for (int i = 0; i < atom->nlocal; i++) {
    int itype = type[i];
    if (nsites[itype] > 0) {
      for (int s = 1; s <= nsites[itype]; ++s) {
        scratch = fabs(molFrameSite[itype][s][0]);
        max_charge_offset_local = scratch > max_charge_offset_local
          ? scratch : max_charge_offset_local;
        scratch = fabs(molFrameSite[itype][s][1]);
        max_charge_offset_local = scratch > max_charge_offset_local
          ? scratch : max_charge_offset_local;
        scratch = fabs(molFrameSite[itype][s][2]);
        max_charge_offset_local = scratch > max_charge_offset_local
          ? scratch : max_charge_offset_local;
      }
    }
  }

  MPI_Allreduce(&max_charge_offset_local,
                &max_charge_offset,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                world);

  // if kspace is TIP4P, extract TIP4P params from pair style
  // bond/angle are not yet init(), so insure equilibrium request is valid

  qdist = 0.0;

  if (tip4pflag) {
    if (me == 0) {
      if (screen) fprintf(screen,"  extracting TIP4P info from pair style\n");
      if (logfile) fprintf(logfile,"  extracting TIP4P info from pair style\n");
    }

    double *p_qdist = (double *) force->pair->extract("qdist",itmp);
    int *p_typeO = (int *) force->pair->extract("typeO",itmp);
    int *p_typeH = (int *) force->pair->extract("typeH",itmp);
    int *p_typeA = (int *) force->pair->extract("typeA",itmp);
    int *p_typeB = (int *) force->pair->extract("typeB",itmp);
    if (!p_qdist || !p_typeO || !p_typeH || !p_typeA || !p_typeB)
      error->all(FLERR,"Pair style is incompatible with TIP4P KSpace style");
    qdist = *p_qdist;
    typeO = *p_typeO;
    typeH = *p_typeH;
    int typeA = *p_typeA;
    int typeB = *p_typeB;

    if (force->angle == NULL || force->bond == NULL ||
        force->angle->setflag == NULL || force->bond->setflag == NULL)
      error->all(FLERR,"Bond and angle potentials must be defined for TIP4P");
    if (typeA < 1 || typeA > atom->nangletypes ||
        force->angle->setflag[typeA] == 0)
      error->all(FLERR,"Bad TIP4P angle type for PPPMOffcentre/TIP4P");
    if (typeB < 1 || typeB > atom->nbondtypes ||
        force->bond->setflag[typeB] == 0)
      error->all(FLERR,"Bad TIP4P bond type for PPPMOffcentre/TIP4P");
    double theta = force->angle->equilibrium_angle(typeA);
    double blen = force->bond->equilibrium_distance(typeB);
    alpha = qdist / (cos(0.5*theta) * blen);
  } else {
    qdist = max_charge_offset;
  }

  scale = 1.0;
  qqrd2e = force->qqrd2e;
  qsum_qsq();
  natoms_original = atom->natoms;

  // set accuracy (force units) from accuracy_relative or accuracy_absolute

  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

  // free all arrays previously allocated

  deallocate();
  if (peratom_allocate_flag) deallocate_peratom();
  if (group_allocate_flag) deallocate_groups();

  // setup FFT grid resolution and g_ewald
  // normally one iteration thru while loop is all that is required
  // if grid stencil does not extend beyond neighbor proc
  //   or overlap is allowed, then done
  // else reduce order and try again

  int (*procneigh)[2] = comm->procneigh;

  GridComm *cgtmp = NULL;
  int iteration = 0;

  while (order >= minorder) {
    if (iteration && me == 0)
      error->warning(FLERR,"Reducing PPPMOffcentre order b/c stencil extends "
                     "beyond nearest neighbor processor");

    if (stagger_flag && !differentiation_flag) compute_gf_denom();
    set_grid_global();
    set_grid_local();
    if (overlap_allowed) break;

    cgtmp = new GridComm(lmp,world,1,1,
                         nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                         nxlo_out,nxhi_out,
                         nylo_out,nyhi_out,
                         nzlo_out,nzhi_out,
                         procneigh[0][0],procneigh[0][1],procneigh[1][0],
                         procneigh[1][1],procneigh[2][0],procneigh[2][1]);
    cgtmp->ghost_notify();
    if (!cgtmp->ghost_overlap()) break;
    delete cgtmp;

    order--;
    iteration++;
  }

  if (order < minorder) error->all(FLERR, "PPPMOffcentre order < minimum "
                                   "allowed order");
  if (!overlap_allowed && cgtmp->ghost_overlap())
    error->all(FLERR,"PPPMOffcentre grid stencil extends "
               "beyond nearest neighbor processor");
  if (cgtmp) delete cgtmp;

  // adjust g_ewald

  if (!gewaldflag) adjust_gewald();

  // calculate the final accuracy

  double estimated_accuracy = final_accuracy();

  // print stats

  int ngrid_max,nfft_both_max;
  MPI_Allreduce(&ngrid,&ngrid_max,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&nfft_both,&nfft_both_max,1,MPI_INT,MPI_MAX,world);

  if (me == 0) {

#ifdef FFT_SINGLE
    const char fft_prec[] = "single";
#else
    const char fft_prec[] = "double";
#endif

    if (screen) {
      fprintf(screen,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(screen,"  grid = %d %d %d\n",nx_pppm,ny_pppm,nz_pppm);
      fprintf(screen,"  stencil order = %d\n",order);
      fprintf(screen,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(screen,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(screen,"  using %s precision FFTs\n",fft_prec);
      fprintf(screen,"  3d grid and FFT values/proc = %d %d\n",
              ngrid_max,nfft_both_max);
    }
    if (logfile) {
      fprintf(logfile,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(logfile,"  grid = %d %d %d\n",nx_pppm,ny_pppm,nz_pppm);
      fprintf(logfile,"  stencil order = %d\n",order);
      fprintf(logfile,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(logfile,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(logfile,"  using %s precision FFTs\n",fft_prec);
      fprintf(logfile,"  3d grid and FFT values/proc = %d %d\n",
              ngrid_max,nfft_both_max);
    }
  }

  // allocate K-space dependent memory
  // don't invoke allocate peratom() or group(), will be allocated when needed

  allocate();
  cg->ghost_notify();
  cg->setup();

  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom();
  if (differentiation_flag == 1) compute_sf_precoeff();
  compute_rho_coeff();
}

/* ----------------------------------------------------------------------
   adjust PPPMOffcentre coeffs, called initially and whenever volume has changed
   ------------------------------------------------------------------------- */

void PPPMOffcentre::setup()
{
  if (triclinic) {
    setup_triclinic();
    return;
  }

  // perform some checks to avoid illegal boundaries with read_data

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with PPPMOffcentre");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab PPPMOffcentre");
  }

  int i,j,k,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPMOffcentre
  // z dimension for 3d PPPMOffcentre is zprd since slab_volfactor = 1.0

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  delxinv = nx_pppm/xprd;
  delyinv = ny_pppm/yprd;
  delzinv = nz_pppm/zprd_slab;

  delvolinv = delxinv*delyinv*delzinv;

  double unitkx = (MY_2PI/xprd);
  double unitky = (MY_2PI/yprd);
  double unitkz = (MY_2PI/zprd_slab);

  // fkx,fky,fkz for my FFT grid pts

  double per;

  for (i = nxlo_fft; i <= nxhi_fft; i++) {
    per = i - nx_pppm*(2*i/nx_pppm);
    fkx[i] = unitkx*per;
  }

  for (i = nylo_fft; i <= nyhi_fft; i++) {
    per = i - ny_pppm*(2*i/ny_pppm);
    fky[i] = unitky*per;
  }

  for (i = nzlo_fft; i <= nzhi_fft; i++) {
    per = i - nz_pppm*(2*i/nz_pppm);
    fkz[i] = unitkz*per;
  }

  // virial coefficients

  double sqk,vterm;

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++) {
    for (j = nylo_fft; j <= nyhi_fft; j++) {
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        sqk = fkx[i]*fkx[i] + fky[j]*fky[j] + fkz[k]*fkz[k];
        if (sqk == 0.0) {
          vg[n][0] = 0.0;
          vg[n][1] = 0.0;
          vg[n][2] = 0.0;
          vg[n][3] = 0.0;
          vg[n][4] = 0.0;
          vg[n][5] = 0.0;
        } else {
          vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
          vg[n][0] = 1.0 + vterm*fkx[i]*fkx[i];
          vg[n][1] = 1.0 + vterm*fky[j]*fky[j];
          vg[n][2] = 1.0 + vterm*fkz[k]*fkz[k];
          vg[n][3] = vterm*fkx[i]*fky[j];
          vg[n][4] = vterm*fkx[i]*fkz[k];
          vg[n][5] = vterm*fky[j]*fkz[k];
        }
        n++;
      }
    }
  }

  if (differentiation_flag == 1) compute_gf_ad();
  else compute_gf_ik();
}

/* ----------------------------------------------------------------------
   adjust PPPMOffcentre coeffs, called initially and whenever volume has changed
   for a triclinic system
   ------------------------------------------------------------------------- */

void PPPMOffcentre::setup_triclinic()
{
  int i,j,k,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPMOffcentre
  // z dimension for 3d PPPMOffcentre is zprd since slab_volfactor = 1.0

  prd = domain->prd;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  // use lamda (0-1) coordinates

  delxinv = nx_pppm;
  delyinv = ny_pppm;
  delzinv = nz_pppm;
  delvolinv = delxinv*delyinv*delzinv/volume;

  // fkx,fky,fkz for my FFT grid pts

  double per_i,per_j,per_k;

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++) {
    per_k = k - nz_pppm*(2*k/nz_pppm);
    for (j = nylo_fft; j <= nyhi_fft; j++) {
      per_j = j - ny_pppm*(2*j/ny_pppm);
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        per_i = i - nx_pppm*(2*i/nx_pppm);

        double unitk_lamda[3];
        unitk_lamda[0] = 2.0*MY_PI*per_i;
        unitk_lamda[1] = 2.0*MY_PI*per_j;
        unitk_lamda[2] = 2.0*MY_PI*per_k;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
        fkx[n] = unitk_lamda[0];
        fky[n] = unitk_lamda[1];
        fkz[n] = unitk_lamda[2];
        n++;
      }
    }
  }

  // virial coefficients

  double sqk,vterm;

  for (n = 0; n < nfft; n++) {
    sqk = fkx[n]*fkx[n] + fky[n]*fky[n] + fkz[n]*fkz[n];
    if (sqk == 0.0) {
      vg[n][0] = 0.0;
      vg[n][1] = 0.0;
      vg[n][2] = 0.0;
      vg[n][3] = 0.0;
      vg[n][4] = 0.0;
      vg[n][5] = 0.0;
    } else {
      vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
      vg[n][0] = 1.0 + vterm*fkx[n]*fkx[n];
      vg[n][1] = 1.0 + vterm*fky[n]*fky[n];
      vg[n][2] = 1.0 + vterm*fkz[n]*fkz[n];
      vg[n][3] = vterm*fkx[n]*fky[n];
      vg[n][4] = vterm*fkx[n]*fkz[n];
      vg[n][5] = vterm*fky[n]*fkz[n];
    }
  }

  compute_gf_ik_triclinic();
}

/* ----------------------------------------------------------------------
   reset local grid arrays and communication stencils
   called by fix balance b/c it changed sizes of processor sub-domains
   ------------------------------------------------------------------------- */

void PPPMOffcentre::setup_grid()
{
  // free all arrays previously allocated

  deallocate();
  if (peratom_allocate_flag) deallocate_peratom();
  if (group_allocate_flag) deallocate_groups();

  // reset portion of global grid that each proc owns

  set_grid_local();

  // reallocate K-space dependent memory
  // check if grid communication is now overlapping if not allowed
  // don't invoke allocate peratom() or group(), will be allocated when needed

  allocate();

  cg->ghost_notify();
  if (overlap_allowed == 0 && cg->ghost_overlap())
    error->all(FLERR,"PPPMOffcentre grid stencil extends "
               "beyond nearest neighbor processor");
  cg->setup();

  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom();
  if (differentiation_flag == 1) compute_sf_precoeff();
  compute_rho_coeff();

  // pre-compute volume-dependent coeffs

  setup();
}

/* ----------------------------------------------------------------------
   compute qsum,qsqsum,q2 and give error/warning if not charge neutral
   called initially, when particle count changes, when charges are changed
   ------------------------------------------------------------------------- */

void PPPMOffcentre::qsum_qsq()
{
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double qsum_local(0.0), qsqsum_local(0.0);

#if defined(_OPENMP)
  // icpc bug: error for nlocal, type
  // solved by adding firstprivate(nlocal, type) to pragma
#pragma omp parallel for default(none) firstprivate(nlocal, type) \
  reduction(+:qsum_local,qsqsum_local)
#endif
  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    for (int s = 1; s <= nsites[itype]; ++s) {
      double qi = molFrameCharge[itype][s];
      qsum_local += qi;
      qsqsum_local += qi*qi;
    }
  }

  MPI_Allreduce(&qsum_local,&qsum,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&qsqsum_local,&qsqsum,1,MPI_DOUBLE,MPI_SUM,world);

  if ((qsqsum == 0.0) && (comm->me == 0) && warn_nocharge) {
    error->warning(FLERR,"Using kspace solver on system with no charge");
    warn_nocharge = 0;
  }

  q2 = qsqsum * force->qqrd2e;

  // not yet sure of the correction needed for non-neutral systems
  // so issue warning or error

  if (fabs(qsum) > SMALL) {
    char str[128];
    sprintf(str,"System is not charge neutral, net charge = %g",qsum);
    if (!warn_nonneutral) error->all(FLERR,str);
    if (warn_nonneutral == 1 && comm->me == 0) error->warning(FLERR,str);
    warn_nonneutral = 2;
  }
}

/* ----------------------------------------------------------------------
   compute the PPPMOffcentre long-range force, energy, virial
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute(int eflag, int vflag)
{
  int i,j;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = evflag_atom = eflag_global = vflag_global =
    eflag_atom = vflag_atom = 0;

  if (evflag_atom && !peratom_allocate_flag) {
    allocate_peratom();
    cg_peratom->ghost_notify();
    cg_peratom->setup();
  }

  // if atom count has changed, update qsum and qsqsum

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  // return if there are no charges

  if (qsqsum == 0.0) return;

  //MATTEO probably should do something here, convert also offcntre positions
  // void Domain::x2lamda(int n) -> for offcentre just use void
  // Domain::lamda2x(double *lamda, double *x) ??? not sure maybe shoudl
  // do some transform to orientations too ? otherwise offcentre
  // position computed on the fly based on centres, which are already
  // converted

  // how are offcentre charges to be remapped among periodic replicas?
  // by now just out of box

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }

  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy(part2grid);
    nmax = atom->nmax;
    memory->create(part2grid,nmax,max_nsites,3,"pppmoffcentre:part2grid");
  }

  // find grid points for all my particles
  // map my particle charge onto my local 3d density grid

  particle_map();
  make_rho();

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  // compute potential gradient on my FFT grid and
  //   portion of e_long on this proc's FFT grid
  // return gradients (electric fields) in 3d brick decomposition
  // also performs per-atom calculations via poisson_peratom()

  poisson();

  // all procs communicate E-field values
  // to fill ghost cells surrounding their 3d bricks

  if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
  else cg->forward_comm(this,FORWARD_IK);

  // extra per-atom energy/virial communication

  if (evflag_atom) {
    if (differentiation_flag == 1 && vflag_atom)
      cg_peratom->forward_comm(this,FORWARD_AD_PERATOM);
    else if (differentiation_flag == 0)
      cg_peratom->forward_comm(this,FORWARD_IK_PERATOM);
  }

  // calculate the force on my particles

  fieldforce();

  // extra per-atom energy/virial communication

  if (evflag_atom) fieldforce_peratom();

  // sum global energy across procs and add in volume-dependent term

  const double qscale = qqrd2e * scale;

  if (eflag_global) {
    double energy_all;
    MPI_Allreduce(&energy,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
    energy = energy_all;

    energy *= 0.5*volume;
    energy -= g_ewald*qsqsum/MY_PIS +
      MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy *= qscale;
  }

  // sum global virial across procs

  if (vflag_global) {
    double virial_all[6];
    MPI_Allreduce(virial,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    for (i = 0; i < 6; i++) virial[i] = 0.5*qscale*volume*virial_all[i];
  }

  // per-atom energy/virial
  // energy includes self-energy correction
  // ntotal accounts for TIP4P tallying eatom/vatom for ghost atoms

  if (evflag_atom) {
    int nlocal = atom->nlocal;
    int *type = atom->type;
    int ntotal = nlocal;
    if (tip4pflag) ntotal += atom->nghost;

    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        int itype = type[i];
        double qi_all = 0.0;
        for (int s = 1; s <= nsites[itype]; ++s) {
          double qi = molFrameCharge[itype][s];
          qi_all += qi;
        }

        eatom[i] *= 0.5;
        eatom[i] -= g_ewald*qi_all*qi_all/MY_PIS + MY_PI2*qi_all*qsum /
          (g_ewald*g_ewald*volume);
        eatom[i] *= qscale;
      }
      for (i = nlocal; i < ntotal; i++) eatom[i] *= 0.5*qscale;
    }

    if (vflag_atom) {
      for (i = 0; i < ntotal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= 0.5*qscale;
    }
  }

  // 2d slab correction

  if (slabflag == 1) slabcorr();

  // convert atoms back from lamda to box coords

  //MATTEO probably should do something here, convert also offcntre positions
  if (triclinic != 0) domain->lamda2x(atom->nlocal);
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors and order
   ------------------------------------------------------------------------- */

void PPPMOffcentre::allocate()
{
  memory->create3d_offset(density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:density_brick");

  memory->create(density_fft,nfft_both,"pppmoffcentre:density_fft");
  memory->create(greensfn,nfft_both,"pppmoffcentre:greensfn");
  memory->create(work1,2*nfft_both,"pppmoffcentre:work1");
  memory->create(work2,2*nfft_both,"pppmoffcentre:work2");
  memory->create(vg,nfft_both,6,"pppmoffcentre:vg");

  if (triclinic == 0) {
    memory->create1d_offset(fkx,nxlo_fft,nxhi_fft,"pppmoffcentre:fkx");
    memory->create1d_offset(fky,nylo_fft,nyhi_fft,"pppmoffcentre:fky");
    memory->create1d_offset(fkz,nzlo_fft,nzhi_fft,"pppmoffcentre:fkz");
  } else {
    memory->create(fkx,nfft_both,"pppmoffcentre:fkx");
    memory->create(fky,nfft_both,"pppmoffcentre:fky");
    memory->create(fkz,nfft_both,"pppmoffcentre:fkz");
  }

  if (differentiation_flag == 1) {
    memory->create3d_offset(u_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"pppmoffcentre:u_brick");

    memory->create(sf_precoeff1,nfft_both,"pppmoffcentre:sf_precoeff1");
    memory->create(sf_precoeff2,nfft_both,"pppmoffcentre:sf_precoeff2");
    memory->create(sf_precoeff3,nfft_both,"pppmoffcentre:sf_precoeff3");
    memory->create(sf_precoeff4,nfft_both,"pppmoffcentre:sf_precoeff4");
    memory->create(sf_precoeff5,nfft_both,"pppmoffcentre:sf_precoeff5");
    memory->create(sf_precoeff6,nfft_both,"pppmoffcentre:sf_precoeff6");

  } else {
    memory->create3d_offset(vdx_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"pppmoffcentre:vdx_brick");
    memory->create3d_offset(vdy_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"pppmoffcentre:vdy_brick");
    memory->create3d_offset(vdz_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"pppmoffcentre:vdz_brick");
  }

  // summation coeffs

  order_allocated = order;
  if (!stagger_flag) memory->create(gf_b,order,"pppmoffcentre:gf_b");
  memory->create2d_offset(rho1d,3,-order/2,order/2,"pppmoffcentre:rho1d");
  memory->create2d_offset(drho1d,3,-order/2,order/2,"pppmoffcentre:drho1d");
  memory->create2d_offset(rho_coeff,order,(1-order)/2,order/2,
                          "pppmoffcentre:rho_coeff");
  memory->create2d_offset(drho_coeff,order,(1-order)/2,order/2,
                          "pppmoffcentre:drho_coeff");

  // create 2 FFTs and a Remap
  // 1st FFT keeps data in FFT decompostion
  // 2nd FFT returns data in 3d brick decomposition
  // remap takes data from 3d brick to FFT decomposition

  int tmp;

  fft1 = new FFT3d(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   0,0,&tmp,collective_flag);

  fft2 = new FFT3d(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                   0,0,&tmp,collective_flag);

  remap = new Remap(lmp,world,
                    nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                    nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                    1,0,0,FFT_PRECISION,collective_flag);

  // create ghost grid object for rho and electric field communication

  int (*procneigh)[2] = comm->procneigh;

  if (differentiation_flag == 1)
    cg = new GridComm(lmp,world,1,1,
                      nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                      nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out,
                      procneigh[0][0],procneigh[0][1],procneigh[1][0],
                      procneigh[1][1],procneigh[2][0],procneigh[2][1]);
  else
    cg = new GridComm(lmp,world,3,1,
                      nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                      nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out,
                      procneigh[0][0],procneigh[0][1],procneigh[1][0],
                      procneigh[1][1],procneigh[2][0],procneigh[2][1]);
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors and order
   ------------------------------------------------------------------------- */

void PPPMOffcentre::deallocate()
{
  memory->destroy3d_offset(density_brick,nzlo_out,nylo_out,nxlo_out);

  if (differentiation_flag == 1) {
    memory->destroy3d_offset(u_brick,nzlo_out,nylo_out,nxlo_out);
    memory->destroy(sf_precoeff1);
    memory->destroy(sf_precoeff2);
    memory->destroy(sf_precoeff3);
    memory->destroy(sf_precoeff4);
    memory->destroy(sf_precoeff5);
    memory->destroy(sf_precoeff6);
  } else {
    memory->destroy3d_offset(vdx_brick,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdy_brick,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdz_brick,nzlo_out,nylo_out,nxlo_out);
  }

  memory->destroy(density_fft);
  memory->destroy(greensfn);
  memory->destroy(work1);
  memory->destroy(work2);
  memory->destroy(vg);

  if (triclinic == 0) {
    memory->destroy1d_offset(fkx,nxlo_fft);
    memory->destroy1d_offset(fky,nylo_fft);
    memory->destroy1d_offset(fkz,nzlo_fft);
  } else {
    memory->destroy(fkx);
    memory->destroy(fky);
    memory->destroy(fkz);
  }

  memory->destroy(gf_b);
  if (stagger_flag) gf_b = NULL;
  memory->destroy2d_offset(rho1d,-order_allocated/2);
  memory->destroy2d_offset(drho1d,-order_allocated/2);
  memory->destroy2d_offset(rho_coeff,(1-order_allocated)/2);
  memory->destroy2d_offset(drho_coeff,(1-order_allocated)/2);

  delete fft1;
  delete fft2;
  delete remap;
  delete cg;
}

/* ----------------------------------------------------------------------
   allocate per-atom memory that depends on # of K-vectors and order
   ------------------------------------------------------------------------- */

void PPPMOffcentre::allocate_peratom()
{
  peratom_allocate_flag = 1;

  if (differentiation_flag != 1)
    memory->create3d_offset(u_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"pppmoffcentre:u_brick");

  memory->create3d_offset(v0_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:v0_brick");

  memory->create3d_offset(v1_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:v1_brick");
  memory->create3d_offset(v2_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:v2_brick");
  memory->create3d_offset(v3_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:v3_brick");
  memory->create3d_offset(v4_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:v4_brick");
  memory->create3d_offset(v5_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:v5_brick");

  // create ghost grid object for rho and electric field communication

  int (*procneigh)[2] = comm->procneigh;

  if (differentiation_flag == 1)
    cg_peratom =
      new GridComm(lmp,world,6,1,
                   nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                   nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out,
                   procneigh[0][0],procneigh[0][1],procneigh[1][0],
                   procneigh[1][1],procneigh[2][0],procneigh[2][1]);
  else
    cg_peratom =
      new GridComm(lmp,world,7,1,
                   nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                   nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out,
                   procneigh[0][0],procneigh[0][1],procneigh[1][0],
                   procneigh[1][1],procneigh[2][0],procneigh[2][1]);
}

/* ----------------------------------------------------------------------
   deallocate per-atom memory that depends on # of K-vectors and order
   ------------------------------------------------------------------------- */

void PPPMOffcentre::deallocate_peratom()
{
  peratom_allocate_flag = 0;

  memory->destroy3d_offset(v0_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(v1_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(v2_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(v3_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(v4_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(v5_brick,nzlo_out,nylo_out,nxlo_out);

  if (differentiation_flag != 1)
    memory->destroy3d_offset(u_brick,nzlo_out,nylo_out,nxlo_out);

  delete cg_peratom;
}

/* ----------------------------------------------------------------------
   set global size of PPPMOffcentre grid = nx,ny,nz_pppm
   used for charge accumulation, FFTs, and electric field interpolation
   ------------------------------------------------------------------------- */

void PPPMOffcentre::set_grid_global()
{
  // use xprd,yprd,zprd (even if triclinic, and then scale later)
  // adjust z dimension for 2d slab PPPMOffcentre
  // 3d PPPMOffcentre just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;

  // printf("PPPMOffcentre::set_grid_global %f %f %f\n", xprd, yprd, zprd);

  // make initial g_ewald estimate
  // based on desired accuracy and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  double h;
  //bigint natoms = atom->natoms;

  if (!gewaldflag) {
    if (accuracy <= 0.0)
      error->all(FLERR,"KSpace accuracy must be > 0");
    g_ewald = accuracy*sqrt(ncharges*cutoff*xprd*yprd*zprd) / (2.0*q2);
    //g_ewald = accuracy*sqrt(atom->natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
    if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
    else g_ewald = sqrt(-log(g_ewald)) / cutoff;
  }

  // set optimal nx_pppm,ny_pppm,nz_pppm based on order and accuracy
  // nz_pppm uses extended zprd_slab instead of zprd
  // reduce it until accuracy target is met

  // printf("PPPMOffcentre::set_grid_global order %i accuracy %f 1/gewald %f\n",
  // 	 order, accuracy, 1.0/g_ewald);

  if (!gridflag) {

    if (differentiation_flag == 1 || stagger_flag) {

      h = h_x = h_y = h_z = 4.0/g_ewald;
      int count = 0;
      while (1) {

        // set grid dimension
        nx_pppm = static_cast<int> (xprd/h_x);
        ny_pppm = static_cast<int> (yprd/h_y);
        nz_pppm = static_cast<int> (zprd_slab/h_z);

        if (nx_pppm <= 1) nx_pppm = 2;
        if (ny_pppm <= 1) ny_pppm = 2;
        if (nz_pppm <= 1) nz_pppm = 2;

        //set local grid dimension
        int npey_fft = -1, npez_fft = -1;
        if (nz_pppm >= nprocs) {
          npey_fft = 1;
          npez_fft = nprocs;
        } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

        int me_y = me % npey_fft;
        int me_z = me / npey_fft;

        nxlo_fft = 0;
        nxhi_fft = nx_pppm - 1;
        nylo_fft = me_y*ny_pppm/npey_fft;
        nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
        nzlo_fft = me_z*nz_pppm/npez_fft;
        nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;

        double df_kspace = compute_df_kspace();

        count++;

        // break loop if the accuracy has been reached or
        // too many loops have been performed

        if (df_kspace <= accuracy) break;
        if (count > 500) error->all(FLERR, "Could not compute grid size");
        h *= 0.95;
        h_x = h_y = h_z = h;
      }

    } else {

      double err;
      h_x = h_y = h_z = 1.0/g_ewald;

      nx_pppm = static_cast<int> (xprd/h_x) + 1;
      ny_pppm = static_cast<int> (yprd/h_y) + 1;
      nz_pppm = static_cast<int> (zprd_slab/h_z) + 1;

      err = estimate_ik_error(h_x,xprd,ncharges);
      while (err > accuracy) {
        err = estimate_ik_error(h_x,xprd,ncharges);
        nx_pppm++;
        h_x = xprd/nx_pppm;
      }

      err = estimate_ik_error(h_y,yprd,ncharges);
      while (err > accuracy) {
        err = estimate_ik_error(h_y,yprd,ncharges);
        ny_pppm++;
        h_y = yprd/ny_pppm;
      }

      err = estimate_ik_error(h_z,zprd_slab,ncharges);
      while (err > accuracy) {
        err = estimate_ik_error(h_z,zprd_slab,ncharges);
        nz_pppm++;
        h_z = zprd_slab/nz_pppm;
      }
    }

    // scale grid for triclinic skew

    if (triclinic) {
      double tmp[3];
      tmp[0] = nx_pppm/xprd;
      tmp[1] = ny_pppm/yprd;
      tmp[2] = nz_pppm/zprd;
      lamda2xT(&tmp[0],&tmp[0]);
      nx_pppm = static_cast<int>(tmp[0]) + 1;
      ny_pppm = static_cast<int>(tmp[1]) + 1;
      nz_pppm = static_cast<int>(tmp[2]) + 1;
    }
  }

  // boost grid size until it is factorable

  while (!factorable(nx_pppm)) nx_pppm++;
  while (!factorable(ny_pppm)) ny_pppm++;
  while (!factorable(nz_pppm)) nz_pppm++;

  // printf("PPPMOffcentre::set_grid_global n_ppm %i %i %i\n",
  // 	 nx_pppm, ny_pppm, nz_pppm);

  if (triclinic == 0) {
    h_x = xprd/nx_pppm;
    h_y = yprd/ny_pppm;
    h_z = zprd_slab/nz_pppm;
  } else {
    double tmp[3];
    tmp[0] = nx_pppm;
    tmp[1] = ny_pppm;
    tmp[2] = nz_pppm;
    x2lamdaT(&tmp[0],&tmp[0]);
    h_x = 1.0/tmp[0];
    h_y = 1.0/tmp[1];
    h_z = 1.0/tmp[2];
  }

  // printf("PPPMOffcentre::set_grid_global h %f %f %f\n",
  // 	 h_x, h_y, h_z);

  if (nx_pppm >= OFFSET || ny_pppm >= OFFSET || nz_pppm >= OFFSET)
    error->all(FLERR,"PPPMOffcentre grid is too large");
}

/* ----------------------------------------------------------------------
   check if all factors of n are in list of factors
   return 1 if yes, 0 if no
   ------------------------------------------------------------------------- */

int PPPMOffcentre::factorable(int n)
{
  int i;

  while (n > 1) {
    for (i = 0; i < nfactors; i++) {
      if (n % factors[i] == 0) {
        n /= factors[i];
        break;
      }
    }
    if (i == nfactors) return 0;
  }

  return 1;
}

/* ----------------------------------------------------------------------
   compute estimated kspace force error
   ------------------------------------------------------------------------- */

double PPPMOffcentre::compute_df_kspace()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  //bigint natoms = atom->natoms;
  double df_kspace = 0.0;
  if (differentiation_flag == 1 || stagger_flag) {
    double qopt = compute_qopt();
    df_kspace = sqrt(qopt/ncharges)*q2/(xprd*yprd*zprd_slab);
  } else {
    double lprx = estimate_ik_error(h_x,xprd,ncharges);
    double lpry = estimate_ik_error(h_y,yprd,ncharges);
    double lprz = estimate_ik_error(h_z,zprd_slab,ncharges);
    df_kspace = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  }
  return df_kspace;
}

/* ----------------------------------------------------------------------
   compute qopt
   ------------------------------------------------------------------------- */

double PPPMOffcentre::compute_qopt()
{
  double qopt = 0.0;
  double *prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double u1, u2, sqk;
  double sum1,sum2,sum3,sum4,dot2;

  int k,l,m,nx,ny,nz;
  const int twoorder = 2*order;

  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    const int mper = m - nz_pppm*(2*m/nz_pppm);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      const int lper = l - ny_pppm*(2*l/ny_pppm);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        const int kper = k - nx_pppm*(2*k/nx_pppm);

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

        if (sqk != 0.0) {

          sum1 = 0.0;
          sum2 = 0.0;
          sum3 = 0.0;
          sum4 = 0.0;
          for (nx = -2; nx <= 2; nx++) {
            qx = unitkx*(kper+nx_pppm*nx);
            sx = exp(-0.25*square(qx/g_ewald));
            argx = 0.5*qx*xprd/nx_pppm;
            wx = powsinxx(argx,twoorder);
            qx *= qx;

            for (ny = -2; ny <= 2; ny++) {
              qy = unitky*(lper+ny_pppm*ny);
              sy = exp(-0.25*square(qy/g_ewald));
              argy = 0.5*qy*yprd/ny_pppm;
              wy = powsinxx(argy,twoorder);
              qy *= qy;

              for (nz = -2; nz <= 2; nz++) {
                qz = unitkz*(mper+nz_pppm*nz);
                sz = exp(-0.25*square(qz/g_ewald));
                argz = 0.5*qz*zprd_slab/nz_pppm;
                wz = powsinxx(argz,twoorder);
                qz *= qz;

                dot2 = qx+qy+qz;
                u1   = sx*sy*sz;
                u2   = wx*wy*wz;
                sum1 += u1*u1/dot2*MY_4PI*MY_4PI;
                sum2 += u1 * u2 * MY_4PI;
                sum3 += u2;
                sum4 += dot2*u2;
              }
            }
          }
          sum2 *= sum2;
          qopt += sum1 - sum2/(sum3*sum4);
        }
      }
    }
  }
  double qopt_all;
  MPI_Allreduce(&qopt,&qopt_all,1,MPI_DOUBLE,MPI_SUM,world);
  return qopt_all;
}

/* ----------------------------------------------------------------------
   estimate kspace force error for ik method
   ------------------------------------------------------------------------- */

double PPPMOffcentre::estimate_ik_error(double h, double prd, bigint ncharges)
{
  double sum = 0.0;
  for (int m = 0; m < order; m++)
    sum += acons[order][m] * pow(h*g_ewald,2.0*m);
  double value = q2 * pow(h*g_ewald,(double)order) *
    sqrt(g_ewald*prd*sqrt(MY_2PI)*sum/ncharges) / (prd*prd);

  return value;
}

/* ----------------------------------------------------------------------
   adjust the g_ewald parameter to near its optimal value
   using a Newton-Raphson solver
   ------------------------------------------------------------------------- */

void PPPMOffcentre::adjust_gewald()
{
  double dx;

  for (int i = 0; i < LARGE; i++) {
    dx = newton_raphson_f() / derivf();
    g_ewald -= dx;
    if (fabs(newton_raphson_f()) < SMALL) return;
  }

  char str[128];
  sprintf(str, "Could not compute g_ewald");
  error->all(FLERR, str);
}

/* ----------------------------------------------------------------------
   calculate f(x) using Newton-Raphson solver
   ------------------------------------------------------------------------- */

double PPPMOffcentre::newton_raphson_f()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  //bigint natoms = atom->natoms;

  double df_rspace = 2.0*q2*exp(-g_ewald*g_ewald*cutoff*cutoff) /
    sqrt(ncharges*cutoff*xprd*yprd*zprd);
  //       sqrt(natoms*cutoff*xprd*yprd*zprd);

  double df_kspace = compute_df_kspace();

  return df_rspace - df_kspace;
}

/* ----------------------------------------------------------------------
   calculate numerical derivative f'(x) using forward difference
   [f(x + h) - f(x)] / h
   ------------------------------------------------------------------------- */

double PPPMOffcentre::derivf()
{
  double h = 0.000001;  //Derivative step-size
  double df,f1,f2,g_ewald_old;

  f1 = newton_raphson_f();
  g_ewald_old = g_ewald;
  g_ewald += h;
  f2 = newton_raphson_f();
  g_ewald = g_ewald_old;
  df = (f2 - f1)/h;

  return df;
}

/* ----------------------------------------------------------------------
   calculate the final estimate of the accuracy
   ------------------------------------------------------------------------- */

double PPPMOffcentre::final_accuracy()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  //bigint natoms = atom->natoms;

  double df_kspace = compute_df_kspace();
  double q2_over_sqrt = q2 / sqrt(ncharges*cutoff*xprd*yprd*zprd);
  double df_rspace = 2.0 * q2_over_sqrt * exp(-g_ewald*g_ewald*cutoff*cutoff);
  double df_table = estimate_table_accuracy(q2_over_sqrt,df_rspace);
  double estimated_accuracy = sqrt(df_kspace*df_kspace + df_rspace*df_rspace +
                                   df_table*df_table);

  return estimated_accuracy;
}

/* ----------------------------------------------------------------------
   set local subset of PPPMOffcentre/FFT grid that I own
   n xyz lo/hi in = 3d brick that I own (inclusive)
   n xyz lo/hi out = 3d brick + ghost cells in 6 directions (inclusive)
   n xyz lo/hi fft = FFT columns that I own (all of x dim, 2d decomp in yz)
   ------------------------------------------------------------------------- */

void PPPMOffcentre::set_grid_local()
{
  // global indices of PPPMOffcentre grid range from 0 to N-1
  // nlo_in,nhi_in = lower/upper limits of the 3d sub-brick of
  //   global PPPMOffcentre grid that I own without ghost cells
  // for slab PPPMOffcentre, assign z grid as if it were not extended

  // printf("PPPMOffcentre::set_grid_local subbox %i %f %f %f %f %f %f\n",
  // 	 comm->me,
  // 	 domain->sublo_lamda[0], domain->sublo_lamda[1], domain->sublo_lamda[2],
  // 	 domain->subhi_lamda[0], domain->subhi_lamda[1], domain->subhi_lamda[2]);

  nxlo_in = static_cast<int> (comm->xsplit[comm->myloc[0]] * nx_pppm);
  nxhi_in = static_cast<int> (comm->xsplit[comm->myloc[0]+1] * nx_pppm) - 1;

  nylo_in = static_cast<int> (comm->ysplit[comm->myloc[1]] * ny_pppm);
  nyhi_in = static_cast<int> (comm->ysplit[comm->myloc[1]+1] * ny_pppm) - 1;

  nzlo_in = static_cast<int>
    (comm->zsplit[comm->myloc[2]] * nz_pppm/slab_volfactor);
  nzhi_in = static_cast<int>
    (comm->zsplit[comm->myloc[2]+1] * nz_pppm/slab_volfactor) - 1;

  // printf("PPPMOffcentre::set_grid_local %i sub\n%i %i\n%i %i\n%i %i\n",
  // 	 comm->me,
  // 	 nxlo_in, nxhi_in,
  // 	 nylo_in, nyhi_in,
  // 	 nzlo_in, nzhi_in);

  // nlower,nupper = stencil size for mapping particles to PPPMOffcentre grid

  nlower = -(order-1)/2;
  nupper = order/2;

  // printf("PPPMOffcentre::set_grid_local stencil %i %i\n",
  // 	 nlower, nupper);

  // shift values for particle <-> grid mapping
  // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

  if (order % 2) shift = OFFSET + 0.5;
  else shift = OFFSET;
  if (order % 2) shiftone = 0.0;
  else shiftone = 0.5;

  // printf("PPPMOffcentre::set_grid_local shift %f %f\n",
  // 	 shift, shiftone);

  // nlo_out,nhi_out = lower/upper limits of the 3d sub-brick of
  //   global PPPMOffcentre grid that my particles can contribute charge to
  // effectively nlo_in,nhi_in + ghost cells
  // nlo,nhi = global coords of grid pt to "lower left" of smallest/largest
  //           position a particle in my box can be at
  // dist[3] = particle position bound = subbox + skin/2.0 + qdist
  //   qdist = offset due to TIP4P fictitious charge
  //   convert to triclinic if necessary
  // nlo_out,nhi_out = nlo,nhi + stencil size for particle mapping
  // for slab PPPMOffcentre, assign z grid as if it were not extended

  double *prd,*sublo,*subhi;

  if (triclinic == 0) {
    prd = domain->prd;
    boxlo = domain->boxlo;
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    prd = domain->prd_lamda;
    boxlo = domain->boxlo_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;

  double dist[3];
  double cuthalf = 0.5*neighbor->skin + qdist;
  if (triclinic == 0) dist[0] = dist[1] = dist[2] = cuthalf;
  else kspacebbox(cuthalf,&dist[0]);

  // printf("PPPM offcentre: set_grid_local DIST %f %f %f\n",
  // 	 dist[0], dist[1], dist[2]);

  int nlo,nhi;

  nlo = static_cast<int> ((sublo[0]-dist[0]-boxlo[0]) *
                          nx_pppm/xprd + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[0]+dist[0]-boxlo[0]) *
                          nx_pppm/xprd + shift) - OFFSET;
  nxlo_out = nlo + nlower;
  nxhi_out = nhi + nupper;

  // printf("PPPM offcentre: X set_grid_local nlo %i %i %i %i\n",
  // 	 nxlo_out, nxhi_out,
  // 	 nlo, nhi
  //   );

  nlo = static_cast<int> ((sublo[1]-dist[1]-boxlo[1]) *
                          ny_pppm/yprd + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[1]+dist[1]-boxlo[1]) *
                          ny_pppm/yprd + shift) - OFFSET;
  nylo_out = nlo + nlower;
  nyhi_out = nhi + nupper;

  // printf("PPPM offcentre: Y set_grid_local nlo %i %i %i %i\n",
  // 	 nylo_out, nyhi_out,
  // 	 nlo, nhi
  //   );

  nlo = static_cast<int> ((sublo[2]-dist[2]-boxlo[2]) *
                          nz_pppm/zprd_slab + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[2]+dist[2]-boxlo[2]) *
                          nz_pppm/zprd_slab + shift) - OFFSET;
  nzlo_out = nlo + nlower;
  nzhi_out = nhi + nupper;

  // printf("PPPM offcentre: Z set_grid_local nlo %i %i %i %i\n",
  // 	 nzlo_out, nzhi_out,
  // 	 nlo, nhi
  //   );

  if (stagger_flag) {
    nxhi_out++;
    nyhi_out++;
    nzhi_out++;
  }

  // for slab PPPMOffcentre, change the grid boundary for processors at +z end
  // to include the empty volume between periodically repeating slabs
  // for slab PPPMOffcentre, want charge data communicated from -z proc to +z
  // proc, but not vice versa, also want field data communicated from +z proc
  // to  -z proc, but not vice versa
  // this is accomplished by nzhi_in = nzhi_out on +z end (no ghost cells)
  // also insure no other procs use ghost cells beyond +z limit

  if (slabflag == 1) {
    if (comm->myloc[2] == comm->procgrid[2]-1)
      nzhi_in = nzhi_out = nz_pppm - 1;
    nzhi_out = MIN(nzhi_out,nz_pppm-1);
  }

  // decomposition of FFT mesh
  // global indices range from 0 to N-1
  // proc owns entire x-dimension, clumps of columns in y,z dimensions
  // npey_fft,npez_fft = # of procs in y,z dims
  // if nprocs is small enough, proc can own 1 or more entire xy planes,
  //   else proc owns 2d sub-blocks of yz plane
  // me_y,me_z = which proc (0-npe_fft-1) I am in y,z dimensions
  // nlo_fft,nhi_fft = lower/upper limit of the section
  //   of the global FFT mesh that I own

  int npey_fft = -1, npez_fft = -1;
  if (nz_pppm >= nprocs) {
    npey_fft = 1;
    npez_fft = nprocs;
  } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

  int me_y = me % npey_fft;
  int me_z = me / npey_fft;

  nxlo_fft = 0;
  nxhi_fft = nx_pppm - 1;
  nylo_fft = me_y*ny_pppm/npey_fft;
  nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
  nzlo_fft = me_z*nz_pppm/npez_fft;
  nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;

  // PPPMOffcentre grid pts owned by this proc, including ghosts

  ngrid = (nxhi_out-nxlo_out+1) * (nyhi_out-nylo_out+1) *
    (nzhi_out-nzlo_out+1);

  // printf("PPPM offcentre: set_grid_local ngrid %i\n%i %i\n%i %i\n%i %i\n",
  // 	 ngrid,
  // 	 nxhi_out, nxlo_out,
  // 	 nyhi_out, nylo_out,
  // 	 nzhi_out, nzlo_out);

  // FFT grids owned by this proc, without ghosts
  // nfft = FFT points in FFT decomposition on this proc
  // nfft_brick = FFT points in 3d brick-decomposition on this proc
  // nfft_both = greater of 2 values

  nfft = (nxhi_fft-nxlo_fft+1) * (nyhi_fft-nylo_fft+1) *
    (nzhi_fft-nzlo_fft+1);
  int nfft_brick = (nxhi_in-nxlo_in+1) * (nyhi_in-nylo_in+1) *
    (nzhi_in-nzlo_in+1);
  nfft_both = MAX(nfft,nfft_brick);
}

/* ----------------------------------------------------------------------
   pre-compute Green's function denominator expansion coeffs, Gamma(2n)
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_gf_denom()
{
  int k,l,m;

  for (l = 1; l < order; l++) gf_b[l] = 0.0;
  gf_b[0] = 1.0;

  for (m = 1; m < order; m++) {
    for (l = m; l > 0; l--)
      gf_b[l] = 4.0 * (gf_b[l]*(l-m)*(l-m-0.5)-gf_b[l-1]*(l-m-1)*(l-m-1));
    gf_b[0] = 4.0 * (gf_b[0]*(l-m)*(l-m-0.5));
  }

  bigint ifact = 1;
  for (k = 1; k < 2*order; k++) ifact *= k;
  double gaminv = 1.0/ifact;
  for (l = 0; l < order; l++) gf_b[l] *= gaminv;
}

/* ----------------------------------------------------------------------
   pre-compute modified (Hockney-Eastwood) Coulomb Green's function
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_gf_ik()
{
  const double * const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double sum1,dot1,dot2;
  double numerator,denominator;
  double sqk;

  int k,l,m,n,nx,ny,nz,kper,lper,mper;

  const int nbx = static_cast<int> ((g_ewald*xprd/(MY_PI*nx_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nby = static_cast<int> ((g_ewald*yprd/(MY_PI*ny_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nbz = static_cast<int> ((g_ewald*zprd_slab/(MY_PI*nz_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int twoorder = 2*order;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    snz = square(sin(0.5*unitkz*mper*zprd_slab/nz_pppm));

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      sny = square(sin(0.5*unitky*lper*yprd/ny_pppm));

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        snx = square(sin(0.5*unitkx*kper*xprd/nx_pppm));

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

        if (sqk != 0.0) {
          numerator = 12.5663706/sqk;
          denominator = gf_denom(snx,sny,snz);
          sum1 = 0.0;

          for (nx = -nbx; nx <= nbx; nx++) {
            qx = unitkx*(kper+nx_pppm*nx);
            sx = exp(-0.25*square(qx/g_ewald));
            argx = 0.5*qx*xprd/nx_pppm;
            wx = powsinxx(argx,twoorder);

            for (ny = -nby; ny <= nby; ny++) {
              qy = unitky*(lper+ny_pppm*ny);
              sy = exp(-0.25*square(qy/g_ewald));
              argy = 0.5*qy*yprd/ny_pppm;
              wy = powsinxx(argy,twoorder);

              for (nz = -nbz; nz <= nbz; nz++) {
                qz = unitkz*(mper+nz_pppm*nz);
                sz = exp(-0.25*square(qz/g_ewald));
                argz = 0.5*qz*zprd_slab/nz_pppm;
                wz = powsinxx(argz,twoorder);

                dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                dot2 = qx*qx+qy*qy+qz*qz;
                sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
              }
            }
          }
          greensfn[n++] = numerator*sum1/denominator;
        } else greensfn[n++] = 0.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pre-compute modified (Hockney-Eastwood) Coulomb Green's function
   for a triclinic system
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_gf_ik_triclinic()
{
  double snx,sny,snz;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double sum1,dot1,dot2;
  double numerator,denominator;
  double sqk;

  int k,l,m,n,nx,ny,nz,kper,lper,mper;

  double tmp[3];
  tmp[0] = (g_ewald/(MY_PI*nx_pppm)) * pow(-log(EPS_HOC),0.25);
  tmp[1] = (g_ewald/(MY_PI*ny_pppm)) * pow(-log(EPS_HOC),0.25);
  tmp[2] = (g_ewald/(MY_PI*nz_pppm)) * pow(-log(EPS_HOC),0.25);
  lamda2xT(&tmp[0],&tmp[0]);
  const int nbx = static_cast<int> (tmp[0]);
  const int nby = static_cast<int> (tmp[1]);
  const int nbz = static_cast<int> (tmp[2]);

  const int twoorder = 2*order;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    snz = square(sin(MY_PI*mper/nz_pppm));

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      sny = square(sin(MY_PI*lper/ny_pppm));

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        snx = square(sin(MY_PI*kper/nx_pppm));

        double unitk_lamda[3];
        unitk_lamda[0] = 2.0*MY_PI*kper;
        unitk_lamda[1] = 2.0*MY_PI*lper;
        unitk_lamda[2] = 2.0*MY_PI*mper;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);

        sqk = square(unitk_lamda[0]) + square(unitk_lamda[1]) +
          square(unitk_lamda[2]);

        if (sqk != 0.0) {
          numerator = 12.5663706/sqk;
          denominator = gf_denom(snx,sny,snz);
          sum1 = 0.0;

          for (nx = -nbx; nx <= nbx; nx++) {
            argx = MY_PI*kper/nx_pppm + MY_PI*nx;
            wx = powsinxx(argx,twoorder);

            for (ny = -nby; ny <= nby; ny++) {
              argy = MY_PI*lper/ny_pppm + MY_PI*ny;
              wy = powsinxx(argy,twoorder);

              for (nz = -nbz; nz <= nbz; nz++) {
                argz = MY_PI*mper/nz_pppm + MY_PI*nz;
                wz = powsinxx(argz,twoorder);

                double b[3];
                b[0] = 2.0*MY_PI*nx_pppm*nx;
                b[1] = 2.0*MY_PI*ny_pppm*ny;
                b[2] = 2.0*MY_PI*nz_pppm*nz;
                x2lamdaT(&b[0],&b[0]);

                qx = unitk_lamda[0]+b[0];
                sx = exp(-0.25*square(qx/g_ewald));

                qy = unitk_lamda[1]+b[1];
                sy = exp(-0.25*square(qy/g_ewald));

                qz = unitk_lamda[2]+b[2];
                sz = exp(-0.25*square(qz/g_ewald));

                dot1 = unitk_lamda[0]*qx + unitk_lamda[1]*qy +
                  unitk_lamda[2]*qz;
                dot2 = qx*qx+qy*qy+qz*qz;
                sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
              }
            }
          }
          greensfn[n++] = numerator*sum1/denominator;
        } else greensfn[n++] = 0.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute optimized Green's function for energy calculation
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_gf_ad()
{
  const double * const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz,sqk;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double numerator,denominator;
  int k,l,m,n,kper,lper,mper;

  const int twoorder = 2*order;

  for (int i = 0; i < 6; i++) sf_coeff[i] = 0.0;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    qz = unitkz*mper;
    snz = square(sin(0.5*qz*zprd_slab/nz_pppm));
    sz = exp(-0.25*square(qz/g_ewald));
    argz = 0.5*qz*zprd_slab/nz_pppm;
    wz = powsinxx(argz,twoorder);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      qy = unitky*lper;
      sny = square(sin(0.5*qy*yprd/ny_pppm));
      sy = exp(-0.25*square(qy/g_ewald));
      argy = 0.5*qy*yprd/ny_pppm;
      wy = powsinxx(argy,twoorder);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        qx = unitkx*kper;
        snx = square(sin(0.5*qx*xprd/nx_pppm));
        sx = exp(-0.25*square(qx/g_ewald));
        argx = 0.5*qx*xprd/nx_pppm;
        wx = powsinxx(argx,twoorder);

        sqk = qx*qx + qy*qy + qz*qz;

        if (sqk != 0.0) {
          numerator = MY_4PI/sqk;
          denominator = gf_denom(snx,sny,snz);
          greensfn[n] = numerator*sx*sy*sz*wx*wy*wz/denominator;
          sf_coeff[0] += sf_precoeff1[n]*greensfn[n];
          sf_coeff[1] += sf_precoeff2[n]*greensfn[n];
          sf_coeff[2] += sf_precoeff3[n]*greensfn[n];
          sf_coeff[3] += sf_precoeff4[n]*greensfn[n];
          sf_coeff[4] += sf_precoeff5[n]*greensfn[n];
          sf_coeff[5] += sf_precoeff6[n]*greensfn[n];
          n++;
        } else {
          greensfn[n] = 0.0;
          sf_coeff[0] += sf_precoeff1[n]*greensfn[n];
          sf_coeff[1] += sf_precoeff2[n]*greensfn[n];
          sf_coeff[2] += sf_precoeff3[n]*greensfn[n];
          sf_coeff[3] += sf_precoeff4[n]*greensfn[n];
          sf_coeff[4] += sf_precoeff5[n]*greensfn[n];
          sf_coeff[5] += sf_precoeff6[n]*greensfn[n];
          n++;
        }
      }
    }
  }

  // compute the coefficients for the self-force correction

  double prex, prey, prez;
  prex = prey = prez = MY_PI/volume;
  prex *= nx_pppm/xprd;
  prey *= ny_pppm/yprd;
  prez *= nz_pppm/zprd_slab;
  sf_coeff[0] *= prex;
  sf_coeff[1] *= prex*2;
  sf_coeff[2] *= prey;
  sf_coeff[3] *= prey*2;
  sf_coeff[4] *= prez;
  sf_coeff[5] *= prez*2;

  // communicate values with other procs

  double tmp[6];
  MPI_Allreduce(sf_coeff,tmp,6,MPI_DOUBLE,MPI_SUM,world);
  for (n = 0; n < 6; n++) sf_coeff[n] = tmp[n];
}

/* ----------------------------------------------------------------------
   compute self force coefficients for ad-differentiation scheme
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_sf_precoeff()
{
  int i,k,l,m,n;
  int nx,ny,nz,kper,lper,mper;
  double wx0[5],wy0[5],wz0[5],wx1[5],wy1[5],wz1[5],wx2[5],wy2[5],wz2[5];
  double qx0,qy0,qz0,qx1,qy1,qz1,qx2,qy2,qz2;
  double u0,u1,u2,u3,u4,u5,u6;
  double sum1,sum2,sum3,sum4,sum5,sum6;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);

        sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = 0.0;
        for (i = 0; i < 5; i++) {

          qx0 = MY_2PI*(kper+nx_pppm*(i-2));
          qx1 = MY_2PI*(kper+nx_pppm*(i-1));
          qx2 = MY_2PI*(kper+nx_pppm*(i  ));
          wx0[i] = powsinxx(0.5*qx0/nx_pppm,order);
          wx1[i] = powsinxx(0.5*qx1/nx_pppm,order);
          wx2[i] = powsinxx(0.5*qx2/nx_pppm,order);

          qy0 = MY_2PI*(lper+ny_pppm*(i-2));
          qy1 = MY_2PI*(lper+ny_pppm*(i-1));
          qy2 = MY_2PI*(lper+ny_pppm*(i  ));
          wy0[i] = powsinxx(0.5*qy0/ny_pppm,order);
          wy1[i] = powsinxx(0.5*qy1/ny_pppm,order);
          wy2[i] = powsinxx(0.5*qy2/ny_pppm,order);

          qz0 = MY_2PI*(mper+nz_pppm*(i-2));
          qz1 = MY_2PI*(mper+nz_pppm*(i-1));
          qz2 = MY_2PI*(mper+nz_pppm*(i  ));

          wz0[i] = powsinxx(0.5*qz0/nz_pppm,order);
          wz1[i] = powsinxx(0.5*qz1/nz_pppm,order);
          wz2[i] = powsinxx(0.5*qz2/nz_pppm,order);
        }

        for (nx = 0; nx < 5; nx++) {
          for (ny = 0; ny < 5; ny++) {
            for (nz = 0; nz < 5; nz++) {
              u0 = wx0[nx]*wy0[ny]*wz0[nz];
              u1 = wx1[nx]*wy0[ny]*wz0[nz];
              u2 = wx2[nx]*wy0[ny]*wz0[nz];
              u3 = wx0[nx]*wy1[ny]*wz0[nz];
              u4 = wx0[nx]*wy2[ny]*wz0[nz];
              u5 = wx0[nx]*wy0[ny]*wz1[nz];
              u6 = wx0[nx]*wy0[ny]*wz2[nz];

              sum1 += u0*u1;
              sum2 += u0*u2;
              sum3 += u0*u3;
              sum4 += u0*u4;
              sum5 += u0*u5;
              sum6 += u0*u6;
            }
          }
        }

        // store values

        sf_precoeff1[n] = sum1;
        sf_precoeff2[n] = sum2;
        sf_precoeff3[n] = sum3;
        sf_precoeff4[n] = sum4;
        sf_precoeff5[n] = sum5;
        sf_precoeff6[n++] = sum6;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   find center grid pt for each of my particles
   check that full stencil for the particle will fit in my 3d brick
   store central grid pt indices in part2grid array
   ------------------------------------------------------------------------- */

void PPPMOffcentre::particle_map()
{
  int nx,ny,nz;

  double **x = atom->x;
  int *type = atom->type;
  int *tag = atom->tag;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int nlocal = atom->nlocal;

  int flag = 0;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  if (!isfinite(boxlo[0]) || !isfinite(boxlo[1]) || !isfinite(boxlo[2]))
    error->one(FLERR,"Non-numeric box dimensions - simulation unstable");

  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
      // current particle coord can be outside global and local box
      // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

      nx = static_cast<int> ((rsite[0]-boxlo[0])*delxinv+shift) - OFFSET;
      ny = static_cast<int> ((rsite[1]-boxlo[1])*delyinv+shift) - OFFSET;
      nz = static_cast<int> ((rsite[2]-boxlo[2])*delzinv+shift) - OFFSET;

      // if (triclinic != 0)
      // 	domain->lamda2x(rsite, rsite);

      // printf("PPPM offcentre %i: particle_map %f %f %f to %i %i %i of %f %f
      // %f\n", comm->me, rsite[0], rsite[1], rsite[2], nx, ny, nz, x[i][0],
      // x[i][1], x[i][2]);

      part2grid[i][s][0] = nx;
      part2grid[i][s][1] = ny;
      part2grid[i][s][2] = nz;

      // check that entire stencil around nx,ny,nz will fit in my 3d brick

      if ((nx+nlower < nxlo_out) || (nx+nupper > nxhi_out) ||
          (ny+nlower < nylo_out) || (ny+nupper > nyhi_out) ||
          (nz+nlower < nzlo_out) || (nz+nupper > nzhi_out)) {
        flag = 1;
        printf("PPPM offcentre (%2i): site at %g %g %g -> %2i %2i %2i\n"
               "PPPM offcentre (%2i): atom %i type %i\n"
               "PPPM offcentre (%2i): out of bounds x %2i < %2i%s%2i > %2i%s\n"
               "PPPM offcentre (%2i): out of bounds y %2i < %2i%s%2i > %2i%s\n"
               "PPPM offcentre (%2i): out of bounds z %2i < %2i%s%2i > %2i%s\n"
               "PPPM offcentre (%2i): with tolerance %i %i\n",
               comm->me,
               rsite[0], rsite[1], rsite[2], nx, ny, nz,
               comm->me,
               tag[i], itype,
               comm->me,
               nx+nlower, nxlo_out, (nx+nlower < nxlo_out) ? "<--" : "",
               nx+nupper, nxhi_out, (nx+nupper > nxhi_out) ? "<--" : "",
               comm->me,
               ny+nlower, nylo_out, (ny+nlower < nylo_out) ? "<--" : "",
               ny+nupper, nyhi_out, (ny+nupper > nyhi_out) ? "<--" : "",
               comm->me,
               nz+nlower, nzlo_out, (nz+nlower < nzlo_out) ? "<--" : "",
               nz+nupper, nzhi_out, (nz+nupper > nzhi_out) ? "<--" : "",
               comm->me,
               nlower,	nupper);
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);

  if (flag)
    error->one(FLERR,"Out of range atoms - cannot compute PPPMOffcentre");
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid
   ------------------------------------------------------------------------- */

void PPPMOffcentre::make_rho()
{
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density array

  memset(&(density_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  int *type = atom->type;
  double **x = atom->x;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int nlocal = atom->nlocal;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }	

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      nx = part2grid[i][s][0];
      ny = part2grid[i][s][1];
      nz = part2grid[i][s][2];
      dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
      dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
      dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

      compute_rho1d(dx,dy,dz);

      z0 = delvolinv * molFrameCharge[itype][s];
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        y0 = z0*rho1d[2][n];
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          x0 = y0*rho1d[1][m];
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            density_brick[mz][my][mx] += x0*rho1d[0][l];
          }
        }
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}

/* ----------------------------------------------------------------------
   remap density from 3d brick decomposition to FFT decomposition
   ------------------------------------------------------------------------- */

void PPPMOffcentre::brick2fft()
{
  int n,ix,iy,iz;

  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  for (iz = nzlo_in; iz <= nzhi_in; iz++)
    for (iy = nylo_in; iy <= nyhi_in; iy++)
      for (ix = nxlo_in; ix <= nxhi_in; ix++)
        density_fft[n++] = density_brick[iz][iy][ix];

  remap->perform(density_fft,density_fft,work1);
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver
   ------------------------------------------------------------------------- */

void PPPMOffcentre::poisson()
{
  if (differentiation_flag == 1) poisson_ad();
  else poisson_ik();
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
   ------------------------------------------------------------------------- */

void PPPMOffcentre::poisson_ik()
{
  int i,j,k,n;
  double eng;

  // transform charge density (r -> k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nfft; i++) {
        eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
        if (eflag_global) energy += eng;
        n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < nfft; i++) {
        energy +=
          s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        n += 2;
      }
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  // extra FFTs for per-atom energy/virial

  if (evflag_atom) poisson_peratom();

  // triclinic system

  if (triclinic) {
    poisson_ik_triclinic();
    return;
  }

  // compute gradients of V(r) in each of 3 dims by transformimg -ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = fkx[i]*work1[n+1];
        work2[n+1] = -fkx[i]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdx_brick[k][j][i] = work2[n];
        n += 2;
      }

  // y direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = fky[j]*work1[n+1];
        work2[n+1] = -fky[j]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdy_brick[k][j][i] = work2[n];
        n += 2;
      }

  // z direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = fkz[k]*work1[n+1];
        work2[n+1] = -fkz[k]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik for a triclinic system
   ------------------------------------------------------------------------- */

void PPPMOffcentre::poisson_ik_triclinic()
{
  int i,j,k,n;

  // compute gradients of V(r) in each of 3 dims by transformimg -ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x direction gradient

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = fkx[i]*work1[n+1];
    work2[n+1] = -fkx[i]*work1[n];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdx_brick[k][j][i] = work2[n];
        n += 2;
      }

  // y direction gradient

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = fky[i]*work1[n+1];
    work2[n+1] = -fky[i]*work1[n];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdy_brick[k][j][i] = work2[n];
        n += 2;
      }

  // z direction gradient

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = fkz[i]*work1[n+1];
    work2[n+1] = -fkz[i]*work1[n];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ad
   ------------------------------------------------------------------------- */

void PPPMOffcentre::poisson_ad()
{
  int i,j,k,n;
  double eng;

  // transform charge density (r -> k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nfft; i++) {
        eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
        if (eflag_global) energy += eng;
        n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < nfft; i++) {
        energy +=
          s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        n += 2;
      }
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  // extra FFTs for per-atom energy/virial

  if (vflag_atom) poisson_peratom();

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n];
    work2[n+1] = work1[n+1];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        u_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for per-atom energy/virial
   ------------------------------------------------------------------------- */

void PPPMOffcentre::poisson_peratom()
{
  int i,j,k,n;

  // energy

  if (eflag_atom && differentiation_flag != 1) {
    n = 0;
    for (i = 0; i < nfft; i++) {
      work2[n] = work1[n];
      work2[n+1] = work1[n+1];
      n += 2;
    }

    fft2->compute(work2,work2,-1);

    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
      for (j = nylo_in; j <= nyhi_in; j++)
        for (i = nxlo_in; i <= nxhi_in; i++) {
          u_brick[k][j][i] = work2[n];
          n += 2;
        }
  }

  // 6 components of virial in v0 thru v5

  if (!vflag_atom) return;

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n]*vg[i][0];
    work2[n+1] = work1[n+1]*vg[i][0];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        v0_brick[k][j][i] = work2[n];
        n += 2;
      }

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n]*vg[i][1];
    work2[n+1] = work1[n+1]*vg[i][1];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        v1_brick[k][j][i] = work2[n];
        n += 2;
      }

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n]*vg[i][2];
    work2[n+1] = work1[n+1]*vg[i][2];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        v2_brick[k][j][i] = work2[n];
        n += 2;
      }

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n]*vg[i][3];
    work2[n+1] = work1[n+1]*vg[i][3];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        v3_brick[k][j][i] = work2[n];
        n += 2;
      }

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n]*vg[i][4];
    work2[n+1] = work1[n+1]*vg[i][4];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        v4_brick[k][j][i] = work2[n];
        n += 2;
      }

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n]*vg[i][5];
    work2[n+1] = work1[n+1]*vg[i][5];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        v5_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles
   ------------------------------------------------------------------------- */

void PPPMOffcentre::fieldforce()
{
  if (differentiation_flag == 1) fieldforce_ad();
  else fieldforce_ik();
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
   ------------------------------------------------------------------------- */

void PPPMOffcentre::fieldforce_ik()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx,eky,ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  int nlocal = atom->nlocal;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      nx = part2grid[i][s][0];
      ny = part2grid[i][s][1];
      nz = part2grid[i][s][2];
      dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
      dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
      dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

      compute_rho1d(dx,dy,dz);

      ekx = eky = ekz = ZEROF;
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        z0 = rho1d[2][n];
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          y0 = z0*rho1d[1][m];
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            x0 = y0*rho1d[0][l];
            ekx -= x0*vdx_brick[mz][my][mx];
            eky -= x0*vdy_brick[mz][my][mx];
            ekz -= x0*vdz_brick[mz][my][mx];
          }
        }
      }

      // convert E-field to force
      const double qfactor = qqrd2e * scale * molFrameCharge[itype][s];
      double force[3] = { qfactor*ekx,
        qfactor*eky,
        0.0 };

      if (slabflag != 2)
        force[2] = qfactor*ekz;

      f[i][0] += force[0];
      f[i][1] += force[1];

      if (slabflag != 2) {
        f[i][2] += force[2];
      }

      double torque[3];
      MathExtra::cross3(labFrameSite, force, torque);
      tor[i][0] += torque[0];
      tor[i][1] += torque[1];
      tor[i][2] += torque[2];

    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ad
   ------------------------------------------------------------------------- */

void PPPMOffcentre::fieldforce_ad()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz;
  FFT_SCALAR ekx,eky,ekz;
  double s1,s2,s3;
  double sf = 0.0;
  double *prd;

  prd = domain->prd;
  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  double hx_inv = nx_pppm/xprd;
  double hy_inv = ny_pppm/yprd;
  double hz_inv = nz_pppm/zprd;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  int nlocal = atom->nlocal;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      nx = part2grid[i][s][0];
      ny = part2grid[i][s][1];
      nz = part2grid[i][s][2];
      dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
      dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
      dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

      compute_rho1d(dx,dy,dz);
      compute_drho1d(dx,dy,dz);

      ekx = eky = ekz = ZEROF;
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            ekx += drho1d[0][l]*rho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
            eky += rho1d[0][l]*drho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
            ekz += rho1d[0][l]*rho1d[1][m]*drho1d[2][n]*u_brick[mz][my][mx];
          }
        }
      }
      ekx *= hx_inv;
      eky *= hy_inv;
      ekz *= hz_inv;

      // convert E-field to force and substract self forces

      const double qfactor = qqrd2e * scale;
      double force[3] = { 0.0, 0.0, 0.0 };
      double torque[3] = { 0.0, 0.0, 0.0 };

      s1 = rsite[0]*hx_inv;
      s2 = rsite[1]*hy_inv;
      s3 = rsite[2]*hz_inv;
      sf = sf_coeff[0]*sin(2*MY_PI*s1);
      sf += sf_coeff[1]*sin(4*MY_PI*s1);
      sf *= 2*molFrameCharge[itype][s]*molFrameCharge[itype][s];
      force[0] = qfactor*(ekx*molFrameCharge[itype][s] - sf);
      f[i][0] += force[0];

      sf = sf_coeff[2]*sin(2*MY_PI*s2);
      sf += sf_coeff[3]*sin(4*MY_PI*s2);
      sf *= 2*molFrameCharge[itype][s]*molFrameCharge[itype][s];
      force[1] = qfactor*(eky*molFrameCharge[itype][s] - sf);
      f[i][1] += force[1];

      sf = sf_coeff[4]*sin(2*MY_PI*s3);
      sf += sf_coeff[5]*sin(4*MY_PI*s3);
      sf *= 2*molFrameCharge[itype][s]*molFrameCharge[itype][s];
      if (slabflag != 2) {
        force[2] = qfactor*(ekz*molFrameCharge[itype][s] - sf);
        f[i][2] += force[2];
      }

      MathExtra::cross3(labFrameSite, force, torque);
      tor[i][0] += torque[0];
      tor[i][1] += torque[1];
      tor[i][2] += torque[2];
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}

/* ----------------------------------------------------------------------
   interpolate from grid to get per-atom energy/virial
   ------------------------------------------------------------------------- */

void PPPMOffcentre::fieldforce_peratom()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR u,v0,v1,v2,v3,v4,v5;

  // loop over my charges, interpolate from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  double **x = atom->x;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  int nlocal = atom->nlocal;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }	

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      // if triclinic all expressed in lamda coords at this point
      if (triclinic != 0)
        domain->x2lamda(rsite, rsite);

      nx = part2grid[i][s][0];
      ny = part2grid[i][s][1];
      nz = part2grid[i][s][2];
      dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
      dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
      dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

      compute_rho1d(dx,dy,dz);

      u = v0 = v1 = v2 = v3 = v4 = v5 = ZEROF;
      for (n = nlower; n <= nupper; n++) {
        mz = n+nz;
        z0 = rho1d[2][n];
        for (m = nlower; m <= nupper; m++) {
          my = m+ny;
          y0 = z0*rho1d[1][m];
          for (l = nlower; l <= nupper; l++) {
            mx = l+nx;
            x0 = y0*rho1d[0][l];
            if (eflag_atom) u += x0*u_brick[mz][my][mx];
            if (vflag_atom) {
              v0 += x0*v0_brick[mz][my][mx];
              v1 += x0*v1_brick[mz][my][mx];
              v2 += x0*v2_brick[mz][my][mx];
              v3 += x0*v3_brick[mz][my][mx];
              v4 += x0*v4_brick[mz][my][mx];
              v5 += x0*v5_brick[mz][my][mx];
            }
          }
        }
      }

      if (eflag_atom) eatom[i] += molFrameCharge[itype][s]*u;
      if (vflag_atom) {
        vatom[i][0] += molFrameCharge[itype][s]*v0;
        vatom[i][1] += molFrameCharge[itype][s]*v1;
        vatom[i][2] += molFrameCharge[itype][s]*v2;
        vatom[i][3] += molFrameCharge[itype][s]*v3;
        vatom[i][4] += molFrameCharge[itype][s]*v4;
        vatom[i][5] += molFrameCharge[itype][s]*v5;
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}

/* ----------------------------------------------------------------------
   pack own values to buf to send to another proc
   ------------------------------------------------------------------------- */

void PPPMOffcentre::pack_forward(int flag, FFT_SCALAR *buf, int nlist, int
                                 *list)
{
  int n = 0;

  if (flag == FORWARD_IK) {
    FFT_SCALAR *xsrc = &vdx_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *ysrc = &vdy_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *zsrc = &vdz_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = xsrc[list[i]];
      buf[n++] = ysrc[list[i]];
      buf[n++] = zsrc[list[i]];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *src = &u_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  } else if (flag == FORWARD_IK_PERATOM) {
    FFT_SCALAR *esrc = &u_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      if (eflag_atom) buf[n++] = esrc[list[i]];
      if (vflag_atom) {
        buf[n++] = v0src[list[i]];
        buf[n++] = v1src[list[i]];
        buf[n++] = v2src[list[i]];
        buf[n++] = v3src[list[i]];
        buf[n++] = v4src[list[i]];
        buf[n++] = v5src[list[i]];
      }
    }
  } else if (flag == FORWARD_AD_PERATOM) {
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = v0src[list[i]];
      buf[n++] = v1src[list[i]];
      buf[n++] = v2src[list[i]];
      buf[n++] = v3src[list[i]];
      buf[n++] = v4src[list[i]];
      buf[n++] = v5src[list[i]];
    }
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's own values from buf and set own ghost values
   ------------------------------------------------------------------------- */

void PPPMOffcentre::unpack_forward(int flag, FFT_SCALAR *buf, int nlist,
                                   int *list)
{
  int n = 0;

  if (flag == FORWARD_IK) {
    FFT_SCALAR *xdest = &vdx_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *ydest = &vdy_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *zdest = &vdz_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      xdest[list[i]] = buf[n++];
      ydest[list[i]] = buf[n++];
      zdest[list[i]] = buf[n++];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *dest = &u_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] = buf[i];
  } else if (flag == FORWARD_IK_PERATOM) {
    FFT_SCALAR *esrc = &u_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      if (eflag_atom) esrc[list[i]] = buf[n++];
      if (vflag_atom) {
        v0src[list[i]] = buf[n++];
        v1src[list[i]] = buf[n++];
        v2src[list[i]] = buf[n++];
        v3src[list[i]] = buf[n++];
        v4src[list[i]] = buf[n++];
        v5src[list[i]] = buf[n++];
      }
    }
  } else if (flag == FORWARD_AD_PERATOM) {
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      v0src[list[i]] = buf[n++];
      v1src[list[i]] = buf[n++];
      v2src[list[i]] = buf[n++];
      v3src[list[i]] = buf[n++];
      v4src[list[i]] = buf[n++];
      v5src[list[i]] = buf[n++];
    }
  }
}

/* ----------------------------------------------------------------------
   pack ghost values into buf to send to another proc
   ------------------------------------------------------------------------- */

void PPPMOffcentre::pack_reverse(int flag, FFT_SCALAR *buf, int nlist,
                                 int *list)
{
  if (flag == REVERSE_RHO) {
    FFT_SCALAR *src = &density_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's ghost values from buf and add to own values
   ------------------------------------------------------------------------- */

void PPPMOffcentre::unpack_reverse(int flag, FFT_SCALAR *buf, int nlist,
                                   int *list)
{
  if (flag == REVERSE_RHO) {
    FFT_SCALAR *dest = &density_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] += buf[i];
  }
}

/* ----------------------------------------------------------------------
   map nprocs to NX by NY grid as PX by PY procs - return optimal px,py
   ------------------------------------------------------------------------- */

void PPPMOffcentre::procs2grid2d(int nprocs, int nx, int ny, int *px, int *py)
{
  // loop thru all possible factorizations of nprocs
  // surf = surface area of largest proc sub-domain
  // innermost if test minimizes surface area and surface/volume ratio

  int bestsurf = 2 * (nx + ny);
  int bestboxx = 0;
  int bestboxy = 0;

  int boxx,boxy,surf,ipx,ipy;

  ipx = 1;
  while (ipx <= nprocs) {
    if (nprocs % ipx == 0) {
      ipy = nprocs/ipx;
      boxx = nx/ipx;
      if (nx % ipx) boxx++;
      boxy = ny/ipy;
      if (ny % ipy) boxy++;
      surf = boxx + boxy;
      if (surf < bestsurf ||
          (surf == bestsurf && boxx*boxy > bestboxx*bestboxy)) {
        bestsurf = surf;
        bestboxx = boxx;
        bestboxy = boxy;
        *px = ipx;
        *py = ipy;
      }
    }
    ipx++;
  }
}

/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_rho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy,
                                  const FFT_SCALAR &dz)
{
  int k,l;
  FFT_SCALAR r1,r2,r3;

  for (k = (1-order)/2; k <= order/2; k++) {
    r1 = r2 = r3 = ZEROF;

    for (l = order-1; l >= 0; l--) {
      r1 = rho_coeff[l][k] + r1*dx;
      r2 = rho_coeff[l][k] + r2*dy;
      r3 = rho_coeff[l][k] + r3*dz;
    }
    rho1d[0][k] = r1;
    rho1d[1][k] = r2;
    rho1d[2][k] = r3;
  }
}

/* ----------------------------------------------------------------------
   charge assignment into drho1d
   dx,dy,dz = distance of particle from "lower left" grid point
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_drho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy,
                                   const FFT_SCALAR &dz)
{
  int k,l;
  FFT_SCALAR r1,r2,r3;

  for (k = (1-order)/2; k <= order/2; k++) {
    r1 = r2 = r3 = ZEROF;

    for (l = order-2; l >= 0; l--) {
      r1 = drho_coeff[l][k] + r1*dx;
      r2 = drho_coeff[l][k] + r2*dy;
      r3 = drho_coeff[l][k] + r3*dz;
    }
    drho1d[0][k] = r1;
    drho1d[1][k] = r2;
    drho1d[2][k] = r3;
  }
}

/* ----------------------------------------------------------------------
   generate coeffients for the weight function of order n

   (n-1)
   Wn(x) =     Sum    wn(k,x) , Sum is over every other integer
   k=-(n-1)
   For k=-(n-1),-(n-1)+2, ....., (n-1)-2,n-1
   k is odd integers if n is even and even integers if n is odd
   ---
   | n-1
   | Sum a(l,j)*(x-k/2)**l   if abs(x-k/2) < 1/2
   wn(k,x) = <  l=0
   |
   |  0                       otherwise
   ---
   a coeffients are packed into the array rho_coeff to eliminate zeros
   rho_coeff(l,((k+mod(n+1,2))/2) = a(l,k)
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_rho_coeff()
{
  int j,k,l,m;
  FFT_SCALAR s;

  FFT_SCALAR **a;
  memory->create2d_offset(a,order,-order,order,"pppmoffcentre:a");

  for (k = -order; k <= order; k++)
    for (l = 0; l < order; l++)
      a[l][k] = 0.0;

  a[0][0] = 1.0;
  for (j = 1; j < order; j++) {
    for (k = -j; k <= j; k += 2) {
      s = 0.0;
      for (l = 0; l < j; l++) {
        a[l+1][k] = (a[l][k+1]-a[l][k-1]) / (l+1);
#ifdef FFT_SINGLE
        s += powf(0.5,(float) l+1) *
          (a[l][k-1] + powf(-1.0,(float) l) * a[l][k+1]) / (l+1);
#else
        s += pow(0.5,(double) l+1) *
          (a[l][k-1] + pow(-1.0,(double) l) * a[l][k+1]) / (l+1);
#endif
      }
      a[0][k] = s;
    }
  }

  m = (1-order)/2;
  for (k = -(order-1); k < order; k += 2) {
    for (l = 0; l < order; l++)
      rho_coeff[l][m] = a[l][k];
    for (l = 1; l < order; l++)
      drho_coeff[l-1][m] = l*a[l][k];
    m++;
  }

  memory->destroy2d_offset(a,-order);
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
   ------------------------------------------------------------------------- */

void PPPMOffcentre::slabcorr()
{
  // compute local contribution to global dipole moment

  double **x = atom->x;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;
  double **tor = atom->torque;

  double dipole = 0.0;
  for (int i = 0; i < nlocal; i++) {
    int itype = type[i];
    double rotMat[3][3];
    if (nsites[itype] > 0) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat(iquat, rotMat);
    }

    for (int s = 1; s <= nsites[itype]; ++s) {
      double labFrameSite[3] = {0.0, 0.0, 0.0};
      if (molFrameSite[itype][s][0] != 0.0 ||
          molFrameSite[itype][s][1] != 0.0 ||
          molFrameSite[itype][s][2] != 0.0) {
        double ms[3] = {
          molFrameSite[itype][s][0],
          molFrameSite[itype][s][1],
          molFrameSite[itype][s][2]
        };

        MathExtra::matvec(rotMat, ms, labFrameSite);
      }

      double rsite[3] = {
        labFrameSite[0]+x[i][0],
        labFrameSite[1]+x[i][1],
        labFrameSite[2]+x[i][2]
      };

      //
      dipole += molFrameCharge[itype][s]*rsite[2];
    }
  }

  // sum local contributions to get global dipole moment

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  double dipole_r2 = 0.0;
  if (eflag_atom || fabs(qsum) > SMALL) {
    for (int i = 0; i < nlocal; i++) {
      int itype = type[i];
      double rotMat[3][3];
      if (nsites[itype] > 0) {
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat(iquat, rotMat);
      }

      for (int s = 1; s <= nsites[itype]; ++s) {
        double labFrameSite[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[itype][s][0] != 0.0 ||
            molFrameSite[itype][s][1] != 0.0 ||
            molFrameSite[itype][s][2] != 0.0) {
          double ms[3] = {
            molFrameSite[itype][s][0],
            molFrameSite[itype][s][1],
            molFrameSite[itype][s][2]
          };

          MathExtra::matvec(rotMat, ms, labFrameSite);
        }

        double rsite[3] = {
          labFrameSite[0]+x[i][0],
          labFrameSite[1]+x[i][1],
          labFrameSite[2]+x[i][2]
        };

        dipole += molFrameCharge[itype][s]*rsite[2]*rsite[2];
      }

      // sum local contributions

      double tmp;
      MPI_Allreduce(&dipole_r2,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
      dipole_r2 = tmp;
    }

    // compute corrections

    const double e_slabcorr = MY_2PI*(dipole_all*dipole_all - qsum*dipole_r2 -
                                      qsum*qsum*zprd*zprd/12.0)/volume;
    const double qscale = qqrd2e * scale;

    if (eflag_global) energy += qscale * e_slabcorr;

    // per-atom energy

    if (eflag_atom) {
      double efact = qscale * MY_2PI/volume;
      for (int i = 0; i < nlocal; i++) {
        int itype = type[i];
        double rotMat[3][3];
        if (nsites[itype] > 0) {
          iquat = bonus[ellipsoid[i]].quat;
          MathExtra::quat_to_mat(iquat, rotMat);
        }

        for (int s = 1; s <= nsites[itype]; ++s) {
          double labFrameSite[3] = {0.0, 0.0, 0.0};
          if (molFrameSite[itype][s][0] != 0.0 ||
              molFrameSite[itype][s][1] != 0.0 ||
              molFrameSite[itype][s][2] != 0.0) {
            double ms[3] = {
              molFrameSite[itype][s][0],
              molFrameSite[itype][s][1],
              molFrameSite[itype][s][2]
            };

            MathExtra::matvec(rotMat, ms, labFrameSite);
          }

          double rsite[3] = {
            labFrameSite[0]+x[i][0],
            labFrameSite[1]+x[i][1],
            labFrameSite[2]+x[i][2]
          };

          eatom[i] +=
            efact * molFrameCharge[itype][s]*(rsite[2]*dipole_all -
                                              0.5*(dipole_r2 +
                                                   qsum*rsite[2]*rsite[2]) -
                                              qsum*zprd*zprd/12.0); } } }

    // add on force corrections

    double ffact = qscale * (-4.0*MY_PI/volume);
    double **f = atom->f;

    for (int i = 0; i < nlocal; i++) {
      int itype = type[i];
      double rotMat[3][3];
      if (nsites[itype] > 0) {
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat(iquat, rotMat);
      }

      for (int s = 1; s <= nsites[itype]; ++s) {
        double labFrameSite[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[itype][s][0] != 0.0 ||
            molFrameSite[itype][s][1] != 0.0 ||
            molFrameSite[itype][s][2] != 0.0) {
          double ms[3] = {
            molFrameSite[itype][s][0],
            molFrameSite[itype][s][1],
            molFrameSite[itype][s][2]
          };

          MathExtra::matvec(rotMat, ms, labFrameSite);
        }

        double rsite[3] = {
          labFrameSite[0]+x[i][0],
          labFrameSite[1]+x[i][1],
          labFrameSite[2]+x[i][2]
        };

        double force[3] = {0.0, 0.0, 0.0};
        force[2] = ffact * molFrameCharge[itype][s]*(dipole_all -
                                                     qsum*rsite[2]);
        f[i][2] += force[2];

        // and torque?

        double torque[3];
        MathExtra::cross3(labFrameSite, force, torque);
        tor[i][0] += torque[0];
        tor[i][1] += torque[1];
        tor[i][2] += torque[2];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   perform and time the 1d FFTs required for N timesteps
   ------------------------------------------------------------------------- */

int PPPMOffcentre::timing_1d(int n, double &time1d)
{
  double time1,time2;

  for (int i = 0; i < 2*nfft_both; i++) work1[i] = ZEROF;

  MPI_Barrier(world);
  time1 = MPI_Wtime();

  for (int i = 0; i < n; i++) {
    fft1->timing1d(work1,nfft_both,1);
    fft2->timing1d(work1,nfft_both,-1);
    if (differentiation_flag != 1) {
      fft2->timing1d(work1,nfft_both,-1);
      fft2->timing1d(work1,nfft_both,-1);
    }
  }

  MPI_Barrier(world);
  time2 = MPI_Wtime();
  time1d = time2 - time1;

  if (differentiation_flag) return 2;
  return 4;
}

/* ----------------------------------------------------------------------
   perform and time the 3d FFTs required for N timesteps
   ------------------------------------------------------------------------- */

int PPPMOffcentre::timing_3d(int n, double &time3d)
{
  double time1,time2;

  for (int i = 0; i < 2*nfft_both; i++) work1[i] = ZEROF;

  MPI_Barrier(world);
  time1 = MPI_Wtime();

  for (int i = 0; i < n; i++) {
    fft1->compute(work1,work1,1);
    fft2->compute(work1,work1,-1);
    if (differentiation_flag != 1) {
      fft2->compute(work1,work1,-1);
      fft2->compute(work1,work1,-1);
    }
  }

  MPI_Barrier(world);
  time2 = MPI_Wtime();
  time3d = time2 - time1;

  if (differentiation_flag) return 2;
  return 4;
}

/* ----------------------------------------------------------------------
   memory usage of local arrays
   ------------------------------------------------------------------------- */

double PPPMOffcentre::memory_usage()
{
  double bytes = nmax*3 * sizeof(double);
  int nbrick = (nxhi_out-nxlo_out+1) * (nyhi_out-nylo_out+1) *
    (nzhi_out-nzlo_out+1);
  if (differentiation_flag == 1) {
    bytes += 2 * nbrick * sizeof(FFT_SCALAR);
  } else {
    bytes += 4 * nbrick * sizeof(FFT_SCALAR);
  }
  if (triclinic) bytes += 3 * nfft_both * sizeof(double);
  bytes += 6 * nfft_both * sizeof(double);
  bytes += nfft_both * sizeof(double);
  bytes += nfft_both*5 * sizeof(FFT_SCALAR);

  if (peratom_allocate_flag)
    bytes += 6 * nbrick * sizeof(FFT_SCALAR);

  if (group_allocate_flag) {
    bytes += 2 * nbrick * sizeof(FFT_SCALAR);
    bytes += 2 * nfft_both * sizeof(FFT_SCALAR);;
  }

  if (cg) bytes += cg->memory_usage();

  return bytes;
}

/* ----------------------------------------------------------------------
   group-group interactions
   ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   compute the PPPMOffcentre total long-range force and energy for groups A and
   B
   ------------------------------------------------------------------------- */

void PPPMOffcentre::compute_group_group(int groupbit_A,
                                        int groupbit_B,
                                        int AA_flag)
{
  if (slabflag && triclinic)
    error->all(FLERR,"Cannot (yet) use K-space slab "
               "correction with compute group/group for triclinic systems");

  if (differentiation_flag)
    error->all(FLERR,"Cannot (yet) use kspace_modify "
               "diff ad with compute group/group");

  if (!group_allocate_flag) allocate_groups();

  // convert atoms from box to lamda coords

  //MATTEO probably should do something here, convert also offcntre positions 

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }

  e2group = 0.0; //energy
  f2group[0] = 0.0; //force in x-direction
  f2group[1] = 0.0; //force in y-direction
  f2group[2] = 0.0; //force in z-direction

  // map my particle charge onto my local 3d density grid

  make_rho_groups(groupbit_A,groupbit_B,AA_flag);

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  // temporarily store and switch pointers so we can
  //  use brick2fft() for groups A and B (without
  //  writing an additional function)

  FFT_SCALAR ***density_brick_real = density_brick;
  FFT_SCALAR *density_fft_real = density_fft;

  // group A

  density_brick = density_A_brick;
  density_fft = density_A_fft;

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  // group B

  density_brick = density_B_brick;
  density_fft = density_B_fft;

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  // switch back pointers

  density_brick = density_brick_real;
  density_fft = density_fft_real;

  // compute potential gradient on my FFT grid and
  //   portion of group-group energy/force on this proc's FFT grid

  poisson_groups(AA_flag);

  const double qscale = qqrd2e * scale;

  // total group A <--> group B energy
  // self and boundary correction terms are in compute_group_group.cpp

  double e2group_all;
  MPI_Allreduce(&e2group,&e2group_all,1,MPI_DOUBLE,MPI_SUM,world);
  e2group = e2group_all;

  e2group *= qscale*0.5*volume;

  // total group A <--> group B force

  double f2group_all[3];
  MPI_Allreduce(f2group,f2group_all,3,MPI_DOUBLE,MPI_SUM,world);

  f2group[0] = qscale*volume*f2group_all[0];
  f2group[1] = qscale*volume*f2group_all[1];
  if (slabflag != 2) f2group[2] = qscale*volume*f2group_all[2];

  // convert atoms back from lamda to box coords

  //MATTEO probably should do something here, convert also offcntre positions 

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  if (slabflag == 1)
    slabcorr_groups(groupbit_A, groupbit_B, AA_flag);
}

/* ----------------------------------------------------------------------
   allocate group-group memory that depends on # of K-vectors and order
   ------------------------------------------------------------------------- */

void PPPMOffcentre::allocate_groups()
{
  group_allocate_flag = 1;

  memory->create3d_offset(density_A_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:density_A_brick");
  memory->create3d_offset(density_B_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"pppmoffcentre:density_B_brick");
  memory->create(density_A_fft,nfft_both,"pppmoffcentre:density_A_fft");
  memory->create(density_B_fft,nfft_both,"pppmoffcentre:density_B_fft");
}

/* ----------------------------------------------------------------------
   deallocate group-group memory that depends on # of K-vectors and order
   ------------------------------------------------------------------------- */

void PPPMOffcentre::deallocate_groups()
{
  group_allocate_flag = 0;

  memory->destroy3d_offset(density_A_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(density_B_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy(density_A_fft);
  memory->destroy(density_B_fft);
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid for group-group interactions
   ------------------------------------------------------------------------- */

void PPPMOffcentre::make_rho_groups(int groupbit_A,
                                    int groupbit_B,
                                    int AA_flag)
{
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density arrays

  memset(&(density_A_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  memset(&(density_B_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  double **x = atom->x;
  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  if (triclinic != 0) domain->lamda2x(atom->nlocal);

  for (int i = 0; i < nlocal; i++) {

    if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
      if (AA_flag) continue;

    if ((mask[i] & groupbit_A) || (mask[i] & groupbit_B)) {
      int itype = type[i];
      double rotMat[3][3];
      if (nsites[itype] > 0) {
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat(iquat, rotMat);
      }

      for (int s = 1; s <= nsites[itype]; ++s) {
        double labFrameSite[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[itype][s][0] != 0.0 ||
            molFrameSite[itype][s][1] != 0.0 ||
            molFrameSite[itype][s][2] != 0.0) {
          double ms[3] = {
            molFrameSite[itype][s][0],
            molFrameSite[itype][s][1],
            molFrameSite[itype][s][2]
          };

          MathExtra::matvec(rotMat, ms, labFrameSite);
        }

        double rsite[3] = {
          labFrameSite[0]+x[i][0],
          labFrameSite[1]+x[i][1],
          labFrameSite[2]+x[i][2]
        };

        // if triclinic all expressed in lamda coords at this point
        if (triclinic != 0)
          domain->x2lamda(rsite, rsite);

        nx = part2grid[i][s][0];
        ny = part2grid[i][s][1];
        nz = part2grid[i][s][2];
        dx = nx+shiftone - (rsite[0]-boxlo[0])*delxinv;
        dy = ny+shiftone - (rsite[1]-boxlo[1])*delyinv;
        dz = nz+shiftone - (rsite[2]-boxlo[2])*delzinv;

        compute_rho1d(dx,dy,dz);

        z0 = delvolinv * molFrameCharge[itype][s];
        for (n = nlower; n <= nupper; n++) {
          mz = n+nz;
          y0 = z0*rho1d[2][n];
          for (m = nlower; m <= nupper; m++) {
            my = m+ny;
            x0 = y0*rho1d[1][m];
            for (l = nlower; l <= nupper; l++) {
              mx = l+nx;

              // group A

              if (mask[i] & groupbit_A)
                density_A_brick[mz][my][mx] += x0*rho1d[0][l];

              // group B

              if (mask[i] & groupbit_B)
                density_B_brick[mz][my][mx] += x0*rho1d[0][l];
            }
          }
        }
      }
    }
  }

  if (triclinic != 0) domain->x2lamda(atom->nlocal);
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for group-group interactions
   ------------------------------------------------------------------------- */

void PPPMOffcentre::poisson_groups(int AA_flag)
{
  int i,j,k,n;

  // reuse memory (already declared)

  FFT_SCALAR *work_A = work1;
  FFT_SCALAR *work_B = work2;

  // transform charge density (r -> k)

  // group A

  n = 0;
  for (i = 0; i < nfft; i++) {
    work_A[n++] = density_A_fft[i];
    work_A[n++] = ZEROF;
  }

  fft1->compute(work_A,work_A,1);

  // group B

  n = 0;
  for (i = 0; i < nfft; i++) {
    work_B[n++] = density_B_fft[i];
    work_B[n++] = ZEROF;
  }

  fft1->compute(work_B,work_B,1);

  // group-group energy and force contribution,
  //  keep everything in reciprocal space so
  //  no inverse FFTs needed

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  // energy

  n = 0;
  for (i = 0; i < nfft; i++) {
    e2group += s2 * greensfn[i] *
      (work_A[n]*work_B[n] + work_A[n+1]*work_B[n+1]);
    n += 2;
  }

  if (AA_flag) return;


  // multiply by Green's function and s2
  //  (only for work_A so it is not squared below)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work_A[n++] *= s2 * greensfn[i];
    work_A[n++] *= s2 * greensfn[i];
  }

  // triclinic system

  if (triclinic) {
    poisson_groups_triclinic();
    return;
  }

  double partial_group;

  // force, x direction

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        partial_group = work_A[n+1]*work_B[n] - work_A[n]*work_B[n+1];
        f2group[0] += fkx[i] * partial_group;
        n += 2;
      }

  // force, y direction

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        partial_group = work_A[n+1]*work_B[n] - work_A[n]*work_B[n+1];
        f2group[1] += fky[j] * partial_group;
        n += 2;
      }

  // force, z direction

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        partial_group = work_A[n+1]*work_B[n] - work_A[n]*work_B[n+1];
        f2group[2] += fkz[k] * partial_group;
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for group-group interactions
   for a triclinic system
   ------------------------------------------------------------------------- */

void PPPMOffcentre::poisson_groups_triclinic()
{
  int i,n;

  // reuse memory (already declared)

  FFT_SCALAR *work_A = work1;
  FFT_SCALAR *work_B = work2;

  double partial_group;

  // force, x direction

  n = 0;
  for (i = 0; i < nfft; i++) {
    partial_group = work_A[n+1]*work_B[n] - work_A[n]*work_B[n+1];
    f2group[0] += fkx[i] * partial_group;
    n += 2;
  }

  // force, y direction

  n = 0;
  for (i = 0; i < nfft; i++) {
    partial_group = work_A[n+1]*work_B[n] - work_A[n]*work_B[n+1];
    f2group[1] += fky[i] * partial_group;
    n += 2;
  }

  // force, z direction

  n = 0;
  for (i = 0; i < nfft; i++) {
    partial_group = work_A[n+1]*work_B[n] - work_A[n]*work_B[n+1];
    f2group[2] += fkz[i] * partial_group;
    n += 2;
  }
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
   ------------------------------------------------------------------------- */

void PPPMOffcentre::slabcorr_groups(int groupbit_A,
                                    int groupbit_B,
                                    int AA_flag)
{
  // compute local contribution to global dipole moment

  int *type = atom->type;
  double *iquat;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  double **x = atom->x;
  double zprd = domain->zprd;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double qsum_A = 0.0;
  double qsum_B = 0.0;
  double dipole_A = 0.0;
  double dipole_B = 0.0;
  double dipole_r2_A = 0.0;
  double dipole_r2_B = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
      if (AA_flag) continue;

    if (mask[i] & groupbit_A) {
      int itype = type[i];
      double rotMat[3][3];
      if (nsites[itype] > 0) {
        iquat = bonus[ellipsoid[i]].quat;
        MathExtra::quat_to_mat(iquat, rotMat);
      }

      for (int s = 1; s <= nsites[itype]; ++s) {
        double labFrameSite[3] = {0.0, 0.0, 0.0};
        if (molFrameSite[itype][s][0] != 0.0 ||
            molFrameSite[itype][s][1] != 0.0 ||
            molFrameSite[itype][s][2] != 0.0) {
          double ms[3] = {
            molFrameSite[itype][s][0],
            molFrameSite[itype][s][1],
            molFrameSite[itype][s][2]
          };

          MathExtra::matvec(rotMat, ms, labFrameSite);
        }

        double rsite[3] = {
          labFrameSite[0]+x[i][0],
          labFrameSite[1]+x[i][1],
          labFrameSite[2]+x[i][2]
        };

        qsum_A += molFrameCharge[itype][s];
        dipole_A += molFrameCharge[itype][s]*rsite[2];
        dipole_r2_A += molFrameCharge[itype][s]*rsite[2]*rsite[2];

        if (mask[i] & groupbit_B) {
          qsum_B += molFrameCharge[itype][s];
          dipole_B += molFrameCharge[itype][s]*rsite[2];
          dipole_r2_B += molFrameCharge[itype][s]*rsite[2]*rsite[2];
        }
      }
    }
  }

  // sum local contributions to get total charge and global dipole moment
  //  for each group

  double tmp;
  MPI_Allreduce(&qsum_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_B = tmp;

  MPI_Allreduce(&dipole_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_A = tmp;

  MPI_Allreduce(&dipole_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_B = tmp;

  MPI_Allreduce(&dipole_r2_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_A = tmp;

  MPI_Allreduce(&dipole_r2_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_B = tmp;

  // compute corrections

  const double qscale = qqrd2e * scale;
  const double efact = qscale * MY_2PI/volume;

  e2group += efact * (dipole_A*dipole_B - 0.5*(qsum_A*dipole_r2_B +
                                               qsum_B*dipole_r2_A)
                      - qsum_A*qsum_B*zprd*zprd/12.0);

  // add on force corrections

  const double ffact = qscale * (-4.0*MY_PI/volume);
  f2group[2] += ffact * (qsum_A*dipole_B - qsum_B*dipole_A);
}

