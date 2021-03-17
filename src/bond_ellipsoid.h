/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   -------------------------------------------------------------------------*/

#ifdef BOND_CLASS

BondStyle(ellipsoid,BondEllipsoid)

#else

#ifndef LMP_BOND_ELLIPSOID_H
#define LMP_BOND_ELLIPSOID_H

#include <stdio.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <math.h>
#include "bond_molc.h"
#include "math_extra.h"
#include "math_const.h"

namespace LAMMPS_NS {
 
  class BondEllipsoid : public BondMolc {
  public:
    BondEllipsoid(class LAMMPS *);
    virtual ~BondEllipsoid();
    virtual void compute(int, int);
    void settings(int, char **);
    void coeff(int, char **);
    void init_style();
    double equilibrium_distance(int);
    void write_restart(FILE *);
    void read_restart(FILE *);
    void write_data(FILE *);
    double single(int, double, int, int, double &);

    std::vector<double>
    force1_numeric(int btype, int atom1, int atom2, double delta_move);
    std::vector<double>
    force2_numeric(int btype, int atom1, int atom2, double delta_move);
    std::vector<double>
    torque1_numeric(int btype, int atom1, int atom2, double delta_move);
    std::vector<double>
    torque2_numeric(int btype, int atom1, int atom2, double delta_move);
    
    class CouplerBondPotential {
    public:
      virtual std::string name() const = 0;
      virtual bool contribute_to_virial() const = 0;      
      virtual std::vector<double> force1(double rnorm, double fbond,
					 double* v1, double* v2) = 0;
      virtual std::vector<double> force2(double rnorm, double fbond,
					 double* v1, double* v2) = 0;
      virtual std::vector<double> torque1(double fbond, double* v1,
					  double* v2) = 0;
      virtual std::vector<double> torque2(double fbond, double* v1,
					  double* v2) = 0;
    };

    class BondPotential {
    public:
      BondPotential(std::string n,
		    int r,
		    CouplerBondPotential *c) : name(n),
					       rank(r),
					       coupler(c)
	{}
    
      virtual int maxparams() const = 0;
      virtual int nparams() const = 0;
      bool contribute_to_virial() const {
	return coupler->contribute_to_virial();
      }      

      std::vector<double>& get_params() { return params; }
      std::string get_name() {
	std::stringstream ss;
	ss << rank;
	return name + ss.str() + coupler->name();
      }
      virtual std::pair<double, double> compute(double x) = 0;
      virtual CouplerBondPotential *potential() { 
	return coupler;
      }

    protected:
      std::string name;
      int rank;
      std::vector<double> params;
      CouplerBondPotential *coupler;
    };

    class CouplerRRBondPotential : public CouplerBondPotential {
    public:
      std::string name() const { return std::string("RR"); } 
      bool contribute_to_virial() const { return true; }
      std::vector<double> force1(double rnorm,
				 double fbond,
				 double* v1,
				 double* v2) {
	std::vector<double> f(3);
	f[0] = -fbond*v1[0];
	f[1] = -fbond*v1[1];
	f[2] = -fbond*v1[2];
	return f;
      }
      std::vector<double> force2(double rnorm,
				 double fbond,
				 double* v1,
				 double* v2) {
	std::vector<double> f(3);
	f[0] = fbond*v2[0];
	f[1] = fbond*v2[1];
	f[2] = fbond*v2[2];
	return f;
      }
      std::vector<double> torque1(double fbond, double* v1, double* v2) {
	return std::vector<double>(3, 0.0);
      }
      std::vector<double> torque2(double fbond, double* v1, double* v2) {
	return std::vector<double>(3, 0.0);
      }
    };
    
    class CouplerXXBondPotential : public CouplerBondPotential {
    public:
      std::string name() const { return std::string("XX"); } 
      bool contribute_to_virial() const { return false; }
      std::vector<double> force1(double rnorm, double fbond,
				 double* v1, double* v2) {
	return std::vector<double>(3, 0.0);
      }
      std::vector<double> force2(double rnorm, double fbond,
				 double* v1, double* v2) {
	return std::vector<double>(3, 0.0);
      }
      std::vector<double> torque1(double fbond, double* v1, double* v2) {
	std::vector<double> t(3);
	double n[3];
	MathExtra::cross3(v1, v2, n);
	t[0] = fbond*n[0];
	t[1] = fbond*n[1];
	t[2] = fbond*n[2];
	return t;
      }
      std::vector<double> torque2(double fbond, double* v1, double* v2) {
	std::vector<double> t(3);
	double n[3];
	MathExtra::cross3(v2, v1, n);
	t[0] = fbond*n[0];
	t[1] = fbond*n[1];
	t[2] = fbond*n[2];
	return t;
      }
    };

    class CouplerXRBondPotential : public CouplerBondPotential {
    public:
      std::string name() const { return std::string("XR"); } 
      bool contribute_to_virial() const { return true; } // true but must use different formulas 
      std::vector<double> force1(double rnorm, double fbond,
				 double* x, double* r) {
	double n[3];
	MathExtra::cross3(x, r, n);

	std::vector<double> f(3);
	double ff[3];
	MathExtra::cross3(n, r, ff);
	f[0] = fbond*ff[0]/rnorm;
	f[1] = fbond*ff[1]/rnorm;
	f[2] = fbond*ff[2]/rnorm;
	return f;
      }
      std::vector<double> force2(double rnorm, double fbond,
				 double* x, double* r) {
	double n[3];
	MathExtra::cross3(x, r, n);

	std::vector<double> f(3);
	double ff[3];
	MathExtra::cross3(n, r, ff);
	f[0] = -fbond*ff[0]/rnorm;
	f[1] = -fbond*ff[1]/rnorm;
	f[2] = -fbond*ff[2]/rnorm;
	return f;
      }
      std::vector<double> torque1(double fbond, double* x, double* r) {
	std::vector<double> t(3);
	double n[3];
	MathExtra::cross3(x, r, n);
	t[0] = fbond*n[0];
	t[1] = fbond*n[1];
	t[2] = fbond*n[2];
	return t;
      }
      std::vector<double> torque2(double fbond, double* x, double* r) {
	return std::vector<double>(3, 0.0);
      }
    };

    class CouplerRXBondPotential : public CouplerBondPotential {
    public:
      std::string name() const { return std::string("RX"); } // true but must use different formulas 
      bool contribute_to_virial() const { return true; }
      std::vector<double> force1(double rnorm, double fbond,
				 double* r, double* x) {
	double n[3];
	MathExtra::cross3(x, r, n);

	std::vector<double> f(3);
	double ff[3];
	MathExtra::cross3(n, r, ff);
	f[0] = fbond*ff[0]/rnorm;
	f[1] = fbond*ff[1]/rnorm;
	f[2] = fbond*ff[2]/rnorm;
	return f;
      }

      std::vector<double> force2(double rnorm, double fbond,
				 double* r, double* x) {
	double n[3];
	MathExtra::cross3(x, r, n);

	std::vector<double> f(3);
	double ff[3];
	MathExtra::cross3(n, r, ff);
	f[0] = -fbond*ff[0]/rnorm;
	f[1] = -fbond*ff[1]/rnorm;
	f[2] = -fbond*ff[2]/rnorm;
	return f;
      }
      std::vector<double> torque1(double fbond, double* r, double* x) {
	return std::vector<double>(3, 0.0);
      }
      std::vector<double> torque2(double fbond, double* r, double* x) {
	std::vector<double> t(3);
	double n[3];
	MathExtra::cross3(r, x, n);
	t[0] = -fbond*n[0];
	t[1] = -fbond*n[1];
	t[2] = -fbond*n[2];
	return t;
      }
    };
    
    class PolynomialBondPotential : public BondPotential {
    public:
      PolynomialBondPotential(std::string n,
			      int r,
			      CouplerBondPotential * c) : BondPotential(n, r, c)
	{}
    
      int nparams() const {
	return rank*2+1;
      }

      int maxparams() const {
	return rank*2+1;
      }
    
      std::pair<double, double> compute(double x) {
      	int const p_rank = 2;

      	double grad = 0.0;
      	double energy = params[0];
      	for (int i=0; i<rank; ++i) {
	  double p1 = params[i*p_rank+1];
	  double p2 = params[i*p_rank+2];

	  double pow1 = p1*pow(x-p2, (double)(i));
	  //double pow1 = p1*pow(x-p2, (double)(i+1));
	  
      	  energy += pow1*(x-p2);

      	  grad += (double)(i+1)*pow1;

      	  // energy +=
      	  //   p1*pow(x-p2, (double)(i+2))
      	  //   +
      	  //   p3*pow(x-p4, (double)(i+2));

      	  // grad +=
      	  //   (double)(i+2)*p1*pow(x-p2, (double)(i+2-1))
      	  //   +
      	  //   (double)(i+2)*p3*pow(x-p4, (double)(i+2-1));
      	}

      	return std::make_pair(energy, grad);
      }
    };  
    
    class ExponentialBondPotential : public BondPotential {
    public:

      ExponentialBondPotential(std::string n,
			       int r,
			       CouplerBondPotential * c)
	: BondPotential(n, r, c)
	{}
    
      int nparams() const {
	return 6;
      }

      int maxparams() const {
	return 6;
      }

      std::pair<double, double> compute(double x) {
	double exp1 = exp(-0.5*params[1]*pow(x-params[2], 2.0));
	double exp2 = params[3]*exp(-0.5*params[4]*pow(x-params[5], 2.0));
	double energy = params[0] - log(exp1 + exp2);

	if (isnan(energy)) {
	  printf("exponential_fun nan @ %f\n", x);
	}

	double dexp1 = -params[1]*(x-params[2])*exp1;
	double dexp2 = -params[3]*params[4]*(x-params[5])*exp2;

	double grad = -1.0/(exp1 + exp2)*(dexp1 + dexp2);

    	return std::make_pair(energy, grad);	
      }
    };
  
    class CosinesBondPotential : public BondPotential {
    public:

      CosinesBondPotential(std::string n,
			   int r,
			   CouplerBondPotential * c)
	: BondPotential(n, r, c)
	{}
    
      int nparams() const {
	return rank*2+1;
      }

      int maxparams() const {
	return rank*2+1;
      }

      std::pair<double, double> compute(double x) {
	int const p_rank = 2;

	double grad = 0.0;
	double energy = params[0];
	for (int i=0; i<rank; ++i) {
	  double p1 = params[i*p_rank+1];
	  double p2 = params[i*p_rank+2];
	  
	  energy += p1*(1.0-cos(MathConst::MY_PI*(i+1)*x-p2));
	  grad   += p1*(i+1)*MathConst::MY_PI*sin(MathConst::MY_PI*(i+1)*x-p2);
	}

    	return std::make_pair(energy, grad);
      }
    };

    class CosinesHalfBondPotential : public BondPotential {
    public:

      CosinesHalfBondPotential(std::string n,
			       int r,
			       CouplerBondPotential * c)
	: BondPotential(n, r, c)
	{}
    
      int nparams() const {
	return rank*2+1;
      }

      int maxparams() const {
	return rank*2+1;
      }

      std::pair<double, double> compute(double x) {
	int const p_rank = 2;

	double grad = 0.0;
	double energy = params[0];
	for (int i=0; i<rank; ++i) {
	  double p1 = params[i*p_rank+1];
	  double p2 = params[i*p_rank+2];
	  
	  energy += p1*(1.0-cos(0.5*MathConst::MY_PI*(i+1)*x-p2));
	  grad   += p1*(i+1)*MathConst::MY_PI*0.5*sin(0.5*MathConst::MY_PI*(i+1)*x-p2);
	}

    	return std::make_pair(energy, grad);
      }
    };
    
    class FourierBondPotential : public BondPotential {
    public:

      FourierBondPotential(std::string n,
			   int r,
			   CouplerBondPotential * c) : BondPotential(n, r, c)
	{}
    
      int nparams() const {
	return rank*3+1;
      }

      int maxparams() const {
	return rank*3+1;
      }

      std::pair<double, double> compute(double x) {
	int const p_rank = 3;

	double grad = 0.0;
	double energy = 0.5*params[0];

	for (int i=0; i<rank; ++i) {
	  double p1 = params[i*p_rank+1];
	  double p2 = params[i*p_rank+2];
	  double p3 = params[i*p_rank+3];

	  if (p3 == 0)
	    p3 = 10.0e-9;
        
	  double arg  = (i+1)*MathConst::MY_PI/p3;
	  double valc = cos(arg*x);
	  double vals = sin(arg*x);
	  // double valc = p1*cos(i*MathConst::MY_PI*x/p3);
	  // double vals = p2*sin(i*MathConst::MY_PI*x/p3);

	  energy += p1*valc + p2*vals;
	  grad   += -arg*p1*vals + arg*p2*valc;
	}
	
    	return std::make_pair(energy, grad);
      }
    };

    class FourierAngBondPotential : public BondPotential {
    public:

      FourierAngBondPotential(std::string n,
			      int r,
			      CouplerBondPotential * c) : BondPotential(n, r, c)
	{}
    
      int nparams() const {
	return rank*2+1;
      }

      int maxparams() const {
	return rank*2+1;
      }

      std::pair<double, double> compute(double x) {
	int const p_rank = 2;

	double grad = 0.0;
	double energy = 0.5*params[0];

	for (int i=0; i<rank; ++i) {
	  double p1 = params[i*p_rank+1];
	  double p2 = params[i*p_rank+2];
        
	  double arg  = (i+1)*MathConst::MY_PI;
	  double valc = cos(arg*x);
	  double vals = sin(arg*x);

	  energy += p1*valc + p2*vals;
	  grad   += -arg*p1*vals + arg*p2*valc;
	}
	
    	return std::make_pair(energy, grad);
      }
    };
    
    class ChebyshevBondPotential : public BondPotential {
    public:
      ChebyshevBondPotential(std::string n,
			     int r,
			     CouplerBondPotential * c) :
	BondPotential(n, r, c)
	{}
    
      int nparams() const {
	return rank;
      }

      int maxparams() const {
	return 12;
      }

      std::pair<double, double> compute(double x) {
      	double energy = 0.0;

	energy += (rank > 0  ? params[0] : 0.0)*1.0;
	energy += (rank > 1  ? params[1] : 0.0)*x;
	energy += (rank > 2  ? params[2] : 0.0)*(2.0*pow(x,2)-1.0);
	energy += (rank > 3  ? params[3] : 0.0)*(4.0*pow(x,3)-3.0*x);
	energy += (rank > 4  ? params[4] : 0.0)*(8.0*pow(x,4)-8.0*pow(x,2)+1.0);
	energy += (rank > 5  ? params[5] : 0.0)*(16.0*pow(x,5)-20.0*pow(x,3)+5.0*x);
	energy += (rank > 6  ? params[6] : 0.0)*(32.0*pow(x,6)-48.0*pow(x,4)+18.0*pow(x,2)-1.0);
	energy += (rank > 7  ? params[7] : 0.0)*(64.0*pow(x,7)-112.0*pow(x,5)+56.0*pow(x,3)-7.0*x);
	energy += (rank > 8  ? params[8] : 0.0)*(128.0*pow(x,8)-256.0*pow(x,6)+160.0*pow(x,4)-32.0*pow(x,2)+1.0);
	energy += (rank > 9  ? params[9] : 0.0)*(256.0*pow(x,9)-576.0*pow(x,7)+432.0*pow(x,5)-120.0*pow(x,3)+9.0*x);
	energy += (rank > 10 ? params[10]: 0.0)*(512.0*pow(x,10)-1280.0*pow(x,8)+1120.0*pow(x,6)-400.0*pow(x,4)+50.0*pow(x,2)-1.0);
	energy += (rank > 11 ? params[11]: 0.0)*(1024.0*pow(x,11)-2816.0*pow(x,9)+2816.0*pow(x,7)-1232.0*pow(x,5)+220.0*pow(x,3)-11.0*x);

      	double grad = 0.0;
	
	grad += (rank > 1  ? params[1] : 0.0);
	grad += (rank > 2  ? params[2] : 0.0)*4.0*x;
	grad += (rank > 3  ? params[3] : 0.0)*(12.0*pow(x,2)-3.0);
	grad += (rank > 4  ? params[4] : 0.0)*(32.0*pow(x,3)-16.0*x);
	grad += (rank > 5  ? params[5] : 0.0)*(80.0*pow(x,4)-60.0*pow(x,2)+5.0);
	grad += (rank > 6  ? params[6] : 0.0)*(192.0*pow(x,5)-192.0*pow(x,3)+32.0*x);
	grad += (rank > 7  ? params[7] : 0.0)*(448.0*pow(x,6)-560.0*pow(x,4)+168.0*pow(x,2)-7.0);
	grad += (rank > 8  ? params[8] : 0.0)*(1024.0*pow(x,7)-1536.0*pow(x,5)+640.0*pow(x,3)-64.0*x);
	grad += (rank > 9  ? params[9] : 0.0)*(2304.0*pow(x,8)-4032.0*pow(x,6)+2160.0*pow(x,4)-360.0*pow(x,2)+9.0);
	grad += (rank > 10 ? params[10]: 0.0)*(5120.0*pow(x,9)-10240.0*pow(x,7)+6720.0*pow(x,5)-1600.0*pow(x,3)+100.0*x);
	grad += (rank > 11 ? params[11]: 0.0)*(11264.0*pow(x,10)-25344.0*pow(x,8)+19712.0*pow(x,6)-6160.0*pow(x,4)+660.0*pow(x,2)-11.0);

	return std::make_pair(energy, grad);
      }
    };

    class HermiteBondPotential : public BondPotential {
    public:
      HermiteBondPotential(std::string n,
			   int r,
			   CouplerBondPotential * c) :
	BondPotential(n, r, c)
	{}
    
      int nparams() const {
	return rank;
      }

      int maxparams() const {
	return 11;
      }
    
      std::pair<double, double> compute(double x) {
      	double energy = 0.0;

	x -= params[0];

	bool const with_measure = false;
	double factor = 1.0;

	if (with_measure)
	  factor = exp(-0.5*x*x);

	energy += (rank > 0 ? factor*params[1] : 0.0)*1.0;
	energy += (rank > 1 ? factor*params[2] : 0.0)*2.0*x;
	energy += (rank > 2 ? factor*params[3] : 0.0)*(4.0*pow(x,2)-2.0);
	energy += (rank > 3 ? factor*params[4] : 0.0)*(8.0*pow(x,3)-12.0*x);
	energy += (rank > 4 ? factor*params[5] : 0.0)*(16.0*pow(x,4)-48.0*pow(x,2)+12.0);
	energy += (rank > 5 ? factor*params[6] : 0.0)*(32.0*pow(x,5)-160.0*pow(x,3)+120.0*x);
	energy += (rank > 6 ? factor*params[7] : 0.0)*(64.0*pow(x,6)-480.0*pow(x,4)+720.0*pow(x,2)-120.0);
	energy += (rank > 7 ? factor*params[8] : 0.0)*(128.0*pow(x,7)-1344.0*pow(x,5)+3360.0*pow(x,3)-1680.0*x);
	energy += (rank > 8 ? factor*params[9] : 0.0)*(256.0*pow(x,8)-3584.0*pow(x,6)+13440.0*pow(x,4)-13440.0*pow(x,2)+1680.0);
	energy += (rank > 9 ? factor*params[10] : 0.0)*(512.0*pow(x,9)-9216.0*pow(x,7)+48384.0*pow(x,5)-80640.0*pow(x,3)+30240.0*x);
	energy += (rank > 10? factor*params[11]: 0.0)*(1024.0*pow(x,10)-23040.0*pow(x,8)+161280.0*pow(x,6)-403200.0*pow(x,4)+302400.0*pow(x,2)-30240.0);

      	double grad = 0.0;

	if (with_measure)
	  factor = -x*exp(-0.5*x*x);

	grad += (rank > 1 ? factor*params[2] : 0.0)*2.0;
	grad += (rank > 2 ? factor*params[3] : 0.0)*8.0*x;
	grad += (rank > 3 ? factor*params[4] : 0.0)*(24.0*pow(x,2)-12.0);
	grad += (rank > 4 ? factor*params[5] : 0.0)*(64.0*pow(x,3)-96.0*x);
	grad += (rank > 5 ? factor*params[6] : 0.0)*(160.0*pow(x,4)-480.0*pow(x,2)+120.0);
	grad += (rank > 6 ? factor*params[7] : 0.0)*(384.0*pow(x,5)-25920.0*pow(x,3)+1440.0*x);
	grad += (rank > 7 ? factor*params[8] : 0.0)*(896.0*pow(x,6)-6720.0*pow(x,4)+10080.0*pow(x,2)-1680.0);
	grad += (rank > 8 ? factor*params[9] : 0.0)*(2048.0*pow(x,7)-21504.0*pow(x,5)+53760.0*pow(x,3)-26880.0*x);
	grad += (rank > 9 ? factor*params[10] : 0.0)*(4608.0*pow(x,8)-64512.0*pow(x,6)+241920.0*pow(x,4)-241920.0*pow(x,2)+30240.0);
	grad += (rank > 10? factor*params[11]: 0.0)*(10240.0*pow(x,9)-184320.0*pow(x,7)+967680.0*pow(x,5)-1612800.0*pow(x,3)+3225600.0*x);

	return std::make_pair(energy, grad);
      }
    };
    
    class BondPotentialFactory {
    public:
      BondPotentialFactory() {
	if (allowed.size() > 0 || coupler.size() > 0) {
	  exit(1);
	}
	
	allowed.push_back("polynomial");
	allowed.push_back("fourier");
	allowed.push_back("cosines");
	allowed.push_back("cosines_half");
	allowed.push_back("exponential");
	allowed.push_back("chebyshev");
	allowed.push_back("hermite");
	allowed.push_back("fourier_ang");

	coupler.push_back(new CouplerRRBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXXBondPotential);
	coupler.push_back(new CouplerXRBondPotential);
	coupler.push_back(new CouplerXRBondPotential);
	coupler.push_back(new CouplerXRBondPotential);
	coupler.push_back(new CouplerRXBondPotential);
	coupler.push_back(new CouplerRXBondPotential);
	coupler.push_back(new CouplerRXBondPotential);
      }

      bool find(std::string name) {
	return std::find(allowed.begin(), allowed.end(), name)
	  != allowed.end();
      }

      BondPotential* create(std::string type, int rank, int index) {
	if (type == std::string("polynomial"))
	  return new PolynomialBondPotential("polynomial", rank, coupler[index]);
	else if (type == std::string("cosines"))
	  return new CosinesBondPotential("cosines", rank, coupler[index]);
	else if (type == std::string("cosines_half"))
	  return new CosinesHalfBondPotential("cosines_half", rank, coupler[index]);
	else if (type == std::string("fourier"))
	  return new FourierBondPotential("fourier", rank,
					  coupler[index]);
	else if (type == std::string("fourier_ang"))
	  return new FourierAngBondPotential("fourier_ang", rank,
					     coupler[index]);
	else if (type == std::string("exponential"))
	  return new ExponentialBondPotential("exponential", rank,
					      coupler[index]);
	else if (type == std::string("chebyshev"))
	  return new ChebyshevBondPotential("chebyshev", rank,
					    coupler[index]);
	else if (type == std::string("hermite"))
	  return new HermiteBondPotential("hermite", rank,
					  coupler[index]);
	else { printf("%s not know. Abort\n", type.c_str()); exit(1); }
      }

      std::vector<std::string> allowed;
      std::vector<CouplerBondPotential*> coupler;
    };
  
  protected:
    BondPotentialFactory factory;
    std::vector<std::vector<BondPotential*> > bondPotSet;

    static int const pterms = 16;
    class AtomVecEllipsoid *avec;
    
    void allocate();
    char* hfilename(char const*);
    
  private:    
    //  FILE* hdebug_coo;
    FILE* hdebug_val;
    FILE* hdebug_ene;
    FILE* hdebug_gra;
    FILE* hdebug_num;
    bool verbose;
    bool numeric_gradients;
    bool use_numeric_gradients;
    double delta_move;
  };
}

#endif
#endif
