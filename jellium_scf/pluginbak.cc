/*
 * @BEGIN LICENSE
 *
 * myscf by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2017 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/molecule.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libqt/qt.h"
#include <math.h>
#include "psi4/pragma.h"
#include"JelliumIntegrals.h"

#ifdef _OPENMP
    #include<omp.h>
#else
    #define omp_get_wtime() ( (double)clock() / CLOCKS_PER_SEC )
    #define omp_get_max_threads() 1
#endif

namespace psi{ namespace jellium_scf {

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "jellium_scf"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
        /*- The number of grid points for the Gauss-Legendre quadrature -*/
        options.add_int("N_GRID_POINTS", 10);
        /*- The number of electrons -*/
        options.add_int("N_ELECTRONS", 2);
        /*- The number of basis functions -*/
        options.add_int("N_BASIS_FUNCTIONS", 26);
        /*- The length of the box -*/
        options.add_double("LENGTH", M_PI);
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction jellium_scf(SharedWavefunction ref_wfn, Options& options)
{
    // number of basis functions
    int nso = options.get_int("N_BASIS_FUNCTIONS"); 

    // number of electrons
    int nelectron = options.get_int("N_ELECTRONS");
    if ( nelectron % 2 != 0 ) {
        throw PsiException("jellium only works with even number of electrons (for now).",__FILE__,__LINE__);
    }
   

    // factor for box size ... coded to give <rho> = 1
    double Lfac = pow((double)nelectron,1.0/3.0)/M_PI;

    //grabbing one-electon integrals from mintshelper
    std::shared_ptr<JelliumIntegrals> Jell (new JelliumIntegrals(options));

    //one-electron kinetic energy integrals
    std::shared_ptr<Matrix> T = Jell->Ke;
    T->scale(1.0/Lfac/Lfac);

    //one-electron potential energy integrals
    std::shared_ptr<Matrix> V = Jell->NucAttrac;
    V->scale(1.0/Lfac);

    //build the core hamiltonian
    std::shared_ptr<Matrix> h = (std::shared_ptr<Matrix>)(new Matrix(T));
    std::shared_ptr<Matrix> Ca = (std::shared_ptr<Matrix>)(new Matrix(nso,nso));
    
    // build core hamiltonian
    V->scale(nelectron); 
    h->add(V);

    int na = nelectron / 2;

    // fock matrix
    std::shared_ptr<Matrix> F = (std::shared_ptr<Matrix>)(new Matrix(h));

    // eigenvectors / eigenvalues of fock matrix
    std::shared_ptr<Vector> Feval (new Vector(nso));

    // diagonalize core hamiltonian, get orbitals
    F->diagonalize(Ca,Feval);

    // build density matrix core hamiltonian
    std::shared_ptr<Matrix> D (new Matrix(nso,nso));
    double ** d_p = D->pointer();
    //#pragma omp parallel for
    for(int mu = 0; mu < nso; ++mu){
        for(int nu = 0; nu< nso; ++nu){
            double dum = 0.0;
            for(int i = 0; i < na; ++i){
                dum += Ca->pointer()[mu][i] * Ca->pointer()[nu][i];
            }
            d_p[mu][nu] = dum;
        }
    }

    double energy = D->vector_dot(h) + D->vector_dot(F);
    outfile->Printf("    initial energy: %20.12lf\n",energy);
    outfile->Printf("\n");

    // containers for J and K 
    std::shared_ptr<Matrix> J (new Matrix(nso,nso));
    std::shared_ptr<Matrix> K (new Matrix(nso,nso));

    double ** j_p = J->pointer();
    double ** k_p = K->pointer();

    // convergence parameters
    double e_convergence = options.get_double("E_CONVERGENCE");
    double d_convergence = options.get_double("D_CONVERGENCE");
    int    maxiter       = options.get_int("MAXITER");

    int iter = 0;
    outfile->Printf("    ");
    outfile->Printf("  iter");
    outfile->Printf("              energy");
    outfile->Printf("                |dE|");
    outfile->Printf("                |dD|\n");

    double dele = 0.0;
    double deld = 0.0;
    
    do {
    
        //#pragma omp parallel for
        for(int mu = 0; mu < nso; ++mu){
            for(int nu = 0; nu < nso; ++nu){
                double myJ = 0.0;
                double myK = 0.0;
                for(int lambda = 0; lambda < nso; ++lambda){
                    for(int sigma = 0; sigma < nso; ++sigma){
                        double d = d_p[lambda][sigma];
                        myJ += d * Jell->ERI_int(mu,nu,lambda,sigma);
                        myK += d * Jell->ERI_int(mu,sigma,lambda,nu);
                    }
                }
                j_p[mu][nu] = myJ / Lfac;
                k_p[mu][nu] = myK / Lfac;
           }
        }
        std::shared_ptr<Matrix> Fa (new Matrix(J));
        Fa->scale(2.0);
        Fa->subtract(K);
        Fa->add(h);

        F->copy(Fa);
        double new_energy = 0.0;
        new_energy += (nelectron*nelectron/2.0)*Jell->selfval/Lfac;
        new_energy += D->vector_dot(h);
        new_energy += D->vector_dot(Fa);

        std::shared_ptr<Matrix> Fprime = (std::shared_ptr<Matrix>)(new Matrix(F));
        std::shared_ptr<Matrix> Fevec (new Matrix(nso,nso));
        std::shared_ptr<Vector> Feval (new Vector(nso));
        Fprime->diagonalize(Ca,Feval);

        //building density matrix

        std::shared_ptr<Matrix> Dnew (new Matrix(nso,nso));
        double ** dnew_p = Dnew->pointer();
        double tmp = 0;
        //#pragma omp parallel for
        for(int mu = 0; mu < nso; ++mu){
            for(int nu = 0; nu < nso; ++nu){
                for(int i = 0; i < na; ++i){
                    dnew_p[mu][nu] += Ca->pointer()[mu][i] * Ca->pointer()[nu][i];
                }
            }
        }

        deld = 0.0;
        for(int mu = 0; mu < nso; ++mu){
            for(int nu = 0; nu < nso; ++nu){
                double dum = d_p[mu][nu] - dnew_p[mu][nu];
                deld += dum * dum;
            }
        }
        deld = sqrt(deld);

        dele = fabs(new_energy-energy);

        outfile->Printf("    %6i%20.12lf%20.12lf%20.12lf\n", iter, new_energy, dele, deld);
        energy = new_energy;
        D->copy(Dnew);

        iter++;
        if( iter > maxiter ) break;

    }while(dele > e_convergence || deld > d_convergence);

    if ( iter > maxiter ) {
        throw PsiException("jellium scf did not converge.",__FILE__,__LINE__);
    }

    outfile->Printf("\n");
    outfile->Printf("      SCF iterations converged!\n");
    outfile->Printf("\n");

    double fock_energy = D->vector_dot(K);
    //V->print();
    outfile->Printf("    * Jellium HF total energy: %20.12lf\n",energy);
    outfile->Printf("      Fock energy:             %20.12lf\n",fock_energy);
    return ref_wfn;

    // Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}

}} // End namespaces

