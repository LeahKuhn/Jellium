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
    std::shared_ptr<Matrix> Firstevec (new Matrix(nso,nso));
    std::shared_ptr<Vector> Firsteval (new Vector(nso));

    // diagonalize core hamiltonian, get orbitals
    std::shared_ptr<Matrix> Cafirst (new Matrix(nso,nso));
    F->diagonalize(Cafirst,Firsteval);

    // build density matrix core hamiltonian
    std::shared_ptr<Matrix> Dfirst (new Matrix(nso,nso));
    double ** d_p = Dfirst->pointer();
    for(int mu = 0; mu < nso; ++mu){
        for(int nu = 0; nu< nso; ++nu){
            double dum = 0.0;
            for(int i = 0; i < na; ++i){
                dum += Ca->pointer()[mu][i] * Ca->pointer()[nu][i];
            }
            d_p[mu][nu] = dum;
        }
    }

    // evaluate initial energy
    double first_energy = 0.0;

    //first_energy += Dfirst->vector_dot(h);
    first_energy += (nelectron*nelectron)/2.0*Jell->selfval/ Lfac;

    // core hamiltonian contribution
    first_energy += 2.0 * Dfirst->vector_dot(h);

    // build J(mu,nu) = D(lambda,sigma) (mu nu | lambda sigma)
    // build K(mu,nu) = D(lambda,sigma) (mu sigma | lambda nu)
    std::shared_ptr<Matrix> J (new Matrix(nso,nso));
    std::shared_ptr<Matrix> K (new Matrix(nso,nso));

    double ** j_p = J->pointer();
    double ** k_p = K->pointer();
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
    // coulomb / exchange contribution to the energy
    first_energy += 2.0 * Dfirst->vector_dot(J);
    first_energy -=       Dfirst->vector_dot(K);
    
    F->copy(h);
    outfile->Printf("guess energy %f \n", first_energy);
    double e_convergence = options.get_double("E_CONVERGENCE");
    double d_convergence = options.get_double("D_CONVERGENCE");
    int Iter = 0;
    outfile->Printf("Iter \t energy \t energy difference \t density difference \n");

    do {
    
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
        double energy = 0.0;
        energy += (nelectron*nelectron/2.0)*Jell->selfval/Lfac;
        energy += Dfirst->vector_dot(h);
        energy += Dfirst->vector_dot(Fa);

        double density_diff = 0.0;
        std::shared_ptr<Matrix> Fprime = (std::shared_ptr<Matrix>)(new Matrix(F));
        std::shared_ptr<Matrix> Fevec (new Matrix(nso,nso));
        std::shared_ptr<Vector> Feval (new Vector(nso));
        Fprime->diagonalize(Ca,Feval);

        //building density matrix

        std::shared_ptr<Matrix> D (new Matrix(nso,nso));
        double tmp = 0;
        for(int mu = 0; mu < nso; ++mu){
            for(int nu = 0; nu < nso; ++nu){
                for(int i = 0; i < na; ++i){
                    D->pointer()[mu][nu] += Ca->pointer()[mu][i] * Ca->pointer()[nu][i];
                }
            }
        }

        for(int mu = 0; mu < nso; ++mu){
            for(int nu = 0; nu < nso; ++nu){
                density_diff += (D->pointer()[mu][nu] - Dfirst->pointer()[mu][nu])*(D->pointer()[mu][nu] - Dfirst->pointer()[mu][nu]);
            }
        }
        outfile->Printf("%i \t %10.10f \t %10.10f \t  %10.10f \n", Iter, energy, fabs(energy-first_energy), fabs(sqrt(density_diff)) );
        double check = fabs(energy-first_energy);
        first_energy = energy;
        ++Iter;
        Dfirst->copy(D);
        if(check < e_convergence && sqrt(density_diff) < d_convergence){
           double fock_energy = Dfirst->vector_dot(K);
           V->print();
           printf("fock energy: %f",fock_energy);
           return ref_wfn;
        }
    }while(1);

    // Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}

}} // End namespaces

