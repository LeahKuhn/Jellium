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
#include "diis_c.cc"
#include "diis_c.h"
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

    outfile->Printf("\n");
    outfile->Printf( "    ****************************************************\n");
    outfile->Printf( "    *                                                  *\n");
    outfile->Printf( "    *    Jellium Hartree-Fock                          *\n");
    outfile->Printf( "    *                                                  *\n");
    outfile->Printf( "    ****************************************************\n");
    outfile->Printf("\n");

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


    // print some information about this computation
    outfile->Printf("\n");
    outfile->Printf("    ==> Hartree-Fock <==\n");
    outfile->Printf("\n");
    outfile->Printf("\n");
    outfile->Printf("    Number of electrons:              %5i\n",nelectron);
    outfile->Printf("    Number of basis functions:        %5i\n",nso);
    outfile->Printf("    Maximum particle-in-a-box state:  %5i\n",Jell->get_nmax());
    outfile->Printf("\n");

    //build the core hamiltonian
    std::shared_ptr<Matrix> h = (std::shared_ptr<Matrix>)(new Matrix(T));
    std::shared_ptr<Matrix> Ca = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> Shalf = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> S = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    for(int h = 0; h < Jell->nirrep_; h++){
    double** Shalfp = Shalf->pointer(h);
    double** Sp = S->pointer(h);
    for(int i = 0; i < Jell->nsopi_[h];i++){
        Shalfp[i][i] = 1.0;
        Sp[i][i] = 1.0;
    }
    }
    std::shared_ptr<DIIS> diis (new DIIS(Jell->nsopi_[0]*Jell->nsopi_[0]));
    // build core hamiltonian
    V->scale(nelectron); 
    h->add(V);
    int na = nelectron / 2;
    // fock matrix
    std::shared_ptr<Matrix> F = (std::shared_ptr<Matrix>)(new Matrix(h));
     
    // eigenvectors / eigenvalues of fock matrix
    std::shared_ptr<Vector> Feval (new Vector(Jell->nirrep_,Jell->nsopi_));
    
    // diagonalize core hamiltonian, get orbitals
    F->diagonalize(Ca,Feval);
    // build density matrix core hamiltonian
    std::shared_ptr<Matrix> D (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    #pragma omp parallel for 
    for(int h = 0; h < Jell->nirrep_; h++){
    double ** d_p = D->pointer(h);
    double ** c_p = Ca->pointer(h);
    for(int mu = 0; mu < Jell->nsopi_[h]; ++mu){
        for(int nu = 0; nu < Jell->nsopi_[h]; ++nu){
            double dum = 0.0;
            //TODO: pretty sure that this is wrong
            for(int i = 0; i < Jell->Eirrep_[h]; ++i){
                dum += c_p[mu][i] * c_p[nu][i];
            }
            d_p[mu][nu] = dum;
        }
    }
    }
    //printf("trace %f\n",D->trace());
    //D->print(); exit(1);
    double energy = D->vector_dot(h) + D->vector_dot(F);
    outfile->Printf("    initial energy: %20.12lf\n",energy);
    outfile->Printf("\n");

    double gnorm;
    // containers for J and K 
    std::shared_ptr<Matrix> J (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> K (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    


    double ** j_p = J->pointer();
    double ** k_p = K->pointer();
    double ** d_p = D->pointer();
    // convergence parameters
    double e_convergence = options.get_double("E_CONVERGENCE");
    double d_convergence = options.get_double("D_CONVERGENCE");
    int    maxiter       = options.get_int("MAXITER");
    maxiter = 1000;
    int iter = 0;
    outfile->Printf("    ");
    outfile->Printf("  iter");
    outfile->Printf("              energy");
    outfile->Printf("                |dE|");
    outfile->Printf("                |dD|\n");
    int * tmp = (int*)malloc(5*sizeof(int));
    double dele = 0.0;
    double deld = 0.0;
    do {
    bool dampening = true;

        K->zero();
// #pragma omp parallel for
//        for(int mu = 0; mu < nso; ++mu){
//            for(int nu = 0; nu < nso; ++nu){
//                for(int lambda = 0; lambda < nso; ++lambda){
//                    double myK = 0.0;
//                    for(int sigma = 0; sigma < nso; ++sigma){
//                        //if(sigma<=lambda) continue;
//                        double dum = Jell->ERI_int(mu,nu,lambda,sigma);
//                        myK += d_p[ sigma][   nu] * dum;//Jell->ERI_int(mu,sigma,lambda,nu);
//                    }
//                    k_p[mu][lambda] += myK;
//                }
//            }
//        }
//#pragma omp parallel for
//        for(int mu = 0; mu < nso; ++mu){
//            for(int nu = 0; nu < nso; ++nu){
//                double myJ = 0.0;
//                for(int lambda = 0; lambda < nso; ++lambda){
//                    for(int sigma = 0; sigma < nso; ++sigma){
//                        //if(sigma<=lambda) continue;
//                        double dum = Jell->ERI_int(mu,nu,lambda,sigma);
//                            myJ += d_p[lambda][sigma] * dum;//Jell->ERI_int(mu,nu,lambda,sigma);
//                    }
//                }
//                j_p[mu][nu] = myJ;
//            }
//        }
    #pragma omp parallel for
    for (short hp = 0; hp < Jell->nirrep_; hp++) {
        short offp = 0;
        double ** k_p = K->pointer(hp);
        double ** j_p = J->pointer(hp);
        for (int myh = 0; myh < hp; myh++) {
            offp += Jell->nsopi_[myh];
        }
        for (short p = 0; p < Jell->nsopi_[hp]; p++) {
            short pp = p + offp;

            for (short q = p; q < Jell->nsopi_[hp]; q++) {
                short qq = q + offp;

                double dum = 0.0;
                double myJ = 0.0;
                double myK = 0.0;
                for (short hr = 0; hr < Jell->nirrep_; hr++) {
                    double ** d_p = D->pointer(hr);

                    short offr = 0;
                    for (short myh = 0; myh < hr; myh++) {
                        offr += Jell->nsopi_[myh];
                    }

                    for (short r = 0; r < Jell->nsopi_[hr]; r++) {
                        short rr = r + offr;
                        for (short s = r; s < Jell->nsopi_[hr]; s++) {
                            short ss = s + offr;
                            if(r==s){
                            myJ += d_p[r][s] * Jell->ERI_int(pp,qq,rr,ss);
                            myK += d_p[r][s] * Jell->ERI_int(pp,ss,rr,qq);
                            } else {
                            myJ += 2*d_p[r][s] * Jell->ERI_int(pp,qq,rr,ss);
                            myK += 2*d_p[r][s] * Jell->ERI_int(pp,ss,rr,qq);
}
                        }
                    }
                    j_p[p][q] = myJ;
                    j_p[q][p] = myJ;
                    k_p[p][q] = myK;
                    k_p[q][p] = myK;
                }
            }
        }
    }
    //for (int hp = 0; hp < Jell->nirrep_; hp++) {
    //    int offp = 0;
    //    double ** k_p = K->pointer(hp);
    //    for (int myh = 0; myh < hp; myh++) {
    //        offp += Jell->nsopi_[myh];
    //    }
    //    for (int p = 0; p < Jell->nsopi_[hp]; p++) {
    //        int pp = p + offp;

    //        for (int q = p; q < Jell->nsopi_[hp]; q++) {
    //            int qq = q + offp;

    //            double dum = 0.0;
    //            double myK = 0.0;

    //            for (int hr = 0; hr < Jell->nirrep_; hr++) {
    //                double ** d_p = D->pointer(hr);

    //                int offr = 0;
    //                for (int myh = 0; myh < hr; myh++) {
    //                    offr += Jell->nsopi_[myh];
    //                }

    //                for (int r = 0; r < Jell->nsopi_[hr]; r++) {
    //                    int rr = r + offr;
    //                    for (int s = 0; s < Jell->nsopi_[hr]; s++) {
    //                        int ss = s + offr;
    //                        myK += d_p[r][s] * Jell->ERI_int(pp,ss,rr,qq);
    //                    }
    //                }
    //                k_p[p][q] = myK;
    //                k_p[q][p] = myK;
    //            }
    //        }
    //    }
    //}
        F->copy(J);
        F->scale(2.0);
        F->subtract(K);
        F->scale(1.0/Lfac);
        F->add(h);



        double new_energy = 0.0;
        new_energy += (nelectron*nelectron/2.0)*Jell->selfval/Lfac;
        new_energy += D->vector_dot(h);
        new_energy += D->vector_dot(F);

        std::shared_ptr<Matrix> Fprime = (std::shared_ptr<Matrix>)(new Matrix(F));
        std::shared_ptr<Matrix> Fevec (new Matrix(nso,nso));
        std::shared_ptr<Vector> Feval (new Vector(Jell->nirrep_,Jell->nsopi_));
        Fprime->diagonalize(Ca,Feval);

        //building density matrix
        std::shared_ptr<Matrix> Dnew (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        for(int h = 0; h < Jell->nirrep_; h++){
        double ** dnew_p = Dnew->pointer(h);
        double ** c_p = Ca->pointer(h);
        for(int mu = 0; mu < Jell->nsopi_[h]; ++mu){
            for(int nu = 0; nu < Jell->nsopi_[h]; ++nu){
                double dum = 0.0;
                //TODO: pretty sure that this is wrong
                for(int i = 0; i < Jell->Eirrep_[h]; ++i){
                    dum += c_p[mu][i] * c_p[nu][i];
                }
                dnew_p[mu][nu] = dum;
            }
        }
        }
        //exit(1);
        //double ** dnew_p = Dnew->pointer();
        //double tmp = 0;
        //#pragma omp parallel for
        //        for(int mu = 0; mu < nso; ++mu){
        //            for(int nu = 0; nu < nso; ++nu){
        //                for(int i = 0; i < na; ++i){
        //                    dnew_p[mu][nu] += Ca->pointer()[mu][i] * Ca->pointer()[nu][i];
        //                }
        //            }
        //        }

                   diis->WriteVector(&(Fprime->pointer()[0][0]));

                   std::shared_ptr<Matrix> FDSmSDF(new Matrix("FDS-SDF",Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
                   std::shared_ptr<Matrix> DS(new Matrix("DS",Jell->nirrep_,Jell->nsopi_, Jell->nsopi_));
                   DS->gemm(false,false,1.0,D,S,0.0);
                   FDSmSDF->gemm(false,false,1.0,Fprime,DS,0.0);
                   DS.reset();

                   std::shared_ptr<Matrix> SDF(FDSmSDF->transpose());
                   FDSmSDF->subtract(SDF);

                   SDF.reset();

                   std::shared_ptr<Matrix> ShalfGrad(new Matrix("ST^{-1/2}(FDS - SDF)",Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
                   ShalfGrad->gemm(true,false,1.0,Shalf,FDSmSDF,0.0);
                   FDSmSDF.reset();

                   std::shared_ptr<Matrix> ShalfGradShalf(new Matrix("ST^{-1/2}(FDS - SDF)S^{-1/2}", Jell->nirrep_,Jell->nsopi_, Jell->nsopi_));
                   ShalfGradShalf->gemm(false,false,1.0,ShalfGrad,Shalf,0.0);

                   ShalfGrad.reset();

                   // We will use the RMS of the orbital gradient 
                   // to monitor convergence.
                   gnorm = ShalfGradShalf->rms();

                   // The DIIS manager will write the current error vector to disk.
                   diis->WriteErrorVector(&(ShalfGradShalf->pointer()[0][0]));
                   if(gnorm<0.1){
                       diis->Extrapolate(&(Fprime->pointer(0)[0][0]));
                       dampening = false;
                       printf("do diis\n");
                   }
        deld = 0.0;
        for(int h = 0; h < Jell->nirrep_; h++){
        double ** d_p = D->pointer(h);
        double ** dnew_p = Dnew->pointer(h);
        for(int mu = 0; mu < Jell->nsopi_[h]; ++mu){
            for(int nu = 0; nu < Jell->nsopi_[h]; ++nu){
                double dum = d_p[mu][nu] - dnew_p[mu][nu];
                deld += dum * dum;
            }
        }
        }
        deld = sqrt(deld);

        dele = fabs(new_energy-energy);

        outfile->Printf("    %6i%20.12lf%20.12lf%20.12lf\n", iter, new_energy, dele, deld);
        energy = new_energy;
        Fprime->diagonalize(Ca,Feval);

        //building density matrix

        //Dnew = (std::shared_ptr<Matrix>)(new Matrix(nso,nso));
        //dnew_p = Dnew->pointer();
        //#pragma omp parallel for
        //for(int mu = 0; mu < nso; ++mu){
        //    for(int nu = 0; nu < nso; ++nu){
        //        for(int i = 0; i < na; ++i){
        //            dnew_p[mu][nu] += Ca->pointer()[mu][i] * Ca->pointer()[nu][i];
        //        }
        //    }
        //}
        //std::shared_ptr<Matrix> Dnew (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        for(int h = 0; h < Jell->nirrep_; h++){
        double ** dnew_p = Dnew->pointer(h);
        double ** d_p = D->pointer(h);
        double ** c_p = Ca->pointer(h);
        for(int mu = 0; mu < Jell->nsopi_[h]; ++mu){
            for(int nu = 0; nu < Jell->nsopi_[h]; ++nu){
                double dum = 0.0;
                //TODO: pretty sure that this is wrong
                for(int i = 0; i < Jell->Eirrep_[h]; ++i){
                    dum += c_p[mu][i] * c_p[nu][i];
                }
                if(fabs(dum-d_p[mu][nu])>0.02 && dampening){
                    if(dum-d_p[mu][nu]<0){
                       dnew_p[mu][nu] = d_p[mu][nu]-0.02;
                    }else{
                       dnew_p[mu][nu] = d_p[mu][nu]+0.02;
                    }
                }else {
                dnew_p[mu][nu] = dum;
                }
            }
        }
        }
        D->copy(Dnew);

        iter++;
        if( iter > maxiter ) break;
        printf("gnorm: %f\n",gnorm);
        }while(dele > e_convergence || deld > d_convergence);

        if ( iter > maxiter ) {
            throw PsiException("jellium scf did not converge.",__FILE__,__LINE__);
        }

        outfile->Printf("\n");
        outfile->Printf("      SCF iterations converged!\n");
        outfile->Printf("\n");

        double fock_energy = D->vector_dot(K) / Lfac;
        outfile->Printf("    * Jellium HF total energy: %20.12lf\n",energy);
        outfile->Printf("      Fock energy:             %20.12lf\n",fock_energy);
        return ref_wfn;

        // Typically you would build a new wavefunction and populate it with data
        return ref_wfn;
    }

}} // End namespaces

