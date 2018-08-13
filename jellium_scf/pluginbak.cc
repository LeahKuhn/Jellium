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
#include "diis_c.h"
#ifdef _OPENMP
    #include<omp.h>
#else
    #define omp_get_wtime() ( (double)clock() / CLOCKS_PER_SEC )
    #define omp_get_max_threads() 1
#endif

namespace psi{ namespace jellium_scf {

//TODO move these to header file
double ext_field_;
double dipole(double x, int n, int m, double L);
double pulse(double time, double time_length);
void buildfock(std::shared_ptr<Matrix> d_re, std::shared_ptr<Matrix> d_im, double time);
void rk_step(std::shared_ptr<Matrix> density_re, std::shared_ptr<Matrix> density_im, double time);
std::shared_ptr<JelliumIntegrals> Jell;
std::shared_ptr<Matrix> h;
std::shared_ptr<Matrix> F_re;
std::shared_ptr<Matrix> F_im;
std::shared_ptr<Matrix> density_re;
double boxlength;
double time_length;
double Lfac;
double time_step;
 
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
	/*- The time in femtoseconds for the pulse -*/
	options.add_double("TIME_LENGTH", 100);
	/*- The time step in femtoseconds -*/
	options.add_double("TIME_STEP", 0.01);
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
    Lfac = pow((double)nelectron,1.0/3.0)/M_PI;
    boxlength = Lfac * M_PI;

    //grabbing one-electon integrals from mintshelper
    Jell = (std::shared_ptr<JelliumIntegrals>)(new JelliumIntegrals(options));

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

    int nmax = Jell->get_nmax();
    h = (std::shared_ptr<Matrix>)(new Matrix(T));
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
        //printf("%d\n",Jell->nsopi_[h]);
    }
    std::shared_ptr<DIIS> diis (new DIIS(nso*nso));
    // build core hamiltonian
    V->scale(nelectron); 
    h->add(V);
    int na = nelectron / 2;
    // fock matrix
    std::shared_ptr<Matrix> F = (std::shared_ptr<Matrix>)(new Matrix(h));
    std::shared_ptr<Matrix> Fim = (std::shared_ptr<Matrix>)(new Matrix(h));
     
    // eigenvectors / eigenvalues of fock matrix
    std::shared_ptr<Vector> Feval (new Vector(Jell->nirrep_,Jell->nsopi_));
    
    //ground state density and fock matrix
    std::shared_ptr<Matrix> D_ground = (std::shared_ptr<Matrix>)(new Matrix(h));
    std::shared_ptr<Matrix> F_ground = (std::shared_ptr<Matrix>)(new Matrix(h));

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
                //printf("%d %d %d \n",Jell->Eirrep_[h],nu,i);
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
    std::shared_ptr<Matrix> J_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> K_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    
    


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
    bool dampening = false;
    bool do_diis = false;
    do {

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
                        for (short s = 0; s < Jell->nsopi_[hr]; s++) {
                            short ss = s + offr;
                            if(r==s){
                            //printf("p %d q %d r %d s %d \t%f\n",pp,qq,rr,ss,Jell->ERI_int(pp,qq,rr,ss));
                            myJ += d_p[r][s] * Jell->ERI_int(pp,qq,rr,ss);
                            myK += d_p[r][s] * Jell->ERI_int(pp,ss,rr,qq);
                            } else {
                            myJ += d_p[r][s] * Jell->ERI_int(pp,qq,rr,ss);
                            myK += d_p[r][s] * Jell->ERI_int(pp,ss,rr,qq);
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
                   std::shared_ptr<Vector> tmp_F(new Vector(nso*nso));
                   int tmp_vec_offset = 0;
                   for(int h = 0; h < Jell->nirrep_; h++){
                      for(int i = 0; i < Jell->nsopi_[h]; i++){
                         for(int j = 0; j < Jell->nsopi_[h]; j++){
                            tmp_F->pointer()[tmp_vec_offset+j] = Fprime->pointer(h)[i][j];
                         }
                         tmp_vec_offset += nso;
                      }
                      tmp_vec_offset += Jell->nsopi_[h];
                   }
                   diis->WriteVector(&(tmp_F->pointer()[0]));

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
                   std::shared_ptr<Vector> tmp_vec(new Vector(nso*nso));
                   tmp_vec_offset = 0;
                   for(int h = 0; h < Jell->nirrep_; h++){
                      for(int i = 0; i < Jell->nsopi_[h]; i++){
                         for(int j = 0; j < Jell->nsopi_[h]; j++){
                         tmp_vec->pointer()[tmp_vec_offset+j] = ShalfGradShalf->pointer(h)[i][j];
                         tmp_F->pointer()[tmp_vec_offset+j] = Fprime->pointer(h)[i][j];
                         }
                         tmp_vec_offset += nso;
                      }
                      tmp_vec_offset += Jell->nsopi_[h];
                   }
                   //tmp_vec->print();
                   //tmp_F->print();
                   //Fprime->print();;
                   // We will use the RMS of the orbital gradient 
                   // to monitor convergence.
                   gnorm = ShalfGradShalf->rms();
                   // The DIIS manager will write the current error vector to disk.
                   diis->WriteErrorVector(&(tmp_vec->pointer()[0]));
                   //if(iter>=2){
                       diis->Extrapolate(&(tmp_F->pointer()[0]));
                       //printf("do diis\n");
                   //}
                   tmp_vec_offset = 0;
                   for(int h = 0; h < Jell->nirrep_; h++){
                      for(int i = 0; i < Jell->nsopi_[h]; i++){
                         for(int j = 0; j < Jell->nsopi_[h]; j++){
                         Fprime->pointer(h)[i][j] = tmp_F->pointer()[tmp_vec_offset+j];
                         }
                         tmp_vec_offset += nso;
                      } 
                      tmp_vec_offset += Jell->nsopi_[h];
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
        double damp = 0.03;
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
                //if(fabs(dum-d_p[mu][nu])>damp && dampening){
                //    if(dum-d_p[mu][nu]<0){
                //       dnew_p[mu][nu] = d_p[mu][nu]-damp;
                //    }else{
                //      dnew_p[mu][nu] = d_p[mu][nu]+damp;
                //   }
                //}else {
                dnew_p[mu][nu] = dum;
                //}
            }
        }
        }
        D->copy(Dnew);
        if(dele < e_convergence && deld < d_convergence){
           F_re = (std::shared_ptr<Matrix>)(new Matrix(Fprime));
           density_re = (std::shared_ptr<Matrix>)(new Matrix(D)); 
        }

        iter++;
        if( iter > maxiter ) break;
        //printf("gnorm: %f\n",gnorm);
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

outfile->Printf("Ground state density\n");
        int points = options.get_int("N_GRID_POINTS");
        double tmp_d = 0.0;
        int nx = points;
        int ny = points;
        int nz = points;
        double dx = boxlength / ( nx - 1 );
        double dy = boxlength / ( ny - 1 );
        double dz = boxlength / ( nz - 1 );

        //double z = 0.25 * boxlength;
        for (int xid = 0; xid < nx; xid++) {
            //printf("grid %f\n",Jell->grid_points[xid]);
            double x = Jell->grid_points[xid];
            for (int yid = 0; yid < ny; yid++) {
                double y = Jell->grid_points[yid];
                double dum = 0.0;
                for(int zid = 0; zid<nz;zid++){
                double z = Jell->grid_points[zid];

                int offset = 0;
                for (int h = 0; h < Jell->nirrep_; h++) {
                    double ** D_p = D->pointer(h);
                    for(int mu = 0; mu < Jell->nsopi_[h]; mu++){
                        int mux = Jell->MO[mu + offset][0];
                        int muy = Jell->MO[mu + offset][1];
                        int muz = Jell->MO[mu + offset][2];

                        double psi_mu = sin(mux*M_PI*x) * sin(muy*M_PI*y) * sin(muz*M_PI*z) * Jell->w[xid] * Jell->w[yid] * Jell->w[zid];

                        for(int nu = 0; nu < Jell->nsopi_[h]; nu++){
                            int nux = Jell->MO[nu + offset][0];
                            int nuy = Jell->MO[nu + offset][1];
                            int nuz = Jell->MO[nu + offset][2];

                            double psi_nu = sin(nux*M_PI*x) * sin(nuy*M_PI*y) * sin(nuz*M_PI*z);

                            dum += D_p[mu][nu] * psi_mu * psi_nu;
                        }
                    }
                    offset += Jell->nsopi_[h];
                }}
                outfile->Printf("%20.12lf %20.12lf %20.12lf\n",x,y,dum);
                tmp_d += dum;
                dum = 0.0;

            }
            //printf("\n");
        }
        //printf("box length: %20.12lf\n",boxlength);
        //printf("total: %20.12lf\n",tmp_d);
        F_im = (std::shared_ptr<Matrix>)(new Matrix(F_re));
        F_im->zero();
        std::shared_ptr<Matrix> density_im = (std::shared_ptr<Matrix>)(new Matrix(F_re));
        density_im->zero();
        time_length = options.get_double("TIME_LENGTH");
        time_step = options.get_double("TIME_STEP");
         
        
	
        iter = 0;	

        while(iter<(time_length/time_step)){	

	    //start of RT-TDHF
            rk_step(density_re, density_im, iter*time_step);
            // evaluate dipole moment
            double dip = 0.0;
            int offset = 0;
            for(int h = 0; h < Jell->nirrep_; h++){
                double ** F_re_p = F_re->pointer(h);
                for(int i = 0; i < Jell->nsopi_[h]; i++){
                    for(int j = 0; j < Jell->nsopi_[h]; j++){
                        dip += density_re->pointer(h)[i][j] * dipole(boxlength,Jell->MO[offset+i][0],Jell->MO[offset+j][0],boxlength);
                    }
                }
                offset += Jell->nsopi_[h];
            }
            printf("%20.12lf %20.12lf %20.12lf %20.12lf %20.12lf\n",iter * time_step,dip,density_re->rms(),density_im->rms(),ext_field_);

	    
            //since only propagating in the Z direction psi(i) psi(j) is integrated over Z
	    iter++;

        }

        //printf("%d\n",iter);
        // Typically you would build a new wavefunction and populate it with data
        return ref_wfn;
    }


//extern "C" PSI_API
double dipole(double x, int n, int m, double L){

        double pixmnL = ((M_PI*x*(m+n))/L);
        double pixm_nL = ((M_PI*x*(m-n))/L);
        if(m==n){
            return (-2/L)*(-L*L*(-2*M_PI*M_PI*n*n+2*M_PI*n*sin(2*M_PI*n)+cos(2*M_PI*n)-1)/(8*M_PI*M_PI*n*n));
        }
        //L is total length
        //m is second sin term n is first term
        //indenfinite integral
        //return -(2/L)*L*((M_PI*x*(m-n)*sin(pixm_nL)+L*cos(pixm_nL))/((m-n)*(m-n))-M_PI*x*(m+n)*sin(pixmnL)+L*cos(pixmnL))/(2*M_PI*M_PI);
        //printf("L %f\n",L);
        //return 0.5; 
        //definite integral
        return (-2/L)*(1/(M_PI*M_PI*(m-n)*(m-n)*(m+n)*(m+n)))*L*L*(-M_PI*n*(n*n-m*m)*sin(M_PI*m)*cos(M_PI*n)+cos(M_PI*m)*((M_PI*m*(n*n-m*m))*sin(M_PI*n)+2*m*n*cos(M_PI*n))+m*m*sin(M_PI*m)*sin(M_PI*n)+n*n*sin(M_PI*m)*sin(M_PI*n)-2*m*n);
}
//extern "C" PSI_API
double pulse(double time, double time_length){
        //TODO find actual weight
        //double weight = 0.00001;
        //if(time-time_length/2 == 0){return 0;}
        //return sin(weight*time)*(pow(sin(M_PI/(time_length*(time-(time_length/2)))),2));



    // TODO: these parameters should be set in the input file
    double laser_time = 4.134;             // 0.1 fs, picked for no reason at all
    double laser_freq = 0.734986612218858; // 20 eV, also picked for no reason
    double laser_amp  = 0.5; 

    // Gaussian pulse
    ext_field_ = laser_amp * exp(-((time-1.5*laser_time)*(time-1.5*laser_time)) / (0.3606738*laser_time*laser_time)) * sin(laser_freq*time);

    // continuous pulse
    //ext_field_ = laser_amp * sin(laser_freq*time);

    return ext_field_;
}
//TODO change this to take in the starting density and fock matrices
void rk_step(std::shared_ptr<Matrix> density_re, std::shared_ptr<Matrix> density_im, double time){

        std::shared_ptr<Matrix> k1_re (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> k2_re (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> k3_re (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> k4_re (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> k1_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> k2_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> k3_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> k4_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> d_re_tmp (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> d_im_tmp(new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
        std::shared_ptr<Matrix> tmp (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));

        //doing first ground state perterbation
        //To evaluate the commutator relation -i[F,D]

        buildfock(density_re, density_im, time);

        k1_re->gemm(false,false,1.0,density_im,F_re,0.0);
        k1_re->gemm(false,false,1.0,density_re,F_im,1.0);
        k1_re->gemm(false,false,-1.0,F_im,density_re,1.0);
        k1_re->gemm(false,false,-1.0,F_re,density_im,1.0);
        
        
        k1_im->gemm(false,false,-1.0,density_re,F_re,0.0);
        k1_im->gemm(false,false,1.0,density_im,F_im,1.0);
        k1_im->gemm(false,false,1.0,F_re,density_re,1.0);
        k1_im->gemm(false,false,-1.0,F_im,density_im,1.0);
        
        k1_re->scale(0.5 * time_step);
        k1_im->scale(0.5 * time_step);

        d_re_tmp->copy(density_re);
        d_im_tmp->copy(density_im);

        d_re_tmp->add(k1_re);
        d_im_tmp->add(k1_im);

        buildfock(d_re_tmp,d_im_tmp, time + 0.5 * time_step);

        k2_re->gemm(false,false,1.0,d_im_tmp,F_re,0.0);
        k2_re->gemm(false,false,1.0,d_re_tmp,F_im,1.0);
        k2_re->gemm(false,false,-1.0,F_im,d_re_tmp,1.0);
        k2_re->gemm(false,false,-1.0,F_re,d_im_tmp,1.0);
        
        k2_im->gemm(false,false,-1.0,d_re_tmp,F_re,0.0);
        k2_im->gemm(false,false,1.0,d_im_tmp,F_im,1.0);
        k2_im->gemm(false,false,1.0,F_re,d_re_tmp,1.0);
        k2_im->gemm(false,false,-1.0,F_im,d_im_tmp,1.0);
        
        k2_re->scale(0.5 * time_step);
        k2_im->scale(0.5 * time_step);

        d_re_tmp->copy(density_re);
        d_im_tmp->copy(density_im);

        d_re_tmp->add(k2_re);
        d_im_tmp->add(k2_im);
        
        buildfock(d_re_tmp,d_im_tmp, time + 0.5 * time_step);

        k3_re->gemm(false,false,1.0,d_im_tmp,F_re,0.0);
        k3_re->gemm(false,false,1.0,d_re_tmp,F_im,1.0);
        k3_re->gemm(false,false,-1.0,F_im,d_re_tmp,1.0);
        k3_re->gemm(false,false,-1.0,F_re,d_im_tmp,1.0);
        
        k3_im->gemm(false,false,-1.0,d_re_tmp,F_re,0.0);
        k3_im->gemm(false,false,1.0,d_im_tmp,F_im,1.0);
        k3_im->gemm(false,false,1.0,F_re,d_re_tmp,1.0);
        k3_im->gemm(false,false,-1.0,F_im,d_im_tmp,1.0);
        
        k3_re->scale(time_step);
        k3_im->scale(time_step);

        d_re_tmp->copy(density_re);
        d_im_tmp->copy(density_im);

        d_re_tmp->add(k3_re);
        d_im_tmp->add(k3_im);

        buildfock(d_re_tmp,d_im_tmp, time+(time_step));

        k4_re->gemm(false,false,1.0,d_im_tmp,F_re,0.0);
        k4_re->gemm(false,false,1.0,d_re_tmp,F_im,1.0);
        k4_re->gemm(false,false,-1.0,F_im,d_re_tmp,1.0);
        k4_re->gemm(false,false,-1.0,F_re,d_im_tmp,1.0);
        
        k4_im->gemm(false,false,-1.0,d_re_tmp,F_re,0.0);
        k4_im->gemm(false,false,1.0,d_im_tmp,F_im,1.0);
        k4_im->gemm(false,false,1.0,F_re,d_re_tmp,1.0);
        k4_im->gemm(false,false,-1.0,F_im,d_im_tmp,1.0);
        
        k4_re->scale(time_step);
        k4_im->scale(time_step);

        d_re_tmp->zero();
        d_im_tmp->zero();
        
        // D += K1
        //rescaling it to original since it was halved for the creation of k2
        k1_re->scale(2.0);
        k1_im->scale(2.0);
        d_re_tmp->add(k1_re);
        d_im_tmp->add(k1_im);

        // D += 2 K2
        //rescaling it to original since it was halved for the creation of k3
        //and we want total scaling of 2
        k2_re->scale(4.0);
        k2_im->scale(4.0);
        d_re_tmp->add(k2_re);
        d_im_tmp->add(k2_im);
   
        // D += 2 K3
        k3_re->scale(2.0);
        k3_im->scale(2.0);
        d_re_tmp->add(k3_re);
        d_im_tmp->add(k3_im);

        // D += K4
        d_re_tmp->add(k4_re);
        d_im_tmp->add(k4_im);

        d_re_tmp->scale(1/6.0);
        d_im_tmp->scale(1/6.0);

        density_re->add(d_re_tmp);
        density_im->add(d_im_tmp);
  
} 


void buildfock(std::shared_ptr<Matrix> d_re, std::shared_ptr<Matrix> d_im, double time){
    std::shared_ptr<Matrix> J (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> K (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> J_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> K_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
          #pragma omp parallel for
          for (short hp = 0; hp < Jell->nirrep_; hp++) {
              short offp = 0;
              double ** k_p = K->pointer(hp);
              double ** j_p = J->pointer(hp);
              double ** kim_p = K_im->pointer(hp);
              double ** jim_p = J_im->pointer(hp);
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
                      double myJim = 0.0;
                      double myKim = 0.0;
                      for (short hr = 0; hr < Jell->nirrep_; hr++) {
                          double ** d_p = d_re->pointer(hr);
                          double ** dim_p = d_im->pointer(hr);

                          short offr = 0;
                          for (short myh = 0; myh < hr; myh++) {
                              offr += Jell->nsopi_[myh];
                          }

                          for (short r = 0; r < Jell->nsopi_[hr]; r++) {
                              short rr = r + offr;
                              for (short s = 0; s < Jell->nsopi_[hr]; s++) {
                                  short ss = s + offr;
                                  myJ += d_p[r][s] * Jell->ERI_int(pp,qq,rr,ss);
                                  double eri = Jell->ERI_int(pp,ss,rr,qq);
                                  myK += d_p[r][s] * eri;
                                  myKim -= dim_p[r][s] * eri;
                              }
                          }
                          j_p[p][q] = myJ;
                          j_p[q][p] = myJ;
                          k_p[p][q] = myK;
                          k_p[q][p] = myK;
                          kim_p[p][q] =  myKim;
                          kim_p[q][p] = -myKim;
                      }
                  }
              }
          }
        F_re->copy(J);
        F_re->scale(2.0);
        F_re->subtract(K);
        F_re->add(h);
        F_im->copy(K_im);
        F_im->scale(1.0/Lfac);
        F_re->scale(1.0/Lfac);

        int offset = 0;
        for(int h = 0; h < Jell->nirrep_; h++){
            double ** F_re_p = F_re->pointer(h);
            for(int i = 0; i < Jell->nsopi_[h]; i++){
                for(int j = 0; j < Jell->nsopi_[h]; j++){
                    F_re_p[i][j] -= dipole(boxlength,Jell->MO[offset+i][0],Jell->MO[offset+j][0],boxlength)*pulse(time,time_length);
                    //F_re_p[i][j] += dipole(boxlength,Jell->MO[offset+i][0],Jell->MO[offset+j][0],boxlength);
                }
            }
            offset += Jell->nsopi_[h];
        }

}
}} // End namespaces


