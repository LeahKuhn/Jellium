/*
 * @BEGIN LICENSE
 *
 * Jellium_scf by Daniel Gibney, a plugin to:
 * Run RHF, RTTDHF, and CIS calculations using particle in a cube basis functions
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
#include "blas.h"

//Define functions used in case OpenMP is not found to allow compilation
#ifdef _OPENMP
    #include<omp.h>
#else
    #define omp_get_wtime() ( (double)clock() / CLOCKS_PER_SEC )
    #define omp_get_max_threads() 1
#endif

using namespace std;
using namespace psi;
namespace psi{ namespace jellium_scf {

//TODO move these to header file
double ext_field_;

//Takes in the vectors containing the mx, my, mz quantum numbers for two particle in a box functions
//and the length of the box to give the dipole moment between the two
double dipole(int * n, int * m, double L);

//Takes in the current time and returns the strength of the external pulse
double pulse(double time);

//Builds the fock matrix from the current real and imaginary densities and time
//The time here is used to apply the external field pulse to the fock matrices
//This function relies on the two shared_ptr<Matrix> given beneath which 
//is where it puts the fock matrices it builds 
void buildfock(std::shared_ptr<Matrix> d_re, std::shared_ptr<Matrix> d_im, double time);
std::shared_ptr<Matrix> F_re;
std::shared_ptr<Matrix> F_im;

//Takes a step through time using the RK4 integrator to evaluate the commutator -i[F,D]
//This function will also rely on the fock matrices declared above
void rk_step(std::shared_ptr<Matrix> density_re, std::shared_ptr<Matrix> density_im, double time);

//This function takes in vectors containing the dipole moment as the system 
//is propagated through time and the assocaited external field at each time step
//and generates the frequency domain response
void fourier(double* Vre, double* Vfield);

//Jellium object which calculates the integrals
std::shared_ptr<JelliumIntegrals> Jell;

//The core hamiltonian
std::shared_ptr<Matrix> h;

//The real density
std::shared_ptr<Matrix> density_re;
double boxlength;
double time_length;

//The factor needed to adjust the integrals calculated assuming a box of length pi
//to any arbitrary length
double Lfac;

double time_step;
double laser_time;
double laser_freq;
double laser_amp;

//Adds options to the plugin that the input file can change 
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
        /*- The length of the box in nm -*/
        options.add_double("LENGTH", 1.0);
        /*- The total time in au -*/
        options.add_double("TIME_LENGTH", 100);
        /*- The time step in au -*/
        options.add_double("TIME_STEP", 0.01);
        ///*- The density of the box in e/nm^3 -*/
        //options.add_double("DENSITY", 92);    length and num electrons determines density
        ///*- Enable ground state density output -*/
        options.add_bool("PRINT_DENSITY", false);
        /*- Laser time length in au -*/
        options.add_double("LASER_TIME",4.134);
        /*- Laser frequency -*/
        options.add_double("LASER_FREQ",0.734986612218858);
        /*- Laser amplitude -*/
        options.add_double("LASER_AMP",0.5); 
        /*- Fast eri? memory expensive -*/
        options.add_bool("FAST_ERI",false); 
        /*- Do DIIS? -*/
        options.add_bool("DIIS",true); 
        /*- Number of Fock matrices to DIIS -*/
        options.add_int("DIIS_NUM",20); 
        /*- Fourier Transform? -*/
        options.add_bool("FOURIER",false); 
        /*- CIS? -*/
        options.add_bool("CIS",false); 
        /*- Use symmetry-*/
        options.add_bool("SYMMETRY",true); 
        /*- Fourier frequency minimum in ev-*/
        options.add_double("FOURIER_MIN", 0); 
        /*- Fourier frequency maximum in ev-*/
        options.add_double("FOURIER_MAX", 5.0); 
        /*- Change in Fourier frequency in ev-*/
        options.add_double("FOURIER_DELTA",0.000001); 
        /*- Print orbitals and energies -*/
        options.add_bool("PRINT_ORBITALS",false);
        /*- Fourier output -*/
        options.add_str("FOURIER_OUTPUT","fourier.dat");
        /*- CIS output -*/
        options.add_str("CIS_OUTPUT","cis.dat");
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
    //Lfac = pow((double)nelectron,1.0/3.0)/M_PI;
    //boxlength = Lfac * M_PI;
    
    //since box is already pi a.u. long
    double length_nm = options.get_double("length");
    Lfac = length_nm * 18.89725988 / M_PI;
    boxlength = options.get_double("LENGTH");
    
    //Creating Jellium objecet
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
    h = (std::shared_ptr<Matrix>)(new Matrix(T));
    std::shared_ptr<Matrix> Ca = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> Shalf = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> S = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    for(int irrep = 0; irrep < Jell->nirrep_; irrep++){
        double** Shalfp = Shalf->pointer(irrep);
        double** Sp = S->pointer(irrep);
        for(int i = 0; i < Jell->nsopi_[irrep]; i++){
            Shalfp[i][i] = 1.0;
            Sp[i][i] = 1.0;
        }
    }
    std::shared_ptr<DIIS> diis (new DIIS(nso*nso, options.get_int("DIIS_NUM")));

    // build core hamiltonian
    V->scale(nelectron); 
    h->add(V);

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
    for(int irrep = 0; irrep < Jell->nirrep_; irrep++){
        double ** d_p = D->pointer(irrep);
        double ** c_p = Ca->pointer(irrep);
        for(int mu = 0; mu < Jell->nsopi_[irrep]; ++mu){
            for(int nu = 0; nu < Jell->nsopi_[irrep]; ++nu){
                double dum = 0.0;
                for(int i = 0; i < Jell->Eirrep_[irrep]; ++i){
                    dum += c_p[mu][i] * c_p[nu][i];
                }
                d_p[mu][nu] = dum;
            }
        }
    }
    double energy = D->vector_dot(h) + D->vector_dot(F);
    outfile->Printf("    initial energy: %20.12lf\n",energy);
    outfile->Printf("\n");

    // containers for J and K 
    std::shared_ptr<Matrix> J (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> K (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> K_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));

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
        K->zero();
        J->zero();
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
        if(iter>=2){Jell->iter = 2;}

        //create fock matrix from pieces
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
        #pragma omp parallel for shared(deld)
        for(int h = 0; h < Jell->nirrep_; h++){
            double ** dnew_p = Dnew->pointer(h);
            double ** c_p = Ca->pointer(h);
            for(int mu = 0; mu < Jell->nsopi_[h]; ++mu){
                for(int nu = 0; nu < Jell->nsopi_[h]; ++nu){
                    double dum = 0.0;
                    for(int i = 0; i < Jell->Eirrep_[h]; ++i){
                        dum += c_p[mu][i] * c_p[nu][i];
                    }
                    dnew_p[mu][nu] = dum;
                }
            }
        }

        //turning fock matrix irreps into vector for DIIS
        std::shared_ptr<Vector> tmp_F(new Vector(nso*nso));
        int tmp_vec_offset = 0;
        for(int irrep = 0; irrep < Jell->nirrep_; irrep++){
            for(int i = 0; i < Jell->nsopi_[irrep]; i++){
                for(int j = 0; j < Jell->nsopi_[irrep]; j++){
                    tmp_F->pointer()[tmp_vec_offset+j] = Fprime->pointer(irrep)[i][j];
                }
                tmp_vec_offset += nso;
            }
            tmp_vec_offset += Jell->nsopi_[irrep];
        }

        //DIIS procedure
        if(options.get_bool("DIIS")){
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
                diis->WriteErrorVector(&(tmp_vec->pointer()[0]));
                diis->Extrapolate(&(tmp_F->pointer()[0]));
        } 
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

        //change in density
        deld = 0.0;
        #pragma omp parallel for shared(deld)
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

        //calculating new density
        Fprime->diagonalize(Ca,Feval);
        #pragma omp parallel for
        for(int h = 0; h < Jell->nirrep_; h++){
            double ** dnew_p = Dnew->pointer(h);
            double ** c_p = Ca->pointer(h);
            for(int mu = 0; mu < Jell->nsopi_[h]; ++mu){
                for(int nu = 0; nu < Jell->nsopi_[h]; ++nu){
                    double dum = 0.0;
                    for(int i = 0; i < Jell->Eirrep_[h]; ++i){
                        dum += c_p[mu][i] * c_p[nu][i];
                    }
                    dnew_p[mu][nu] = dum;
                }
            }
        }
        D->copy(Dnew);
        if(options.get_bool("FAST_ERI")){
           Jell->fast_eri_done = true;
        }
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
    if(options.get_bool("DIIS")){
       diis = std::shared_ptr<DIIS> (new DIIS(0,0));
    }
    outfile->Printf("\n");
    outfile->Printf("      SCF iterations converged!\n");
    outfile->Printf("\n");

    double fock_energy = D->vector_dot(K) / Lfac;
    outfile->Printf("    * Jellium HF total energy: %20.12lf\n",energy);
    outfile->Printf("      Fock energy:             %20.12lf\n",fock_energy);

    if(options.get_bool("print_density")){
        outfile->Printf("Ground state density\n");
        int points = options.get_int("N_GRID_POINTS");
        double tmp_d = 0.0;
        int nx = points;
        int ny = points;
        int nz = points;

        for (int xid = 0; xid < nx; xid++) {
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
        }
    }
    F_im = (std::shared_ptr<Matrix>)(new Matrix(F_re));
    F_im->zero();
    std::shared_ptr<Matrix> density_im = (std::shared_ptr<Matrix>)(new Matrix(F_re));
    density_im->zero();

    //get input options value's
    time_length = options.get_double("TIME_LENGTH");
    time_step = options.get_double("TIME_STEP");
    laser_time = options.get_double("LASER_TIME");
    laser_freq = options.get_double("LASER_FREQ");
    laser_amp  = options.get_double("LASER_AMP");


    iter = 0;	
    

    if(options.get_bool("CIS")){
    int ** cis_nsopi = (int**)malloc(Jell->nirrep_*sizeof(double*));
    int * new_nsopi = (int*)malloc(Jell->nirrep_*sizeof(double));
    for(int i = 0; i < Jell->nirrep_; i++){
        cis_nsopi[i] = (int*)malloc(2*sizeof(double*));
    }
    for(int i = 0; i < Jell->nirrep_; i++){
                cis_nsopi[i][0] = Jell->Eirrep_[i];
                cis_nsopi[i][1] = Jell->nsopi_[i]-Jell->Eirrep_[i];
        new_nsopi[i] = (((Jell->nsopi_[i]-Jell->Eirrep_[i])*Jell->Eirrep_[i]));
    }
    std::shared_ptr<Matrix> cis_matrix = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,new_nsopi,new_nsopi));
    std::shared_ptr<Matrix> cis_diagonal = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,new_nsopi,new_nsopi));

    //Converting Fock to MO basis (should be diagonal)
    std::shared_ptr<Matrix> MO_fock = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    for(int h = 0; h < Jell->nirrep_; h++){
       double ** F_ptr = F_re->pointer(h);
       double ** MOF_ptr = MO_fock->pointer(h);
       double ** Ca_ptr = Ca->pointer(h);
       #pragma omp parallel for
       for(int i = 0; i < Jell->nsopi_[h]; i++){
          for(int j = 0; j < Jell->nsopi_[h]; j++){
             for(int mu = 0; mu < Jell->nsopi_[h]; mu++){
                for(int nu = 0; nu < Jell->nsopi_[h]; nu++){            
                   MOF_ptr[i][j] += Ca_ptr[mu][i]*Ca_ptr[nu][j]*F_ptr[mu][nu];
                }
             }
          }
       }
    }
    //TODO add an option to remove symmetry and then use that for the first test of this stuff and also test
    //that removing symmetry gives the same results for time propagation and then CIS when done
    //For the time being we are storing everything...

    //Creating MO_eri structure
    double **** MO_eri = (double****)malloc(nso*sizeof(double***));
    double **** MO_eri_test = (double****)malloc(nso*sizeof(double***));
    for(int i = 0; i < nso; i++){
        MO_eri[i] = (double***)malloc(nso*sizeof(double***));
        MO_eri_test[i] = (double***)malloc(nso*sizeof(double***));
        for(int j = 0; j < nso; j++){
            MO_eri[i][j] = (double**)malloc(nso*sizeof(double**));
            MO_eri_test[i][j] = (double**)malloc(nso*sizeof(double**));
            for(int k = 0; k < nso; k++){
                MO_eri[i][j][k] = (double*)malloc(nso*sizeof(double*));
                MO_eri_test[i][j][k] = (double*)malloc(nso*sizeof(double*));
                for(int l = 0; l < nso; l++){
                    MO_eri[i][j][k][l] = 0;
                    MO_eri_test[i][j][k][l] = 0;
                }
            }
        }
    }

    //Calculating MO_eri's
    double ** c_p = Ca->pointer(0);

    // AO basis integrals
    for(int mu = 0; mu < nso; mu++){
        for(int nu = 0; nu < nso; nu++){
            for(int lambda = 0; lambda < nso; lambda++){
                for(int sigma = 0; sigma < nso; sigma++){
                    MO_eri_test[mu][nu][lambda][sigma] = Jell->ERI_int(mu,nu,lambda,sigma);
                }
            }
        }
    }
    for(int p = 0; p < nso; p++){
        for(int nu = 0; nu < nso; nu++){
            for(int lambda = 0; lambda < nso; lambda++){
                for(int sigma = 0; sigma < nso; sigma++){
                    double dum = 0.0;
                    for(int mu = 0; mu < nso; mu++){
                        dum += MO_eri_test[mu][nu][lambda][sigma] * c_p[mu][p];
                    }
                    MO_eri[p][nu][lambda][sigma] = dum;
                }
            }
        }
    }
    for(int p = 0; p < nso; p++){
        for(int q = 0; q < nso; q++){
            for(int lambda = 0; lambda < nso; lambda++){
                for(int sigma = 0; sigma < nso; sigma++){
                    double dum = 0.0;
                    for(int nu = 0; nu < nso; nu++){
                        dum += MO_eri[p][nu][lambda][sigma] * c_p[nu][q];
                    }
                    MO_eri_test[p][q][lambda][sigma] = dum;
                }
            }
        }
    }
    for(int p = 0; p < nso; p++){
        for(int q = 0; q < nso; q++){
            for(int r = 0; r < nso; r++){
                for(int sigma = 0; sigma < nso; sigma++){
                    double dum = 0.0;
                    for(int lambda = 0; lambda < nso; lambda++){
                        dum += MO_eri_test[p][q][lambda][sigma] * c_p[lambda][r];
                    }
                    MO_eri[p][q][r][sigma] = dum;
                }
            }
        }
    }
    for(int p = 0; p < nso; p++){
        for(int q = 0; q < nso; q++){
            for(int r = 0; r < nso; r++){
                for(int s = 0; s < nso; s++){
                    double dum = 0.0;
                    for(int sigma = 0; sigma < nso; sigma++){
                        dum += MO_eri[p][q][r][sigma] * c_p[sigma][s];
                    }
                    MO_eri_test[p][q][r][s] = dum;
                }
            }
        }
    }
    // dumb way
    //for(int p = 0; p < nso; p++){
    //   for(int q = 0; q < nso; q++){
    //      for(int r = 0; r < nso; r++){
    //         for(int s = 0; s < nso; s++){
    //            double dum = 0.0;
    //            for(int theta = 0; theta < nso; theta++){
    //               for(int lambda = 0; lambda < nso; lambda++){
    //                  for(int nu = 0; nu < nso; nu++){
    //                     for(int mu = 0; mu < nso; mu++){
    //                        dum += c_p[p][theta]*c_p[q][lambda]*c_p[r][nu]*c_p[s][mu]*Jell->ERI_int(theta,lambda,nu,mu); 
    //                     }
    //                  }
    //               }
    //            }
    //            MO_eri[p][q][r][s] = dum;
    //            //printf("%f\n",MO_eri[p][q][r][s]);
    //         }
    //      }
    //   }
    //} 
    
    // for(int p = 0; p < nso; p++){
    //   for(int q = 0; q < nso; q++){
    //      for(int r = 0; r < nso; r++){
    //         for(int s = 0; s < nso; s++){
    //            if( fabs(MO_eri_test[p][q][r][s]-MO_eri[p][q][r][s]) > 1e-8){
    //                printf("%f ",MO_eri[p][q][r][s]);
    //                printf("%f\n",MO_eri_test[p][q][r][s]);
    //            }
    //         }
    //      }
    //   }
    //}

    std::shared_ptr<Matrix> MO_dipole (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));

    //converting dipole to the MO basis
    //TODO this wont work with symmetry

    if ( Jell->nirrep_ > 1 ) {
        throw PsiException("Jellium CIS does not work with symmetry",__FILE__,__LINE__);
    }

    double ** MO_dipole_ptr = MO_dipole->pointer(0);
    
    for(int i = 0; i < nso; i++){
       for(int j = 0; j < nso; j++){
          for(int theta = 0; theta < nso; theta++){
             for(int lambda = 0; lambda < nso; lambda++){
                MO_dipole_ptr[i][j] += dipole(Jell->MO[theta], Jell->MO[lambda], boxlength) * c_p[theta][i] * c_p[lambda][j];
             }
          }
       }
    }

    //Doing CIS - this definitely won't work with symmetry
    for(int h = 0; h < Jell->nirrep_; h++){
       double ** cis_ptr = cis_matrix->pointer(h);
       double ** F_ptr = F_re->pointer(h);
       double ** MOF_ptr = MO_fock->pointer(h);
       for(int i = 0; i < cis_nsopi[h][0]; i++){
          for(int a = 0; a < cis_nsopi[h][1]; a++){
             int offset = 0;
             for(int k = 0; k < h; k++){
                 offset += Jell->nsopi_[k];
             }
             int ai = a + cis_nsopi[h][0];
             int Hai = i*cis_nsopi[h][1]+a;
             for(int j = 0; j < cis_nsopi[h][0]; j++){
                for(int b = 0; b < cis_nsopi[h][1]; b++){
                   int bj = b + cis_nsopi[h][0];
                   int Hbj = j*cis_nsopi[h][1]+b;
                   if(i == j && a == b){
                      cis_ptr[Hai][Hbj] += MO_fock->pointer()[ai][ai];
                      cis_ptr[Hai][Hbj] -= MO_fock->pointer()[i][i];
                   }
                   cis_ptr[Hai][Hbj] += (2*MO_eri_test[i+offset][ai+offset][j+offset][bj+offset])/Lfac;
                   cis_ptr[Hai][Hbj] -= MO_eri_test[i+offset][j+offset][ai+offset][bj+offset]/Lfac;
                }
             }    
          }
       }
    }
    std::shared_ptr<Vector> cis_eval = (std::shared_ptr<Vector>)(new Vector(Jell->nirrep_,new_nsopi));
    std::shared_ptr<Matrix> cis_trans = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,new_nsopi,new_nsopi));
    std::shared_ptr<Matrix> cis_matrix_changing = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,new_nsopi,new_nsopi));
    std::shared_ptr<Matrix> cis_matrix_changing1 = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,new_nsopi,new_nsopi));
    std::shared_ptr<Matrix> cis_matrix_invert = (std::shared_ptr<Matrix>)(new Matrix(Jell->nirrep_,new_nsopi,new_nsopi));
    cis_matrix->print();
    cis_matrix->diagonalize(cis_trans,cis_eval);
    cis_matrix_changing->copy(cis_matrix);
    
    outfile->Printf("Coefficients\n");
    cis_trans->print();

    outfile->Printf("Eigenvalues\n");
    cis_eval->print();

    double energy = 0.0;
    int other_num = 0;
    bool there = false;
    double * tmp_vector = (double*)malloc(new_nsopi[0]*sizeof(double));
    double * tmp_vector_two = (double*)malloc(new_nsopi[0]*sizeof(double));
    for(int i = 0; i < new_nsopi[0]; i++){
       tmp_vector[i] = 0.0;
       tmp_vector_two[i] = 0.0;
    }
    for(int i = 0; i < new_nsopi[0]; i++){
       for(int j = 0; j < new_nsopi[0]; j++){
          tmp_vector_two[i] += cis_trans->pointer()[j][i];

       }
    }
    for(int k = 0; k < cis_nsopi[0][0]*cis_nsopi[0][1]; k++){
       double energy = 0.0;
       for(int i = 0; i < cis_nsopi[0][0]*cis_nsopi[0][1]; i++){
          for(int j = 0; j < cis_nsopi[0][0]*cis_nsopi[0][1]; j++){
             energy += cis_trans->pointer()[i][k]*cis_matrix->pointer()[j][i]*cis_trans->pointer()[j][k];
          }
       }
       printf("energy: %f dipole: %f\n",energy*27.211,MO_dipole->pointer()[k/cis_nsopi[0][1]][(k%cis_nsopi[0][1])+cis_nsopi[0][0]]);
       printf("K: %d i: %d a: %d\n",k,k/cis_nsopi[0][1],(k%cis_nsopi[0][1])+cis_nsopi[0][0]);
    }
    cis_matrix_changing->zero();
    for(int i = 0; i < cis_nsopi[0][0]*cis_nsopi[0][1]; i++){
       cis_matrix_changing->pointer()[i][i] = cis_eval->pointer()[i];
    }

    for(int i = 0; i < cis_nsopi[0][0]*cis_nsopi[0][1]; i++){
       double tmp = 0.0;
       double largest = 0.0;
       int location = 0;
       for(int j = 0; j < cis_nsopi[0][0]*cis_nsopi[0][1]; j++){
          tmp += cis_trans->pointer()[j][i] * cis_trans->pointer()[j][i];
          if(cis_trans->pointer()[j][i] * cis_trans->pointer()[j][i] > largest){
             largest = cis_trans->pointer()[j][i] * cis_trans->pointer()[j][i];
             location = j;
          }
       }
       printf("eval: %f location: %d\n",cis_eval->pointer()[i],location);
    }
    exit(1);

    cis_matrix_invert->copy(cis_trans); 
    //cis_matrix_invert->transpose_this(); 
    cis_matrix_invert->general_invert();
 
    cis_matrix_changing1->gemm(false,false,1.0,cis_trans,cis_matrix_changing,0.0); 
    cis_matrix_changing->gemm(false,false,1.0,cis_matrix_changing1,cis_matrix_invert,0.0);

    cis_matrix_changing->print();
exit(1);
    
    for(int k = 0; k < cis_nsopi[0][0]*cis_nsopi[0][1]; k++){
       std::shared_ptr<Vector> tmp = (std::shared_ptr<Vector>)(new Vector(Jell->nirrep_,new_nsopi));
       for(int i = 0; i < cis_nsopi[0][0]*cis_nsopi[0][1]; i++){
          for(int j = 0; j < cis_nsopi[0][0]*cis_nsopi[0][1]; j++){
             tmp->pointer()[i] += cis_matrix_changing->pointer()[j][i] * cis_trans->pointer()[k][j];
          }
       }
       tmp->print();
    }
    exit(1); 
       


//    for(int num = 0; num < cis_nsopi[0][0]*cis_nsopi[0][1]; num++){
//    for(int h = 0; h < Jell->nirrep_; h++){
//       double ** cis_ptr = cis_matrix->pointer(h);
//       double ** F_ptr = F_re->pointer(h);
//       double ** MOF_ptr = MO_fock->pointer(h);
//       for(int i = 0; i < cis_nsopi[h][0]; i++){
//          for(int a = 0; a < cis_nsopi[h][1]; a++){
//            energy = 0.0;
//             int Hai = i*cis_nsopi[h][1]+a;
//             int offset = 0;
//             for(int k = 0; k < h; k++){
//                 offset += Jell->nsopi_[k];
//             }
//             int ai = a + cis_nsopi[h][0];
//             for(int j = 0; j < cis_nsopi[h][0]; j++){
//                for(int b = 0; b < cis_nsopi[h][1]; b++){
//                   int Hbj = j*cis_nsopi[h][1]+b;
//                   int bj = b + cis_nsopi[h][0];
//                   //if(i == j && a == b){
//                   //   energy += MO_fock->pointer()[ai][ai]*fabs(cis_trans->pointer()[num][Hbj]);
//                   //   energy -= MO_fock->pointer()[i][i]*fabs(cis_trans->pointer()[num][Hbj]);
//                   //}
//                   energy += cis_matrix->pointer()[Hai][Hbj]*cis_trans->pointer()[Hbj][Hai];
//                   //energy += (2*MO_eri_test[i+offset][ai+offset][j+offset][bj+offset])/Lfac*fabs(cis_trans->pointer()[num][Hbj]);
//                   //energy -= MO_eri_test[i+offset][j+offset][ai+offset][bj+offset]/Lfac*fabs(cis_trans->pointer()[num][Hbj]);
//
//
//                }
//             }
//           //for(int num = 0; num < cis_nsopi[0][0]*cis_nsopi[0][1]; num++){
//             if(cis_trans->pointer()[Hai][Hai] >= 0.00000000001){
//             there = false;
//             for(int loop = 0; loop < new_nsopi[0]; loop++){
//                if(fabs(tmp_vector[loop] - energy/cis_trans->pointer()[Hai][Hai]) <= 0.000001){
//                   there = true;
//                }
//             }
//             //if(!there){
//                tmp_vector[other_num] = energy/cis_trans->pointer()[Hai][Hai];
//                printf("energy: %f dipole: %f\n", energy/cis_trans->pointer()[Hai][Hai]*27.211, fabs(MO_dipole->pointer()[i][ai]));
//                other_num++;
//             //}
//             }
//           //}
//          }
//       }
//    }//}
    


    //cis_matrix->dsyev(cis_trans,cis_eval);
    //cis_matrix->print();
    outfile->Printf("cis eval before dipole\n");
    cis_eval->scale(27.211);
    cis_eval->print();
    exit(1);
    c_p = cis_trans->pointer();
    for(int i = 0; i < new_nsopi[0]; i++){
       for(int j = 0; j < new_nsopi[0]; j++){
          for(int theta = 0; theta < new_nsopi[0]; theta++){
             for(int lambda = 0; lambda < new_nsopi[0]; lambda++){
                cis_diagonal->pointer()[i][j] += cis_matrix->pointer()[theta][lambda] * c_p[theta][i] * c_p[lambda][j];
             }
          }
       }
    }
    //F_re->diagonalize(Ca,Feval);
    cis_trans->print();
    
    exit(1);

    //multiplying CIS results by dipole moment
    for(int h = 0; h < Jell->nirrep_; h++){
       double * cis_ptr = cis_eval->pointer(h);
       for(int i = 0; i < cis_nsopi[h][0]; i++){
          for(int j = 0; j < cis_nsopi[h][1]; j++){
             int offset = 0;
             for(int k = 0; k < h; k++){
                 offset += Jell->nsopi_[k];
             }
             printf("%d %f %f %f\n",i+offset,0.1000,cis_eval->pointer()[i*cis_nsopi[h][1]+j],(MO_dipole_ptr[i][cis_nsopi[h][0] + j]));
             //cis_ptr[i] *= MO_dipole_ptr[Jell->MO[offset+i/(Jell->nsopi_[h]-Jell->Eirrep_[h])][0]][Jell->MO[offset+Jell->Eirrep_[h]+i%(Jell->nsopi_[h]-Jell->Eirrep_[h])][0]];
             }
       }
    }
    //for(int i = 0; i < cis_nsopi[h][0]*cis_nsopi[h][1]; i++){
   // }
    //TODO make this an input option
    exit(1);
   
//TODO Clean up all memory used for CIS
//TODO Make CIS an input option
    }
    //exit(1);

//TODO Print a message stating units and that RTTDHF has started
    //Start of RTTDHF
    double * Vre = (double*)malloc((int)(time_length/time_step)*sizeof(double));
    double * Vfield = (double*)malloc((int)(time_length/time_step)*sizeof(double));

    if(options.get_bool("FOURIER")){
       outfile->Printf("\n");
       outfile->Printf( "    ****************************************************\n");
       outfile->Printf( "    *                                                  *\n");
       outfile->Printf( "    *   Jellium Real-Time Time-Dependent Hartree-Fock  *\n");
       outfile->Printf( "    *                                                  *\n");
       outfile->Printf( "    ****************************************************\n");
       outfile->Printf("\n");

       outfile->Printf("        Time (a.u.)           Dipole moment        Real Density rms     Imaginary Density rms     External field\n");
       while(iter<(time_length/time_step)){	

           //start of RT-TDHF
           rk_step(density_re, density_im, iter*time_step);

           //to reset ext_field
           pulse(iter*time_step);

           // evaluate dipole moment
           double dip = 0.0;
           double dip_im = 0.0;
           int offset = 0;
          
           for(int h = 0; h < Jell->nirrep_; h++){
               for(int i = 0; i < Jell->nsopi_[h]; i++){
                   for(int j = 0; j < Jell->nsopi_[h]; j++){
                       dip += density_re->pointer(h)[i][j] * dipole(Jell->MO[offset+i], Jell->MO[offset+j], boxlength);
                   }
               }
               offset += Jell->nsopi_[h];
           }
           outfile->Printf("%20.12lf   %20.12lf   %20.12lf   %20.12lf   %20.12lf\n",iter * time_step,dip,density_re->rms(),density_im->rms(),ext_field_);

           Vre[iter] = dip;
           Vfield[iter] = ext_field_;
           //since only propagating in the x direction psi(i) psi(j) is integrated over x
           iter++;
       }
       //Prints frequency data
       fourier(Vre, Vfield);
    }
    // Typically you would build a new wavefunction and populate it with data

    return ref_wfn;
}

double dipole(int * n, int * m, double L){
   
        //grab first set of quantum numbers
        int nx = n[0];
        int ny = n[1];
        int nz = n[2];

        //grab second set of quantum numbers
        int mx = m[0];
        int my = m[1];
        int mz = m[2];

        //Dipole function is only non-zero if the y and z quantum numbers are equal
        if(my != ny){
           return 0.0;
        }
        if(mz != nz){
           return 0.0;
        }
        if((mx+nx)%2!=1){
           return 0.0;
        }else if(nx == mx){
           return 0.0;
        }
        else {
           return abs((((L)/(M_PI*M_PI))*((nx*mx)/pow((nx*nx-mx*mx),2))));
           //return (8*L/(M_PI*M_PI))*((n*m)/pow((n*n-m*m),2));
           //return 1.0;
        }
}

double pulse(double time){
    // Gaussian pulse
    ext_field_ = laser_amp * exp(-((time-1.5*laser_time)*(time-1.5*laser_time)) / (0.3606738*laser_time*laser_time)) * sin(laser_freq*time);

    // continuous pulse
    //ext_field_ = laser_amp * sin(laser_freq*time);

    return ext_field_;
}
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
    std::shared_ptr<Matrix> J_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> K (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
    std::shared_ptr<Matrix> K_im (new Matrix(Jell->nirrep_,Jell->nsopi_,Jell->nsopi_));
          #pragma omp parallel for
          for (int hp = 0; hp < Jell->nirrep_; hp++) {
              int offp = 0;
              double ** k_p = K->pointer(hp);
              double ** j_p = J->pointer(hp);
              double ** jim_p = J_im->pointer(hp);
              double ** kim_p = K_im->pointer(hp);
              for (int myh = 0; myh < hp; myh++) {
                  offp += Jell->nsopi_[myh];
              }
              for (int p = 0; p < Jell->nsopi_[hp]; p++) {
                  int pp = p + offp;

                  for (int q = p; q < Jell->nsopi_[hp]; q++) {
                      int qq = q + offp;

                      double myJ = 0.0;
                      double myJim = 0.0;
                      double myK = 0.0;
                      double myKim = 0.0;
                      for (int hr = 0; hr < Jell->nirrep_; hr++) {
                          double ** d_p = d_re->pointer(hr);
                          double ** dim_p = d_im->pointer(hr);

                          int offr = 0;
                          for (int myh = 0; myh < hr; myh++) {
                              offr += Jell->nsopi_[myh];
                          }

                          for (int r = 0; r < Jell->nsopi_[hr]; r++) {
                              int rr = r + offr;
                              for (int s = 0; s < Jell->nsopi_[hr]; s++) {
                                  int ss = s + offr;
                                  myJ += d_p[r][s] * Jell->ERI_int(pp,qq,rr,ss);
                                  myJim += dim_p[r][s] * Jell->ERI_int(pp,qq,rr,ss);
                                  double eri = Jell->ERI_int(pp,ss,rr,qq);
                                  myK += d_p[r][s] * eri;
                                  myKim += dim_p[r][s] * eri;
                              }
                          }
                      }
                      j_p[p][q] = myJ;
                      j_p[q][p] = myJ;
                      jim_p[p][q] = myJim;
                      jim_p[q][p] = -myJim;
                      k_p[p][q] = myK;
                      k_p[q][p] = myK;
                      kim_p[p][q] =  myKim;
                      kim_p[q][p] = -myKim;
                  }
              }
          }
        F_re->copy(J);
        F_re->scale(2.0);
        F_re->subtract(K);
        F_im->copy(J_im);
        F_im->scale(2.0);
        F_im->subtract(K_im);
        F_im->scale(1.0/Lfac);
        F_re->scale(1.0/Lfac);
        F_re->add(h);

        int offset = 0;
        for(int h = 0; h < Jell->nirrep_; h++){
            double ** F_re_p = F_re->pointer(h);
            for(int i = 0; i < Jell->nsopi_[h]; i++){
               for(int j = 0; j < Jell->nsopi_[h]; j++){
                   //F_re_p[i][j] += dipole(Jell->MO[offset+i][0],Jell->MO[offset+j][0],boxlength)*pulse(time);
                   F_re_p[i][j] -= dipole(Jell->MO[offset+i], Jell->MO[offset+j], boxlength)*pulse(time);
               }
            }
            offset += Jell->nsopi_[h];
        }

}

void fourier(double* Vre, double* Vfield){
     double dampening = 5/time_length;
     //double min_freq = 400.0;
     //double max_freq = 700.0;
     double delta_freq = 0.05;
     int N = (int)time_length/time_step;
     for(int i = 0; i < (int)time_length/time_step; i++){
         double t = i*time_step;
         Vre[i] *= exp(-t/500);
         Vfield[i] *= exp(-t/500);
     }
     outfile->Printf("\t Starting Fourier transform...\n\n");
    
    double max_freq = 5.0;
    double dw = 0.000001;
    double w = 0.0;

    while ( w * 27.21138 < max_freq ) {

        w  += 2.0 * M_PI * dw;

        double dip_r = 0.0;
        double dip_i = 0.0;
        double ext_r = 0.0;
        double ext_i = 0.0;
        for (int i = 0; i < N; i++) {
            double t = i * time_step;
            dip_r += cos(w * t) * Vre[i];
            dip_i += sin(w * t) * Vre[i];
            ext_r += cos(w * t) * Vfield[i];
            ext_i += sin(w * t) * Vfield[i];
        }

        if (w*27.21138>max_freq) break;
        printf("%20.12f %20.12f\n",1240.0/(w*27.21138),(dip_r*dip_r + dip_i*dip_i)/(ext_r*ext_r+ext_i*ext_i));
    }
}
}} // End namespaces


