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
#include <omp.h>
#include "psi4/pragma.h"
#include"JelliumIntegrals.h"
namespace psi{ namespace jellium_scf {

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "jellium_scf"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
        options.add_int("N_GRID_POINTS", 0);
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction jellium_scf(SharedWavefunction ref_wfn, Options& options)
{
    int print = options.get_int("PRINT");
    double Lfac = pow(2.,(1./3))*(3.14159/2); //this is copied from JCIP.c not sure if this is gonna work but we'll see

 
    //grabbing one-electon integrals from mintshelper
    std::shared_ptr<JelliumIntegrals> Jell (new JelliumIntegrals(options));
    //one-electron kinetic energy integrals
    std::shared_ptr<Matrix> T = Jell->Ke;
    //T->scale(1/(Lfac));
    int nso = 26; 
    //one-electron potential energy integrals
    std::shared_ptr<Matrix> V = Jell->NucAttrac;
    //V->scale(1/(Lfac));
    //overlap integrals
    std::shared_ptr<Matrix> S = (std::shared_ptr<Matrix>)(new Matrix(nso,nso));
    for(int i = 0; i < nso; ++i){
       S->pointer()[i][i] = 1;
    }
    //build the core hamiltonian
    std::shared_ptr<Matrix> h = (std::shared_ptr<Matrix>)(new Matrix(T));
    
    //get molecule from the wavefunction that was passed into the plugin
    std::shared_ptr<Molecule> mol = ref_wfn->molecule();
    
    //get primary basis
    std::shared_ptr<BasisSet> primary = ref_wfn->get_basisset("ORBITAL");
    
    //total number of basis functions
    //int nso = primary->nbf();
    
    //get auxiliary basis
    //std::shared_ptr<BasisSet> auxiliary = ref_wfn->get_basisset("DF_BASIS_SCF");
    
    // total number of auxiliary basis functions
    //int nQ = auxiliary->nbf();
    
    //getting number of electrons
    int charge = mol->molecular_charge();
    int nelectron = 0;
    for(int i = 0; i < mol->natom(); ++i){
    	nelectron += (int)mol->Z(i);
    } 
    nelectron -= charge;
    
    if ( nelectron % 2 != 0){
    	throw PsiException("this will only worked with closed shells",__FILE__,__LINE__);
    }
    nelectron = 2;
    //double occupied orbitals
    int na = nelectron / 2;	
    
    for(int i = 0; i < nelectron; ++i){
    h->add(V);
    }
    h->print();
    //setting memory for SO->MO coefficients
    std::shared_ptr<Matrix> Ca = (std::shared_ptr<Matrix>)(new Matrix(nso,nso));
    
    //construct the three-index intregrals
    //since we want the SO-basis integrals, it is fine to pass empty Ca matrix
    //similarly, the number of active vs inactive orbitals isn't really important here
    //std::shared_ptr<DFTensor> DF (new DFTensor(primary,auxiliary,Ca,na,nso-na,na,nso-na,options));
    //std::shared_ptr<Matrix> Qso = DF->Qso();
    
    //allocate memory for eigenvectors and eigenvalues of the overlap matrix
    std::shared_ptr<Matrix> Sevec (new Matrix(nso,nso));
    std::shared_ptr<Vector> Seval (new Vector(nso));
    
    //build S^(-1/2) symmetric orthogonalization matrix
    S->diagonalize(Sevec,Seval);
    
    std::shared_ptr<Matrix> Shalf = (std::shared_ptr<Matrix>)( new Matrix(nso,nso));	
    for (int mu = 0; mu < nso; ++mu){
    	Shalf->pointer()[mu][mu] = 1.0 / sqrt(Seval->pointer()[mu]);
    }
    
    //transform Seval back to nonorthogonal basis
    Shalf->back_transform(Sevec);
    //Shalf->print();    
    std::shared_ptr<Matrix> F = (std::shared_ptr<Matrix>)(new Matrix(h));
    //F->back_transform(Shalf);
    std::shared_ptr<Matrix> Firstevec (new Matrix(nso,nso));
    std::shared_ptr<Vector> Firstevac (new Vector(nso));
    F->diagonalize(Firstevec,Firstevac);
    std::shared_ptr<Matrix> Cafirst (new Matrix(nso,nso));
    std::shared_ptr<Matrix> Dfirst (new Matrix(nso,nso));
    Ca->gemm(false,false,1.0,Shalf,Firstevec,0.0);
   // #pragma omp parallel for schedule (static) num_threads(6)
    for(int u = 0; u < nso; ++u){
   //     printf("thread %i", omp_get_thread_num());
        for(int v = 0; v< nso; ++v){
                for(int i = 0; i < na; ++i){
                        Dfirst->pointer()[u][v] += 2*(nelectron/Lfac)*Ca->pointer()[u][i] * Ca->pointer()[v][i];
                }
        }
    }
    double first_energy = 0.0;
    double e_nuc = Jell->selfval;
    //first_energy += Dfirst->vector_dot(h);
    first_energy += nelectron*nelectron/2*Jell->selfval;
    for(int a = 0; a < nso; ++a){
       for(int b = 0; b < nso; ++b){
          first_energy += Dfirst->pointer()[a][b]*(h->pointer()[a][b]);
       }
    }
    for(int a = 0; a < nso; ++a){
       for(int b = 0; b < nso; ++b){
          for(int c = 0; c < nso; ++c){
             for(int d = 0; d < nso; ++d){
                first_energy += 0.5*(Dfirst->pointer()[a][b]*Dfirst->pointer()[c][d]-0.5*(Dfirst->pointer()[a][d]*Dfirst->pointer()[b][c]))*(Jell->ERI_int(a,b,c,d));
             }
          }
       }
    }

    F->copy(h);
    outfile->Printf("guess energy %f \n", first_energy);
    double e_convergence = options.get_double("E_CONVERGENCE");
    double d_convergence = options.get_double("D_CONVERGENCE");
    int Iter = 0;
    outfile->Printf("Iter \t energy \t energy difference \t density difference \n");
    do{
//    std::shared_ptr<Matrix> Fprime = (std::shared_ptr<Matrix>)(new Matrix(F));
//    Fprime->back_transform(Shalf);
//    std::shared_ptr<Matrix> Fevec (new Matrix(nso,nso));
//    std::shared_ptr<Vector> Fevac (new Vector(nso));
//    Fprime->diagonalize(Fevec,Fevac);
//    
//    
//    Ca->gemm(false,false,1.0,Shalf,Fevec,0.0);
//    //Ca->print();	
//    
//    //building density matrix
//    
//    std::shared_ptr<Matrix> D (new Matrix(nso,nso));
//    
//    for(int u = 0; u < nso; ++u){
//    	for(int v = 0; v< nso; ++v){
//    		for(int i = 0; i < na; ++i){
//    			D->pointer()[u][v] += Ca->pointer()[u][i] * Ca->pointer()[v][i];			
//    		}
//    	}
//    }
//	D->print();
    
//    std::shared_ptr<Matrix> J (new Matrix(nso,nso));
//    std::shared_ptr<Vector> Iq (new Vector(nQ));
//    for (int Q = 0; Q < nQ; ++Q){
//        double dum = 0.0;
//        for (int i = 0; i < nso; ++i){
//            for (int j = 0; j < nso; ++j){
//                dum += Dfirst->pointer()[i][j] * Qso->pointer()[Q][i*nso+j];	
//            }
//        }
//        Iq->pointer()[Q] = dum;
//    }
//    
//    //std::shared_ptr<Matrix> Qmo = DF->Qmo();
//    //std::shared_ptr<Matrix> K (new Matrix(nso,nso));
//    for(int i = 0; i < nso; ++i){
//        for(int j = 0; j < nso; ++j){
//            double dum = 0.0;
//            for(int Q = 0; Q < nQ; ++Q){
//                dum += Iq->pointer()[Q] * Qso->pointer()[Q][i*nso+j];
//            }
//            J->pointer()[i][j] =  dum;
//        }
//    }
//
//
//    std::shared_ptr<Vector> IvoQ (new Vector(nso*nso*nQ));
//    for(int nu = 0; nu < nso; ++nu){
//        for(int sigma = 0; sigma < nso; ++sigma){
//            for(int Q = 0; Q < nQ; ++Q){
//                double dum = 0.0;
//                for(int lambda = 0; lambda < nso; ++lambda){
//                    dum += Dfirst->pointer()[lambda][sigma]*Qso->pointer()[Q][lambda*nso+nu];
//                }
//                IvoQ->pointer()[nu*nso*nQ+sigma*nQ+Q] = dum;
//           }
//        }
//    }
//                
//    std::shared_ptr<Matrix> K (new Matrix(nso,nso));
//    for(int mu = 0; mu < nso; ++mu){
//        for(int nu = 0; nu < nso; ++nu){
//            double dum = 0.0;
//            for(int sigma = 0; sigma < nso; ++sigma){
//                for(int Q = 0; Q < nQ; ++Q){
//                    dum += IvoQ->pointer()[nu*nso*nQ+sigma*nQ+Q] * Qso->pointer()[Q][mu*nso+sigma];
//                }
//            }
//            K->pointer()[mu][nu] -= dum;
//        }
//    }
    //K->print();
    std::shared_ptr<Matrix> Fa (new Matrix(h));
    for(int a = 0; a < nso; ++a){
       for(int b = 0; b < nso; ++b){
          for(int c = 0; c < nso; ++c){
             for(int d = 0; d < nso; ++d){
                Fa->pointer()[a][b] += Dfirst->pointer()[c][d]*(Jell->ERI_int(a,b,c,d)-0.5*Jell->ERI_int(a,d,b,c)); 
             }
          }
       }
    }
//    Fa->add(h);
//    Fa->add(J);
//    Fa->add(J);
//    Fa->add(K);
    F->copy(Fa);
    //Fa->print();
    double energy = 0.0;
//    double e_nuc = Jell->selfval;
//    energy += Dfirst->vector_dot(Fa);
//    energy += Dfirst->vector_dot(h);
//    energy += e_nuc;
    double e_nuc = Jell->selfval;
    //first_energy += Dfirst->vector_dot(h);
    energy += (nelectron*nelectron/2)*e_nuc;
    for(int a = 0; a < nso; ++a){
       for(int b = 0; b < nso; ++b){
          energy += Dfirst->pointer()[a][b]*(h->pointer()[a][b]);
       }
    }
    for(int a = 0; a < nso; ++a){
       for(int b = 0; b < nso; ++b){
          for(int c = 0; c < nso; ++c){
             for(int d = 0; d < nso; ++d){
                energy += 0.5*(Dfirst->pointer()[a][b]*Dfirst->pointer()[c][d]-0.5*(Dfirst->pointer()[a][d]*Dfirst->pointer()[b][c]))*(Jell->ERI_int(a,b,c,d));
             } 
          }
       }
    }

    double density_diff = 0.0;
    std::shared_ptr<Matrix> Fprime = (std::shared_ptr<Matrix>)(new Matrix(F));
    Fprime->back_transform(Shalf);
    std::shared_ptr<Matrix> Fevec (new Matrix(nso,nso));
    std::shared_ptr<Vector> Fevac (new Vector(nso));
    Fprime->diagonalize(Fevec,Fevac);


    Ca->gemm(false,false,1.0,Shalf,Fevec,0.0);
    //Ca->print();      

    //building density matrix

    std::shared_ptr<Matrix> D (new Matrix(nso,nso));
    double tmp = 0;
    for(int u = 0; u < nso; ++u){
        for(int v = 0; v< nso; ++v){
                for(int i = 0; i < na; ++i){
                        D->pointer()[u][v] += (nelectron/Lfac)*Ca->pointer()[u][i] * Ca->pointer()[v][i];
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
        F->print();
	return ref_wfn;
    }
    }while(1);
	//while(e_convergence > echange && d_convergence > dchange);
//	IvoQ->print();

//        for (int nu = 0; nu < nso; nu++) {
//            for (int sig = 0; sig < nso; sig++) {
//                for (int Q = 0; Q < nQ; Q++) {
//                    double dum = 0.0;
//                    for (int lam = 0; lam < nso; lam++) {
//                        dum += D->pointer()[lam][sig] * Qso->pointer()[Q][lam*nso+nu];
//                    }
//                    IvoQ->pointer()[nu*nso*nQ+sig*nQ+Q] =dum;
//              }
//            }
//        }
//	IvoQ->print();
    // Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}

}} // End namespaces

