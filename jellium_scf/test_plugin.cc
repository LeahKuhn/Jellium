/*
 * @BEGIN LICENSE
 *
 * jellium_scf by Psi4 Developer, a plugin to:
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

#include "JelliumIntegrals.h"

#include<iostream>
#include<fstream>
#include<string>

namespace psi{ namespace jellium_scf {

extern "C"
int read_options(std::string name, Options& options)
{
    if (name == "JELLIUM_SCF"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
        /*- The number of grid points -*/
        options.add_int("N_GRID_POINTS", 20);
    }

    return true;
}

extern "C"
SharedWavefunction jellium_scf(SharedWavefunction ref_wfn, Options& options)
{
    int print = options.get_int("PRINT");
    // evaluate jellium integrals
    std::shared_ptr<JelliumIntegrals> ints (new JelliumIntegrals(options));
    
    std::ifstream Nuc, Self, Kin, ERI;
    Nuc.open("test/NucAttraction_ref.dat");
    Self.open("test/SelfEnergy_ref.dat");
    Kin.open("test/Kinetic_ref.dat");
    ERI.open("test/ERI_ref.dat");
    std::string tmp;
    bool Nuc_test = true, Self_test = true, Kin_test = true, ERI_test = true;
    
    std::getline(Self,tmp);
    if(fabs(std::stod(tmp)-ints->selfval) > 0.00000001)
       Self_test = false;
    for(int i = 0; i < 26; ++i){
       for(int j = i; j < 26; ++j){
          std::getline(Kin,tmp);
          if(fabs(std::stod(tmp)-ints->Ke->pointer()[i][j]) > 0.00000001)
             Kin_test = false;
          std::getline(Nuc,tmp);
          if(fabs(std::stod(tmp)-ints->NucAttrac->pointer()[i][j]) > 0.00000001)
             Nuc_test = false;
          for(int k = 0; k < 26; ++k){
             for(int l = k; l < 26; ++l){
             std::getline(ERI,tmp);
             if(fabs(std::stod(tmp)-ints->ERI_int(i,j,k,l)) > 0.00000001)
                ERI_test = false;
             }
          }
       }
    }
    std::cout << "Test complete" << std::endl;
    std::cout << "Self Energy:        ";
    if(Self_test)
       std::cout << "Passed" << std::endl;
    else
       std::cout << "Failed" << std::endl;
  
    std::cout << "Kinetic Energy:     ";
    if(Kin_test)
       std::cout << "Passed" << std::endl;
    else
       std::cout << "Failed" << std::endl;
   
    std::cout << "Nuclear Attraction: ";
    if(Nuc_test)
       std::cout << "Passed" << std::endl;
    else
       std::cout << "Failed" << std::endl;
   
    std::cout << "ERI:                ";
    if(ERI_test)
       std::cout << "Passed" << std::endl;
    else
       std::cout << "Failed" << std::endl;

             
    //// Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}

}} // End namespaces

