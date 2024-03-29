/*
 *@BEGIN LICENSE
 *
 * myscf, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Copyright (c) 2014, The Florida State University. All rights reserved.
 * 
 *@END LICENSE
 *
 */

#ifndef DIIS_H
#define DIIS_H

namespace psi{

class DIIS {

public:

    DIIS(int n, int diis_max);
    ~DIIS();

    /// write Fock matrix disk
    void WriteVector(double * vector);

    /// write error vector to disk
    void WriteErrorVector(double * vector);

    /// perform diis extrapolation
    void Extrapolate(double * vector);

    /// check that emax is below a threshold
    bool Threshold_Check();
private:

    /// temporary storage
    double * tmp1_;
    double * tmp2_;

    /// determine diis expansion coefficients
    void DIISCoefficients(int nvec);

    /// maximum number of diis vectors
    int maxdiis_;          

    /// dimension of each diis vector
    double * diisvec_;      

    /// dimension of each diis vector
    int dimdiis_;           

    /// current number of diis vectors
    int diis_iter_;         

    /// diis vector to be replaced
    int replace_diis_iter_;

    /// threshold to start diis procedure
    double threshold;
};

} // end of namespace

#endif
