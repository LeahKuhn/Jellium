/*
 *@BEGIN LICENSE
 *
 * jellium_scf, a plugin to:
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

#ifndef JELLIUM_INTEGRALS_H
#define JELLIUM_INTEGRALS_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include <psi4/libplugin/plugin.h>
#include <psi4/psi4-dec.h>
#include <psi4/liboptions/liboptions.h>
#include <psi4/libpsio/psio.hpp>
#include <psi4/libqt/qt.h>

#include <psi4/libmints/wavefunction.h>
#include <psi4/libmints/matrix.h>
#include <psi4/libmints/vector.h>

namespace psi{ namespace jellium_scf {

class JelliumIntegrals{

  public:
    JelliumIntegrals(Options & options);
    ~JelliumIntegrals();

    void common_init();
    //Integral matricies
    std::shared_ptr<Matrix> NucAttrac;
    std::shared_ptr<Matrix> Ke;
    std::shared_ptr<Matrix> PQ;
    double selfval = 0.0;
    double ERI_int(int a, int b, int c, int d);

  private:
    void compute();
    std::shared_ptr<Vector> sqrt_tensor;
    std::shared_ptr<Vector> g_tensor;
    double n_order_;
    int ** MO;
    int *** PQmap;
    int orbitalMax = 26;
    /// Options object
    Options & options_;
    //  Electron integral functions
    double ERI(int dim, double *xa, double *w, double *a, double *b, double *c, double *d);
    double g_pq(double p, double q, double r);
    double pq_int(int dim, double *x, double *w, double px, double py, double pz, double qx, double qy, double qz);
    double E0_Int(int dim, double *xa, double *w);
    double Vab_Int(int dim, double *xa, double *w, double *a, double *b);

    double ERI_new(std::shared_ptr<Vector> a, std::shared_ptr<Vector> b, std::shared_ptr<Vector> c, std::shared_ptr<Vector> d, double ** PQ, int *** PQmap);
    double pq_int_new(int dim, int px, int py, int pz, int qx, int qy, int qz, std::shared_ptr<Vector> g_tensor,int orbitalMax, std::shared_ptr<Vector> sqrt_tensor);
    
    void OrderPsis3D(int norbs, int *E, int **MO);
    int **MAT_INT(int dim1, int dim2);
    int *VEC_INT(int dim);
};

}}

#endif
