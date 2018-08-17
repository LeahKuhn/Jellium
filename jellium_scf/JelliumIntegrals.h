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
    int* Eirrep_;
    double selfval = 0.0;
    double ERI_int(int a, int b, int c, int d);
    int* nsopi_;
    int get_nmax();
    int nirrep_;
    double* grid_points;
    std::shared_ptr<Vector> sqrt_tensor;
    std::shared_ptr<Vector> g_tensor;
    int ** MO;
    double* w;
    double pq_int_new(int dim, int px, int py, int pz, int qx, int qy, int qz);
  private:
    void compute();
    double**** ERI_mem;
    void Orderirrep(int &norbs, double *E, int **MO, int electrons);
    double n_order_;
    int electrons;
    int *** PQmap;
    int orbitalMax;
    int nmax = 0;
    double pi;
    /// Options object
    Options & options_;
    //  Electron integral functions
    double ERI(int dim, double *xa, double *w, int *a, int *b, int *c, int *d);
    double g_pq(int p, int q, double r);
    double pq_int(int dim, double *x, double *w, int px, int py, int pz, int qx, int qy, int qz);
    double E0_Int(int dim, double *xa, double *w);
    double Vab_Int(int dim, double *xa, double *w, int *a, int *b);
    double Vab_Int_new(int dim, double *xa, double *w, int *a, int *b);

    double ERI_unrolled(int * a, int * b, int * c, int * d, double ** PQ, int *** PQmap);
    double ERI_new(int * a, int * b, int * c, int * d, double ** PQ, int *** PQmap);
    
    void OrderPsis3D(int &norbs, double *E, int **MO);
    int **MAT_INT(int dim1, int dim2);
};

}}

#endif
