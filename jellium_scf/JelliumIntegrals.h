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

#ifndef V2RDM_SOLVER_H
#define V2RDM_SOLVER_H

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

    void compute();

  private:

    double n_order_;

    /// Options object
    Options & options_;

    // Gauss-Legendre quadrature functions
    void legendre_compute_glr ( int n, double x[], double w[] );
    void legendre_compute_glr0 ( int n, double *p, double *pp );
    void legendre_compute_glr1 ( int n, double *roots, double *ders );
    void legendre_compute_glr2 ( double p, int n, double *roots, double *ders );
    void legendre_handle ( int n, double a, double b );
    void r8mat_write ( char *output_filename, int m, int n, double table[] );
    void rescale ( double a, double b, int n, double x[], double w[] );
    double rk2_leg ( double t, double tn, double x, int n );
    void timestamp ( void );
    double ts_mult ( double *u, double h, int n );
    double wtime ( );

    //  Electron integral functions
    double ERI(int dim, double *xa, double *w, double *a, double *b, double *c, double *d);
    double ERI_new(int dim, double *xa, double *a, double *b, double *c, double *d, double * g_tensor, int orbitalMax, double * sqrt_tensor, double ** PQ, int *** PQmap);
    double g_pq(double p, double q, double r);
    double pq_int(int dim, double *x, double *w, double px, double py, double pz, double qx, double qy, double qz);
    double pq_int_new(int dim, int px, int py, int pz, int qx, int qy, int qz, double * g_tensor,int orbitalMax, double * sqrt_tensor);
    double E0_Int(int dim, double *xa, double *w);
    double Vab_Int(int dim, double *xa, double *w, double *a, double *b);
    
    void OrderPsis3D(int norbs, int *E, int **MO);
    int **MAT_INT(int dim1, int dim2);
    int *VEC_INT(int dim);

};

}}

#endif
