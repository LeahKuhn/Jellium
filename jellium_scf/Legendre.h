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

#ifndef LEGENDRE_H
#define LEGENDRE_H
class Legendre {
    public:
       void legendre_compute_glr ( int n, double x[], double w[] );
       void legendre_compute_glr0 ( int n, double *p, double *pp );
       void legendre_compute_glr1 ( int n, double *roots, double *ders );
       void legendre_compute_glr2 ( double p, int n, double *roots, double *ders );
       void rescale ( double a, double b, int n, double x[], double w[] );
       double rk2_leg ( double t, double tn, double x, int n );
       double ts_mult ( double *u, double h, int n );
};
#endif
