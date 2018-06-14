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

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>

#include <psi4/libplugin/plugin.h>
#include <psi4/psi4-dec.h>
#include <psi4/liboptions/liboptions.h>
#include <psi4/libqt/qt.h>

#include <psi4/libpsi4util/process.h>
#include <psi4/libpsi4util/PsiOutStream.h>
#include <psi4/libpsio/psio.hpp>
#include <psi4/libmints/wavefunction.h>
#include <psi4/psifiles.h>
#include <psi4/libpsio/psio.hpp>
#include <psi4/libmints/vector.h>
#include <psi4/libmints/matrix.h>

#include "JelliumIntegrals.h"
#include "Legendre.h"
using namespace psi;

namespace psi{ namespace jellium_scf {

JelliumIntegrals::JelliumIntegrals(Options & options):
        options_(options)
{
    orbitalMax = options.get_int("N_BASIS_FUNCTIONS");
    length = options.get_int("LENGTH");
    common_init();
    compute();
}

// free memory here
JelliumIntegrals::~JelliumIntegrals()
{
}

void JelliumIntegrals::common_init() {

    outfile->Printf("\n");
    outfile->Printf("\n");
    outfile->Printf( "        ****************************************************\n");
    outfile->Printf( "        *                                                  *\n");
    outfile->Printf( "        *    Jellium Hartree-Fock                          *\n");
    outfile->Printf( "        *                                                  *\n");
    outfile->Printf( "        ****************************************************\n");
    outfile->Printf("\n");
    outfile->Printf("\n");
    x1 = (int *)malloc(3*sizeof(int));
    x2 = (int *)malloc(3*sizeof(int));
    y1 = (int *)malloc(3*sizeof(int));
    y2 = (int *)malloc(3*sizeof(int));
    z1 = (int *)malloc(3*sizeof(int));
    z2 = (int *)malloc(3*sizeof(int));
    
}

void JelliumIntegrals::compute() {

 //printf ( "\n" );
  //printf ( "LEGENDRE_RULE_FAST:\n" );
  //printf ( "  Normal end of execution.\n" );

  //printf ( "\n" );

  double a = 0.0;
  double b = 1.0;
  int n = options_.get_int("N_GRID_POINTS");
  double *x, *w;
  int *mu, *nu, *sig, *lam;

  x   = (double *)malloc(n*sizeof(double));
  w   = (double *)malloc(n*sizeof(double));
  
  mu  = (int*)malloc(3*sizeof(int));
  nu  = (int*)malloc(3*sizeof(int));
  sig  = (int*)malloc(3*sizeof(int));
  lam  = (int*)malloc(3*sizeof(int));

  nmax=3;

  std::shared_ptr<Vector> ORBE = std::shared_ptr<Vector>( new Vector(3*nmax*nmax*nmax));//VEC_INT(3*nmax*nmax*nmax);
  MO  = MAT_INT(3*nmax*nmax*nmax,3);

  OrderPsis3D(nmax, ORBE->pointer(), MO);
  Legendre tmp;
  //  Constructe grid and weights, store them to the vectors x and w, respectively.
  //  This is one of John Burkhardt's library functions
  tmp.legendre_compute_glr(n, x, w);

  // Scale the grid to start at value a and end at value b. 
  // We want our integration range to go from 0 to 1, so a = 0, b = 1
  // This is also one of John Burkhardt's library functions
  tmp.rescale( a, b, n, x, w);
  // build g tensor g[npq] * w[n]
  outfile->Printf("\n");
  outfile->Printf("    build g tensor......."); fflush(stdout);
  g_tensor = std::shared_ptr<Vector>( new Vector(n * orbitalMax * orbitalMax));
  for (int pt = 0; pt < n; pt++) {
      double xval = x[pt];
      for (int p = 0; p < orbitalMax; p++) {
          for (int q = 0; q < orbitalMax; q++) {
              g_tensor->pointer()[pt*orbitalMax*orbitalMax+p*orbitalMax+q] = g_pq(p, q, xval) * w[pt];
          }
      }
  }
  outfile->Printf("done.\n");
  // build sqrt(x*x+y*y+z*z)
  outfile->Printf("    build sqrt tensor...."); fflush(stdout);
  sqrt_tensor = std::shared_ptr<Vector>(new Vector(n*n*n));
  for (int i = 0; i < n; i++) {
      double xval = x[i];
      for (int j = 0; j < n; j++) {
          double yval = x[j];
          for (int k = 0; k < n; k++) {
              double zval = x[k];
              double val = sqrt(xval*xval+yval*yval+zval*zval);
              sqrt_tensor->pointer()[i*n*n + j*n + k] = 1.0/val;
          }
      }
  }
  outfile->Printf("done.\n");


  int start_pq = clock();
  // now, compute (P|Q)
  outfile->Printf("    build (P|Q).........."); fflush(stdout);
  PQmap = (int ***)malloc((2*nmax+2)*sizeof(int**));
  for (int i = 0; i < 2*nmax+2; i++) {
      PQmap[i] = (int **)malloc((2*nmax+2)*sizeof(int*));
      for (int j = 0; j < 2*nmax+2; j++) {
          PQmap[i][j] = (int *)malloc((2*nmax+2)*sizeof(int));
          for (int k = 0; k < 2*nmax+2; k++) {
              PQmap[i][j][k] = 999;
          }
      }
  }
  int Pdim = 0;
  for (int px = 0; px < 2*nmax+2; px++) {
      for (int py = 0; py < 2*nmax+2; py++) {
          for (int pz = 0; pz < 2*nmax+2; pz++) {
              PQmap[px][py][pz] = Pdim;
              Pdim++;
          }
      }
  }
  //printf("1, -2, -3, 4, 2, 0\n"); 
  //pq_int(orbitalMax, x, w, 1, -2, -3, 4, 2, 0);
  //exit(1);
  PQ = std::shared_ptr<Matrix>(new Matrix(Pdim,Pdim));
  Ke = std::shared_ptr<Matrix>(new Matrix(orbitalMax,orbitalMax));
  NucAttrac = std::shared_ptr<Matrix>(new Matrix(orbitalMax,orbitalMax));
  for (int px = 0; px < 2*nmax+2; px++) {
      for (int qx = px; qx < 2*nmax+2; qx++) {

          int pq_x = px*(2*nmax+2) + qx;

          for (int py = 0; py < 2*nmax+2; py++) {
              for (int qy = py; qy < 2*nmax+2; qy++) {

                  int pq_y = py*(2*nmax+2) + qy;
                  if ( pq_x > pq_y ) continue;

                  for (int pz = 0; pz < 2*nmax+2; pz++) {
                      for (int qz = pz; qz < 2*nmax+2; qz++) {

                          int pq_z = pz*(2*nmax+2) + qz;
                          if ( pq_y > pq_z ) continue;


                          //if ( P > Q ) continue;

                          double dum = pq_int_new(n, px, py, pz, qx, qy, qz, g_tensor,orbitalMax,sqrt_tensor);
//                         printf("dum %f",dum); 
                          int P,Q;

                          // start 
                          P = PQmap[px][py][pz];
                          Q = PQmap[qx][qy][qz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qx][py][pz];
                          Q = PQmap[px][qy][qz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[px][qy][pz];
                          Q = PQmap[qx][py][qz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qx][qy][pz];
                          Q = PQmap[px][py][qz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[px][py][qz];
                          Q = PQmap[qx][qy][pz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qx][py][qz];
                          Q = PQmap[px][qy][pz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[px][qy][qz];
                          Q = PQmap[qx][py][pz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qx][qy][qz];
                          Q = PQmap[px][py][pz];
                          PQ->pointer()[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[py][px][pz];
                          Q = PQmap[qy][qx][qz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[py][qx][pz];
                          Q = PQmap[qy][px][qz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qy][px][pz];
                          Q = PQmap[py][qx][qz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qy][qx][pz];
                          Q = PQmap[py][px][qz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[py][px][qz];
                          Q = PQmap[qy][qx][pz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[py][qx][qz];
                          Q = PQmap[qy][px][pz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qy][px][qz];
                          Q = PQmap[py][qx][pz];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qy][qx][qz];
                          Q = PQmap[py][px][pz];
                          PQ->pointer()[P][Q] = dum;

                          // now begins pxqx < pyqy < pzqz

                          P = PQmap[pz][px][py];
                          Q = PQmap[qz][qx][qy];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[pz][qx][py];
                          Q = PQmap[qz][px][qy];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[pz][px][qy];
                          Q = PQmap[qz][qx][py];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[pz][qx][qy];
                          Q = PQmap[qz][px][py];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qz][px][py];
                          Q = PQmap[pz][qx][qy];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qz][qx][py];
                          Q = PQmap[pz][px][qy];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qz][px][qy];
                          Q = PQmap[pz][qx][py];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qz][qx][qy];
                          Q = PQmap[pz][px][py];
                          PQ->pointer()[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[pz][py][px];
                          Q = PQmap[qz][qy][qx];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[pz][py][qx];
                          Q = PQmap[qz][qy][px];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[pz][qy][px];
                          Q = PQmap[qz][py][qx];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[pz][qy][qx];
                          Q = PQmap[qz][py][px];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qz][py][px];
                          Q = PQmap[pz][qy][qx];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qz][py][qx];
                          Q = PQmap[pz][qy][px];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qz][qy][px];
                          Q = PQmap[pz][py][qx];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qz][qy][qx];
                          Q = PQmap[pz][py][px];
                          PQ->pointer()[P][Q] = dum;

                          // now begins last set of 16

                          P = PQmap[px][pz][py];
                          Q = PQmap[qx][qz][qy];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qx][pz][py];
                          Q = PQmap[px][qz][qy];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[px][pz][qy];
                          Q = PQmap[qx][qz][py];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qx][pz][qy];
                          Q = PQmap[px][qz][py];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[px][qz][py];
                          Q = PQmap[qx][pz][qy];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qx][qz][py];
                          Q = PQmap[px][pz][qy];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[px][qz][qy];
                          Q = PQmap[qx][pz][py];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qx][qz][qy];
                          Q = PQmap[px][pz][py];
                          PQ->pointer()[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[py][pz][px];
                          Q = PQmap[qy][qz][qx];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[py][pz][qx];
                          Q = PQmap[qy][qz][px];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qy][pz][px];
                          Q = PQmap[py][qz][qx];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qy][pz][qx];
                          Q = PQmap[py][qz][px];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[py][qz][px];
                          Q = PQmap[qy][pz][qx];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[py][qz][qx];
                          Q = PQmap[qy][pz][px];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qy][qz][px];
                          Q = PQmap[py][pz][qx];
                          PQ->pointer()[P][Q] = dum;

                          P = PQmap[qy][qz][qx];
                          Q = PQmap[py][pz][px];
                          PQ->pointer()[P][Q] = dum;

                      }
                  }
              }
          }
      }
  }
                          //int P = PQmap[ 0 ][ 0 ][ 0 ];
                          //int Q = PQmap[ 0 ][ 0 ][ 0 ];
                          //double dum = PQ->pointer()[P][Q];
  int end_pq = clock();
  outfile->Printf("done.\n");fflush(stdout);
  outfile->Printf("\n");
  outfile->Printf("    time for (P|Q) construction: %12.6f\n",(double)(end_pq-start_pq)/CLOCKS_PER_SEC); fflush(stdout);
  outfile->Printf("\n");
  outfile->Printf("canonical integrals");

  // Four nested loops to compute lower triange of electron repulsion integrals - roughly have of the non-unique integrals
  // will not be computed, but this is still not exploiting symmetry fully
  outfile->Printf("    build ERIs...........");fflush(stdout);
  int start = clock();
  for (int i=0; i<orbitalMax; i++) {
      mu[0] = MO[i][0];
      mu[1] = MO[i][1];
      mu[2] = MO[i][2];

      for (int j=0; j<orbitalMax; j++) {
          nu[0] = MO[j][0];
          nu[1] = MO[j][1];
          nu[2] = MO[j][2];

          // Lower triangle of 1-electron integrals will be computed, fully exploiting symmetry (I think!)
          // Kinetic Energy Integrals - already computed and stored in ORBE vector    
          if (i==j) { 
              Ke->pointer()[i][j] = 0.5*ORBE->pointer()[i];
          }
          else {
              Ke->pointer()[i][j] = 0.0;
          }
          //printf("%f",kinval);
          // Print Kinetic Energy Integral to file
          // Nuclear-attraction Integrals
          double dum = Vab_Int(n, x, w, mu, nu);
          NucAttrac->pointer()[i][j] = dum;
          //NucAttrac->pointer()[j][i] = dum;

          // Print Nuclear-attraction integral to file

          // loop over indices for electron 2       
          for (int k=0; k<orbitalMax; k++) {
              lam[0] = MO[k][0];
              lam[1] = MO[k][1];
              lam[2] = MO[k][2];
              for (int l=k; l<orbitalMax; l++) {
                  sig[0] = MO[l][0];
                  sig[1] = MO[l][1];
                  sig[2] = MO[l][2];
   
                  // Compute 2-electron integral
                 // double erival = ERI_new(mu, nu, lam, sig, PQ->pointer(), PQmap);
                 // double dum = ERI(n, x, w, mu, nu, lam, sig);
                 // if ( fabs(erival - dum) > 1e-14 ) {
                 //     outfile->Printf("uh-oh. %20.12lf %20.12lf\n",erival,dum);
                 // }else {
                 //     printf("sweet! %20.12lf %20.12lf\n",erival,dum);
                 // }

                  // Print ERI to file
                  //fprintf(erifp," %i  %i  %i  %i  %17.14f\n",i+1,j+1,k+1,l+1,erival);
             }
          }
      }
  }
  // hey Danny, why isn't this tensor symmetric?
  //for (int i=0; i<orbitalMax; i++) {
  //    for (int j=0; j<=i; j++) {
  //        double dum1 = NucAttrac->pointer()[i][j];
  //        double dum2 = NucAttrac->pointer()[j][i];
  //        //NucAttrac->pointer()[i][j] = NucAttrac->pointer()[j][i] = 0.5 * ( dum1 + dum2 );
  //    }
  //}
  int end = clock();
  outfile->Printf("done.\n");fflush(stdout);
  outfile->Printf("\n");
  outfile->Printf("    time for eri construction:   %12.6f\n",(double)(end-start)/CLOCKS_PER_SEC); fflush(stdout);
  outfile->Printf("\n");

  // Compute self energy
  selfval = E0_Int(n, x, w);
  //outfile->Printf("%f",E0_Int(n, x, w));
  // Print to file
  //fprintf(selffp, "  %17.14f\n",selfval); 
  
  free(x);
  free(w);
  free(mu);
  free(nu);

}

double JelliumIntegrals::ERI_int(int a, int b, int c, int d){

   //int * lam = (int*)malloc(3 * sizeof(int));
   //int * sig = (int*)malloc(3 * sizeof(int));
   //int * nu  = (int*)malloc(3 * sizeof(int));
   //int * mu  = (int*)malloc(3 * sizeof(int));
   //for (int i = 0; i < 3; ++i){
   //    mu[i] = MO[a][i];
   //    nu[i] = MO[b][i];
   //    lam[i] = MO[c][i];
   //    sig[i] = MO[d][i];
   //}

   //std::shared_ptr<Vector> lam (new Vector(3));
   //std::shared_ptr<Vector> sig (new Vector(3));
   //std::shared_ptr<Vector> nu (new Vector(3));
   //std::shared_ptr<Vector> mu (new Vector(3));
   //for (int i = 0; i < 3; ++i){
   //    mu->pointer()[i] = MO[a][i];
   //    nu->pointer()[i] = MO[b][i];
   //    lam->pointer()[i] = MO[c][i];
   //    sig->pointer()[i] = MO[d][i];
   //}


   //mu->pointer()[0] = MO[a][0];
   //mu->pointer()[1] = MO[a][1];
   //mu->pointer()[2] = MO[a][2];
   //nu->pointer()[0] = MO[b][0];
   //nu->pointer()[1] = MO[b][1];
   //nu->pointer()[2] = MO[b][2];
   //lam->pointer()[0] = MO[c][0];
   //lam->pointer()[1] = MO[c][1];
   //lam->pointer()[2] = MO[c][2];
   //sig->pointer()[0] = MO[d][0];
   //sig->pointer()[1] = MO[d][1];
   //sig->pointer()[2] = MO[d][2];

   //double val = ERI_new(mu, nu, lam, sig, PQ->pointer(), PQmap);
   //double val = ERI_new(MO[a], MO[b], MO[c], MO[d], PQ->pointer(), PQmap);
   //free(mu);
   //free(nu);
   //free(lam);
   //free(sig);
   //return val;
   return ERI_new(MO[a], MO[b], MO[c], MO[d], PQ->pointer(), PQmap);
}

double JelliumIntegrals::g_pq(int p, int q, double x) {
  int d = (abs(p-q));
  double pi = M_PI;
  if(q < 0 || p < 0){
     return 0;
  }
  if (p == q && p == 0) {
    return 1.0 - x;
  }
  else if ( p == q && p > 0 ) {
    return (1.0 - x)*cos(p*pi*x)/2.0 - sin(p*pi*x)/(2*p*pi);
  }
  else if ( (d % 2)==0 && d!=0) {
    return (q*sin(q*pi*x) - p*sin(p*pi*x))/((p*p-q*q)*pi);
  }
  else 
    return 0.0;
}


// From Eq. 3.6 in the Peter Gill paper, 
// the Vab integrals are -1/pi^3 \int (phi_a phi_b )/(|r1 - r2|) dr1 dr2
// This essentially leads to (p|q) integrals with 
// px = a_x - b_x
// py = a_y - b_y
// pz = a_z - b_z
// qx = a_x + b_x
// qy = a_y + b_y
// qz = a_z + b_z
// Note the expansion of the trigonetric identity:
// Cos[px x1] Cos[py y1] Cos[pz z1] - Cos[qx x1] Cos[py y1] Cos[pz z1] - 
// Cos[px x1] Cos[qy y1] Cos[pz z1] + Cos[qx x1] Cos[qy y1] Cos[pz z1] -
// Cos[px x1] Cos[py y1] Cos[qz z1] + 
// Cos[qx x1] Cos[py y1] Cos[qz z1] + Cos[px x1] Cos[qy y1] Cos[qz z1] -
// Cos[qx x1] Cos[qy y1] Cos[qz z1]
// In order to be consistent with the defintiion of the (p|q) integrals, 
// the term Cos[px x1] Cos[py y1] Cos[pz z1] -> Cos[px x1] Cos[py y1] Cos[pz z1] Cos[0 x2] Cos[0 y2] Cos[0 z2]
// In terms of how the pq_int function is called for the above integral, it should be
// pq_int(dim, xa, w, px, py, pz, 0, 0, 0)

double JelliumIntegrals::Vab_Int(int dim, double *xa, double *w, int *a, int *b){
  int px, py, pz, qx, qy, qz;
  double Vab;
  px = a[0] - b[0];
  py = a[1] - b[1];
  pz = a[2] - b[2];
  

  qx = a[0] + b[0];
  qy = a[1] + b[1];
  qz = a[2] + b[2];

  Vab = 0.0;
 
  //make sure to check that these integrals are correct

  //printf("%f %f %f %f %f %f %f %f\n", pq_int(dim, xa, w, px, py, pz, 0, 0, 0), pq_int(dim, xa, w, 0,  py, pz, qx,0, 0), pq_int(dim, xa, w, px, 0, pz, 0, qy, 0), pq_int(dim, xa, w, 0, 0, pz, qx, qy, 0), pq_int(dim, xa, w, px, py, 0, 0, 0, qz), pq_int(dim, xa, w, 0, py, 0, qx, 0, qz), pq_int(dim, xa, w, px, 0, 0, 0, qy, qz), pq_int(dim, xa, w, 0, 0, 0, qx, qy, qz));

  // Cos[px x1] Cos[py y1] Cos[pz z1]
  Vab  += pq_int(dim, xa, w, px, py, pz, 0, 0, 0);
       
  // -  Cos[qx x1] Cos[py y1] Cos[pz z1]
  Vab  -= pq_int(dim, xa, w, 0,  py, pz, qx,0, 0);
       
  // -  Cos[px x1] Cos[qy y1] Cos[pz z1]
  Vab  -= pq_int(dim, xa, w, px, 0, pz, 0, qy, 0);
       
  // +  Cos[qx x1] Cos[qy y1] Cos[pz z1]
  Vab  += pq_int(dim, xa, w, 0, 0, pz, qx, qy, 0);   
       
  // - Cos[px x1] Cos[py y1] Cos[qz z1]  
  Vab  -= pq_int(dim, xa, w, px, py, 0, 0, 0, qz);
       
  // + Cos[qx x1] Cos[py y1] Cos[qz z1] 
  Vab  += pq_int(dim, xa, w, 0, py, 0, qx, 0, qz);
       
  // +  Cos[px x1] Cos[qy y1] Cos[qz z1]
  Vab  += pq_int(dim, xa, w, px, 0, 0, 0, qy, qz);
       
  // - Cos[qx x1] Cos[qy y1] Cos[qz z1]
  Vab  -= pq_int(dim, xa, w, 0, 0, 0, qx, qy, qz);

  return -Vab;//*8.0/(M_PI * M_PI * M_PI);

}

// the E integral is 1/pi^6 \int 1/(|r1 - r2|) dr1 dr2
// which is equivalent to 
// 1/pi^6 \int cos(0 x1) cos(0 y1) cos(0 z1) cos(0 x2) cos(0 y2) cos(0 z2)/|r1-r2| dr1 dr2
//
double JelliumIntegrals::E0_Int(int dim, double *xa, double *w) {
  return  pq_int(dim, xa, w, 0, 0, 0, 0, 0, 0);
}


//  Arguments:  dim = number of points for gauss-legendre grid
//              xa[]  = points on gauss-legendre grid
//              w[]   = weights from gauss-legendre grid
//              a[]   = array of nx, ny, and nz for orbital a
//              b[]   = array of nx, ny, and nz for orbital b
//              c[]   = array of nx, ny, and nz for orbital c
//              d[]   = array of nx, ny, and nz for orbital d
//  This function computes the ERI (a b | c d) where a, b, c, d are
//  all associated with three unique quantum numbers (nx, ny, nz)
//  According to Gill paper, each ERI can be written as a linear combination of (p|q) 
//  integrals where p is related to (a-b) or (a+b) and q is related to (c-d) or (c+d)
//  This function automatically enumerates all the appropriate (p|q), computes them, and
//  accumulates the total... Hopefully it works!
double JelliumIntegrals::ERI(int dim, double *xa, double *w, int *a, int *b, int *c, int *d) {

  int i, j, k, l, m, n;
 // double *x1, *x2, *y1, *y2, *z1, *z2;
  int faci, facj, fack, facl, facm, facn, fac;;
  //char *cp, *cq, *cr, *cs;
  double eri_val;
  //static const char *cx1[] = {"px x1", "qx x1"};
  //static const char *cx2[] = {"rx x2", "sx x2"};
  //static const char *cy1[] = {"py y1", "qy y1"};
  //static const char *cy2[] = {"ry y2", "sy y2"};
  //static const char *cz1[] = {"pz z1", "qz z1"};
  //static const char *cz2[] = {"rz z2", "sz z2"};  


  //x1 = (double *)malloc(3*sizeof(double));
  //x2 = (double *)malloc(3*sizeof(double));
  //y1 = (double *)malloc(3*sizeof(double));
  //y2 = (double *)malloc(3*sizeof(double));
  //z1 = (double *)malloc(3*sizeof(double));
  //z2 = (double *)malloc(3*sizeof(double));

  //x1[0] = ax-bx, x1[1] = ax+bx
  x1[0] = a[0] - b[0];
  x1[1] = a[0] + b[0];
  y1[0] = a[1] - b[1];
  y1[1] = a[1] + b[1];
  z1[0] = a[2] - b[2];
  z1[1] = a[2] + b[2];

  //x1[0] = cx-dx, x1[1] = cx+dx
  x2[0] = c[0] - d[0];
  x2[1] = c[0] + d[0];
  y2[0] = c[1] - d[1];
  y2[1] = c[1] + d[1];
  z2[0] = c[2] - d[2];
  z2[1] = c[2] + d[2];

  double tempval = 0.0;
  eri_val = 0.;
  // Generate all combinations of phi_a phi_b phi_c phi_d in expanded cosine form
  for (i=0; i<2; i++) {
    faci = (int)pow(-1,i);
    for (j=0; j<2; j++) {
      facj = (int)pow(-1,j);
      for (k=0; k<2; k++) {
        fack = (int)pow(-1,k);
        for (l=0; l<2; l++) { 
          facl = (int)pow(-1,l);
          for (m=0; m<2; m++) {
            facm = (int)pow(-1,m);
            for (n=0; n<2; n++) {
              facn = (int)pow(-1,n);
   
              fac = faci*facj*fack*facl*facm*facn;          
              // Uncomment to see the functions being integrated in each call to pq_int 
              //printf(" + %f Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] \n",
              //fac,cx1[n],cx2[m],cy1[l],cy2[k],cz1[j],cz2[i]);
              // recall pq_int args are -> dim, *xa, *w, px, py, pz, qx, qy, qz
              // order of indices to get these values is a bit strange, see print statement
              // for example of ordering!
              tempval = pq_int(dim, xa, w, x1[n], y1[l], z1[j], x2[m], y2[k], z2[i]);
             //printf("%d %d %d | %d %d %d -> ",x1[n],y1[l],z1[j],x2[m],y2[k],z2[i]);
             //printf("%f",tempval);

              // TABLE IV DEBUG LINE!!!!!!
              //printf("  (%d %d %d | %d %d %d) -> %17.14f\n",x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],tempval);
              eri_val += fac*tempval;

            }
          } 
        }
      }
    }
  }

 
  //free(x1);
  //free(x2);
  //free(y1);
  //free(y2);
  //free(z1);
  //free(z2);

  return eri_val;

}

//double JelliumIntegrals::ERI_new(std::shared_ptr<Vector> a, std::shared_ptr<Vector> b, std::shared_ptr<Vector> c, std::shared_ptr<Vector> d, double ** PQ, int *** PQmap) {
double JelliumIntegrals::ERI_new(int * a, int * b, int * c, int * d, double ** PQ, int *** PQmap) {

  //int * x1 = (int *)malloc(3*sizeof(int));
  //int * x2 = (int *)malloc(3*sizeof(int));
  //int * y1 = (int *)malloc(3*sizeof(int));
  //int * y2 = (int *)malloc(3*sizeof(int));
  //int * z1 = (int *)malloc(3*sizeof(int));
  //int * z2 = (int *)malloc(3*sizeof(int));

  //x1[0] = ax-bx, x1[1] = ax+bx
  x1[0] = a[0] - b[0];
  x1[1] = a[0] + b[0];
  y1[0] = a[1] - b[1];
  y1[1] = a[1] + b[1];
  z1[0] = a[2] - b[2];
  z1[1] = a[2] + b[2];

  //x1[0] = cx-dx, x1[1] = cx+dx
  x2[0] = c[0] - d[0];
  x2[1] = c[0] + d[0];
  y2[0] = c[1] - d[1];
  y2[1] = c[1] + d[1];
  z2[0] = c[2] - d[2];
  z2[1] = c[2] + d[2];

 //x1[0] = (int)(a->pointer()[0] - b->pointer()[0]);
 //x1[1] = (int)(a->pointer()[0] + b->pointer()[0]);
 //y1[0] = (int)(a->pointer()[1] - b->pointer()[1]);
 //y1[1] = (int)(a->pointer()[1] + b->pointer()[1]);
 //z1[0] = (int)(a->pointer()[2] - b->pointer()[2]);
 //z1[1] = (int)(a->pointer()[2] + b->pointer()[2]);

 ////x1[0] = cx-dx, x1[1] = cx+dx
 //x2[0] = (int)(c->pointer()[0] - d->pointer()[0]);
 //x2[1] = (int)(c->pointer()[0] + d->pointer()[0]);
 //y2[0] = (int)(c->pointer()[1] - d->pointer()[1]);
 //y2[1] = (int)(c->pointer()[1] + d->pointer()[1]);
 //z2[0] = (int)(c->pointer()[2] - d->pointer()[2]);
 //z2[1] = (int)(c->pointer()[2] + d->pointer()[2]);

  // Generate all combinations of phi_a phi_b phi_c phi_d in expanded cosine form

  double eri_val = 0.0;

  for (int i = 0; i < 2; i++) {
      if ( z2[i] < 0 ) continue;
      int faci = (int)pow(-1,i);
      for (int j = 0; j < 2; j++) {
          if ( z1[j] < 0 ) continue;
          int facij = faci * (int)pow(-1,j);
          for (int k = 0; k < 2; k++) {
              if ( y2[k] < 0 ) continue;
              int facijk = facij * (int)pow(-1,k);
              for (int l = 0; l < 2; l++) { 
                  if ( y1[l] < 0 ) continue;
                  int facijkl = facijk * (int)pow(-1,l);
                  for (int m = 0; m < 2; m++) {
                      if ( x2[m] < 0 ) continue;
                      int facijklm = facijkl * (int)pow(-1,m);
                      for (int n = 0; n < 2; n++) {

                          if ( x1[n] < 0 ) continue;
                          int facijklmn = facijklm * (int)pow(-1,n);
   
                          // Uncomment to see the functions being integrated in each call to pq_int 
                          //printf(" + %f Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] \n",
                          //fac,cx1[n],cx2[m],cy1[l],cy2[k],cz1[j],cz2[i]);
                          // recall pq_int args are -> dim, *xa, *w, px, py, pz, qx, qy, qz
                          // order of indices to get these values is a bit strange, see print statement
                          // for example of ordering!

                          //double dum = pq_int(dim, xa, w, x1[n], y1[l], z1[j], x2[m], y2[k], z2[i]);

                          int P = PQmap[ x1[n] ][ y1[l] ][ z1[j] ];
                          int Q = PQmap[ x2[m] ][ y2[k] ][ z2[i] ];
                          if ( P == -999 || Q == -999 ) {
                              outfile->Printf("\n");
                              outfile->Printf("    well, something is wrong with the indexing.\n");
                              outfile->Printf("    %5i %5i\n",P,Q);
                              outfile->Printf("    %5i %5i %5i; %5i %5i %5i\n",x1[n],y1[l],z1[j],x2[m],y2[k],z2[i]);
                              outfile->Printf("\n");
                              exit(1);
                          }
                          double dum = PQ[P][Q];

                          //double dum = pq_int_new(dim, x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],g_tensor,orbitalMax,sqrt_tensor);

                          // TABLE IV DEBUG LINE!!!!!!
                         // printf("  (%d %d %d | %d %d %d) -> %17.14f\n",x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],dum);
                          eri_val += facijklmn * dum;

                      }
                  } 
              }
          }
      }
  }

 
  //free(x1);
  //free(x2);
  //free(y1);
  //free(y2);
  //free(z1);
  //free(z2);

  return eri_val;

}

// This function implements Eq. 4.7 and 4.8 in Peter Gills paper on 2-electrons in a cube
// Gauss-Legendre quadrature is used for the 3d integral on the range 0->1 for x, y, and z
// int dim is the number of points on this grid, double *xa is a vector containing the actual points on this grid, and
// double *w is a vector containing the weights associated with this grid (analogous to differential length elements
// in rectangle rule integration).
// double px, py, pz, qx, qy, qz has the same interpretation as it does in the Gill paper.
double JelliumIntegrals::pq_int(int dim, double *xa, double *w, int px, int py, int pz, int qx, int qy, int qz) {
  double sum = 0.;
  double num, denom, x, y, z, dx, dy, dz, gx, gy, gz;
  double pi = M_PI;
  if (px<0 || qx<0 || py<0 || qy<0 || pz<0 || qz<0) {
         return 0.;
  } else {
    for (int i=0; i<dim; i++) {
        x = xa[i];
        dx = w[i];
        gx = g_pq(px, qx, x);
        for (int j=0; j<dim; j++) {
            y = xa[j];
            dy = w[j];
            gy = g_pq(py, qy, y);
            for (int k=0; k<dim; k++) {
                z = xa[k];
                dz = w[k];
                gz = g_pq(pz, qz, z);
                num = gx*gy*gz;
                denom = sqrt(x*x+y*y+z*z);
                sum += (num/denom)*dx*dy*dz;
                 
        //printf("  sum %f  i %d  j %d  k %d denom %f x %f dx %f y %f dy %f z %f dz %f gx %f gy %f gz %f\n",sum, i, j, k,denom,x,dx,y,dy,z,dz,gx,gy,gz);
            }
        }
    }
    return (8./pi)*sum;
  }
}

double JelliumIntegrals::pq_int_new(int dim, int px, int py, int pz, int qx, int qy, int qz, std::shared_ptr<Vector> g_tensor, int orbitalMax, std::shared_ptr<Vector> sqrt_tensor) {
    double pi = M_PI;
    if (px<0 || qx<0 || py<0 || qy<0 || pz<0 || qz<0){
        return 0.;
    }
    double sum = 0.;
    for (int i = 0; i < dim; i++){
        double gx = g_tensor->pointer()[i * orbitalMax * orbitalMax + px * orbitalMax + qx];
        for (int j = 0; j < dim; j++){
            double gxgy = gx * g_tensor->pointer()[j * orbitalMax * orbitalMax + py * orbitalMax + qy];
            for (int k = 0; k < dim; k++){
                double gxgygz = gxgy * g_tensor->pointer()[k * orbitalMax * orbitalMax + pz * orbitalMax + qz];
                sum += gxgygz * sqrt_tensor->pointer()[i*dim*dim + j*dim + k];
                //printf("  sum %f  x %f  y %f  z %f\n",sum, x, y, z);
            }
        }
    }
    return 8. * sum / pi;
}

/* 
/ Function to Determine Energy Calculation
/ Take function and loop through n to keep all atomic orbital energy levels.
*/
void JelliumIntegrals::OrderPsis3D(int norbs, double *E, int **MO) {

  int c, d, i, j, k, l, idx;
  double swap;
  int **N;
  int cond, Ecur;
  N = MAT_INT(2*(norbs+1)*(norbs+1)*(norbs+1),3);

  // Going to start with i and j=0 because that's the beginning
  // of the array... nx=i+1, ny=j+1

  for ( i=0; i<norbs; i++) {
    for ( j=0; j<norbs; j++) {
      for ( k=0; k<norbs; k++) {

        idx = i*norbs*norbs+j*norbs+k;
        // l is related to nx^2 + ny^2 + nz^2, aka, (i+1)^2 + (j+1)^2 (k+1)^2
        l = (i+1)*(i+1) + (j+1)*(j+1) + (k+1)*(k+1);
        E[idx] = l;
        // index is and energy is ...
        outfile->Printf("  Index is %i and Energy[%i,%i,%i] is %i\n",idx,i+1,j+1,k+1,l);
        // element N[k][0] is nx = i+1
        N[l][0] = i+1;
        // element N[k][1] is ny = j+1
        N[l][1] = j+1;
        // element N[k][2] is nz = k+1
        N[l][2] = k+1;

      }
    }
  }

  for ( c = 0 ; c < ( norbs*norbs*norbs-1 ); c++){
    for (d = 0 ; d < norbs*norbs*norbs - c - 1; d++){
      if (E[d] > E[d+1]) /* For decreasing order use < */
      {
        swap       = E[d];
        E[d]   = E[d+1];
        E[d+1] = swap;
      }
    }
  }

// print all energy values
for (int i=0; i<(norbs*norbs*norbs); i++) {
  outfile->Printf(" E[%i] is %f \n",i,E[i]);
}
  c=0;
  do {
    Ecur = E[c];
    i=0;
    do {
      i++;
      j=0;
      do {
        j++;
        k=0;
        do {
          k++;
          cond=Ecur-(i*i+j*j+k*k);
          if (cond==0) {
            MO[c][0] = i;
            MO[c][1] = j;
            MO[c][2] = k;
            c++;
          }
        }while( Ecur==E[c] && k<norbs);
      }while( Ecur==E[c] && j<norbs);
    }while (Ecur==E[c] && i<norbs);
  }while(c<norbs*norbs*norbs);
  for ( c = 0 ; c < ( norbs*norbs*norbs ); c++){
     E[c] = E[c];///(length*length);
  }
  outfile->Printf(" exit successful \n");

//  for (i=0; i<(norbs*norbs*norbs); i++) {
//    outfile->Printf("  Psi( %i , %i, %i ) %i\n",MO[i][0],MO[i][1],MO[i][2],E[i]);
//  }

}

int * JelliumIntegrals::VEC_INT(int dim){
  int *v,i;
  v = (int *)malloc(dim*sizeof(int));
  if (v==NULL) {
     outfile->Printf("\n\nVEC_INT: Memory allocation error\n\n");
     exit(0);
  }
  for (i=0; i<dim; i++) v[i] = 0;
  return v;
}

int ** JelliumIntegrals::MAT_INT(int dim1, int dim2){
  int i,j;
  int **M;
  M = (int **)malloc(dim1*sizeof(int *));
  if (M==NULL) {
     outfile->Printf("\n\nMAT_INT: Memory allocation error\n\n");
     exit(0);
  }
  for (i=0; i<dim1; i++){
      M[i] = (int *)malloc(dim2*sizeof(int));
      if (M[i]==NULL) {
         outfile->Printf("\n\nMAT_INT: Memory allocation error\n\n");
         exit(0);
      }
      for (j=0; j<dim2; j++){
          M[i][j] = 0;
      }
  }
  return M;
}

}} // end of namespaces
