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

#ifdef _OPENMP
    #include<omp.h>
#else
    #define omp_get_wtime() ( (double)clock() / CLOCKS_PER_SEC )
    #define omp_get_max_threads() 1
#endif

using namespace psi;

namespace psi{ namespace jellium_scf {

JelliumIntegrals::JelliumIntegrals(Options & options):
        options_(options)
{

    outfile->Printf("\n");
    outfile->Printf("    ==> Jellium Integral Construction <==\n");
    outfile->Printf("\n");

    orbitalMax = options.get_int("N_BASIS_FUNCTIONS");
    length = options.get_double("LENGTH");
    electrons = options.get_int("N_ELECTRONS");
    common_init();
    compute();
}

// free memory here
JelliumIntegrals::~JelliumIntegrals()
{
}

void JelliumIntegrals::common_init() {

   // x1 = (int *)malloc(3*sizeof(int));
   // x2 = (int *)malloc(3*sizeof(int));
   // y1 = (int *)malloc(3*sizeof(int));
   // y2 = (int *)malloc(3*sizeof(int));
   // z1 = (int *)malloc(3*sizeof(int));
   // z2 = (int *)malloc(3*sizeof(int));
    
}

void JelliumIntegrals::compute() {

 //printf ( "\n" );
  //printf ( "LEGENDRE_RULE_FAST:\n" );
  //printf ( "  Normal end of execution.\n" );

  //printf ( "\n" );
  double a = 0.0;
  double b = 1.0;
  int n = options_.get_int("N_GRID_POINTS");
  double *x;
  int *mu, *nu, *sig, *lam;

  x   = (double *)malloc(n*sizeof(double));
  w   = (double *)malloc(n*sizeof(double));
  grid_points = x; 
 
  sig  = (int*)malloc(3*sizeof(int));
  lam  = (int*)malloc(3*sizeof(int));

  nmax=30;
  //TODO: make this vector irrep from the get go
  std::shared_ptr<Vector> ORBE = std::shared_ptr<Vector>( new Vector(3*nmax*nmax*nmax));//VEC_INT(3*nmax*nmax*nmax);
  MO  = MAT_INT(3*nmax*nmax*nmax,3);
  OrderPsis3D(nmax, ORBE->pointer(), MO);
  Orderirrep(nmax, ORBE->pointer(), MO, electrons);
  Legendre tmp;
  //  Constructe grid and weights, store them to the vectors x and w, respectively.
  //  This is one of John Burkhardt's library functions
  tmp.legendre_compute_glr(n, x, w);

  // Scale the grid to start at value a and end at value b. 
  // We want our integration range to go from 0 to 1, so a = 0, b = 1
  // This is also one of John Burkhardt's library functions
  tmp.rescale( a, b, n, x, w);

  for(int i = 0; i < n; i++){
    //  printf("weight %d\t%f\n",i,w[i]);
  }
  // build g tensor g[npq] * w[n]
  outfile->Printf("\n");
  outfile->Printf("    build g tensor................"); fflush(stdout);
  g_tensor = std::shared_ptr<Vector>( new Vector(n * 2 * (nmax + 1) * 2 * (nmax + 1)));
  for (int pt = 0; pt < n; pt++) {
      double xval = x[pt];
      for (int p = 0; p <= nmax*2; p++) {
          for (int q = 0; q <= nmax*2; q++) {
              g_tensor->pointer()[pt*2*nmax*2*nmax+p*2*nmax+q] = g_pq(p, q, xval) * w[pt];
          }
      }
  }
  for( int i = 0; i < n;i++){
     //outfile->Printf("%d\t%f\n",i,g_tensor->pointer()[i*nmax*nmax*4+2*nmax*2]);
  }
  outfile->Printf("done.\n");
  // build sqrt(x*x+y*y+z*z)
  outfile->Printf("    build sqrt tensor............."); fflush(stdout);
  sqrt_tensor = std::shared_ptr<Vector>(new Vector(n*n*n));
  double * s_p = sqrt_tensor->pointer();
  for (int i = 0; i < n; i++) {
      double xval = x[i];
      for (int j = 0; j < n; j++) {
          double yval = x[j];
          for (int k = 0; k < n; k++) {
              double zval = x[k];
              double val = sqrt(xval*xval+yval*yval+zval*zval);
              s_p[i*n*n + j*n + k] = 1.0/val;
          }
      }
  }
  outfile->Printf("done.\n");


  unsigned long start_pq = clock();
  // now, compute (P|Q)
  outfile->Printf("    build (P|Q)..................."); fflush(stdout);
  PQmap = (int ***)malloc((2*nmax+1)*sizeof(int**));
  for (int i = 0; i < 2*nmax+1; i++) {
      PQmap[i] = (int **)malloc((2*nmax+1)*sizeof(int*));
      for (int j = 0; j < 2*nmax+1; j++) {
          PQmap[i][j] = (int *)malloc((2*nmax+1)*sizeof(int));
          for (int k = 0; k < 2*nmax+1; k++) {
              PQmap[i][j][k] = 999;
          }
      }
  }
  int Pdim = 0;
  for (int px = 0; px < 2*nmax+1; px++) {
      for (int py = 0; py < 2*nmax+1; py++) {
          for (int pz = 0; pz < 2*nmax+1; pz++) {
              PQmap[px][py][pz] = Pdim;
              Pdim++;
          }
      }
  }
  //printf("1, -2, -3, 4, 2, 0\n"); 
  //pq_int(orbitalMax, x, w, 1, -2, -3, 4, 2, 0);
  //exit(1);
  PQ = std::shared_ptr<Matrix>(new Matrix(Pdim,Pdim));
  double ** PQ_p = PQ->pointer();
  //printf("pdim %d\n",Pdim);
  Ke = std::shared_ptr<Matrix>(new Matrix(nirrep_,nsopi_,nsopi_));
  NucAttrac = std::shared_ptr<Matrix>(new Matrix(nirrep_,nsopi_,nsopi_));
  int complete = 0;
  long counter = 0;
  //printf("hello world %d\n", omp_get_max_threads());
  long iterations = 1.3333*pow(nmax,6)+10*pow(nmax,5)+33*pow(nmax,4)+60.833*pow(nmax,3)+65.667*pow(nmax,2)+39.168*nmax+9.9869;
  iterations/=2000;
  #pragma omp parallel
  {
  #pragma omp for schedule(dynamic) nowait 
  for (int px = 0; px < 2*nmax+1; px++) {
      for (int qx = px; qx < 2*nmax+1; qx++) {

          int pq_x = px*(2*nmax+1) + qx;

          for (int py = 0; py < 2*nmax+1; py++) {
              for (int qy = py; qy < 2*nmax+1; qy++) {

                  int pq_y = py*(2*nmax+1) + qy;
                  if ( pq_x > pq_y ) continue;

                  for (int pz = 0; pz < 2*nmax+1; pz++) {
                      for (int qz = pz; qz < 2*nmax+1; qz++) {

                          int pq_z = pz*(2*nmax+1) + qz;
                          if ( pq_y > pq_z ) continue;

                          if(int(counter/(iterations))>complete){
                             //printf("\r%i%% complete",complete);
                             fflush(stdout);
                             complete++;
                          }
                          //if ( P > Q ) continue;
                          if((px+qx)%2==0 && (py+qy)%2==0 && (pz+qz)%2==0){
                          double dum = pq_int_new(n, px, py, pz, qx, qy, qz);
//                         printf("dum %f",dum); 
                          int P,Q;

                          // start 
                          P = PQmap[px][py][pz];
                          Q = PQmap[qx][qy][qz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qx][py][pz];
                          Q = PQmap[px][qy][qz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[px][qy][pz];
                          Q = PQmap[qx][py][qz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qx][qy][pz];
                          Q = PQmap[px][py][qz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[px][py][qz];
                          Q = PQmap[qx][qy][pz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qx][py][qz];
                          Q = PQmap[px][qy][pz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[px][qy][qz];
                          Q = PQmap[qx][py][pz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qx][qy][qz];
                          Q = PQmap[px][py][pz];
                          PQ_p[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[py][px][pz];
                          Q = PQmap[qy][qx][qz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[py][qx][pz];
                          Q = PQmap[qy][px][qz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qy][px][pz];
                          Q = PQmap[py][qx][qz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qy][qx][pz];
                          Q = PQmap[py][px][qz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[py][px][qz];
                          Q = PQmap[qy][qx][pz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[py][qx][qz];
                          Q = PQmap[qy][px][pz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qy][px][qz];
                          Q = PQmap[py][qx][pz];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qy][qx][qz];
                          Q = PQmap[py][px][pz];
                          PQ_p[P][Q] = dum;

                          // now begins pxqx < pyqy < pzqz

                          P = PQmap[pz][px][py];
                          Q = PQmap[qz][qx][qy];
                          PQ_p[P][Q] = dum;

                          P = PQmap[pz][qx][py];
                          Q = PQmap[qz][px][qy];
                          PQ_p[P][Q] = dum;

                          P = PQmap[pz][px][qy];
                          Q = PQmap[qz][qx][py];
                          PQ_p[P][Q] = dum;

                          P = PQmap[pz][qx][qy];
                          Q = PQmap[qz][px][py];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qz][px][py];
                          Q = PQmap[pz][qx][qy];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qz][qx][py];
                          Q = PQmap[pz][px][qy];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qz][px][qy];
                          Q = PQmap[pz][qx][py];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qz][qx][qy];
                          Q = PQmap[pz][px][py];
                          PQ_p[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[pz][py][px];
                          Q = PQmap[qz][qy][qx];
                          PQ_p[P][Q] = dum;

                          P = PQmap[pz][py][qx];
                          Q = PQmap[qz][qy][px];
                          PQ_p[P][Q] = dum;

                          P = PQmap[pz][qy][px];
                          Q = PQmap[qz][py][qx];
                          PQ_p[P][Q] = dum;

                          P = PQmap[pz][qy][qx];
                          Q = PQmap[qz][py][px];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qz][py][px];
                          Q = PQmap[pz][qy][qx];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qz][py][qx];
                          Q = PQmap[pz][qy][px];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qz][qy][px];
                          Q = PQmap[pz][py][qx];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qz][qy][qx];
                          Q = PQmap[pz][py][px];
                          PQ_p[P][Q] = dum;

                          // now begins last set of 16

                          P = PQmap[px][pz][py];
                          Q = PQmap[qx][qz][qy];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qx][pz][py];
                          Q = PQmap[px][qz][qy];
                          PQ_p[P][Q] = dum;

                          P = PQmap[px][pz][qy];
                          Q = PQmap[qx][qz][py];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qx][pz][qy];
                          Q = PQmap[px][qz][py];
                          PQ_p[P][Q] = dum;

                          P = PQmap[px][qz][py];
                          Q = PQmap[qx][pz][qy];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qx][qz][py];
                          Q = PQmap[px][pz][qy];
                          PQ_p[P][Q] = dum;

                          P = PQmap[px][qz][qy];
                          Q = PQmap[qx][pz][py];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qx][qz][qy];
                          Q = PQmap[px][pz][py];
                          PQ_p[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[py][pz][px];
                          Q = PQmap[qy][qz][qx];
                          PQ_p[P][Q] = dum;

                          P = PQmap[py][pz][qx];
                          Q = PQmap[qy][qz][px];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qy][pz][px];
                          Q = PQmap[py][qz][qx];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qy][pz][qx];
                          Q = PQmap[py][qz][px];
                          PQ_p[P][Q] = dum;

                          P = PQmap[py][qz][px];
                          Q = PQmap[qy][pz][qx];
                          PQ_p[P][Q] = dum;

                          P = PQmap[py][qz][qx];
                          Q = PQmap[qy][pz][px];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qy][qz][px];
                          Q = PQmap[py][pz][qx];
                          PQ_p[P][Q] = dum;

                          P = PQmap[qy][qz][qx];
                          Q = PQmap[py][pz][px];
                          PQ_p[P][Q] = dum;

                          counter++;
                          }
                      }
                  }
              }
          }
      }
  }
  }
//printf("%ld\n",counter);
                          //int P = PQmap[ 0 ][ 0 ][ 0 ];
                          //int Q = PQmap[ 0 ][ 0 ][ 0 ];
                          //double dum = PQ->pointer()[P][Q];
  unsigned long end_pq = clock();
  outfile->Printf("done.\n");fflush(stdout);

  outfile->Printf("\n");
  outfile->Printf("    time for (P|Q) construction:                %6.1f s\n",(double)(end_pq-start_pq)/CLOCKS_PER_SEC); fflush(stdout);
  outfile->Printf("\n");
  //outfile->Printf("canonical integrals");

  // Four nested loops to compute lower triange of electron repulsion integrals - roughly have of the non-unique integrals
  // will not be computed, but this is still not exploiting symmetry fully
  outfile->Printf("    build potential integrals.....");fflush(stdout);
  unsigned long start = clock();
  #pragma omp parallel
  {
  #pragma omp for schedule(dynamic) nowait
  for(int h = 0; h < nirrep_;h++){
     int offset = 0;
     double** Ke_p = Ke->pointer(h);
     for(int i = 0; i < h; i ++){
        offset += nsopi_[i];
     }
     double** Nu_p = NucAttrac->pointer(h);
  for (int i=0; i< nsopi_[h]; i++) {
  int* mu;
  int* nu;
      mu = MO[i+offset];

      for (int j=i; j< nsopi_[h]; j++) {
          nu = MO[j+offset];

          // Lower triangle of 1-electron integrals will be computed, fully exploiting symmetry (I think!)
          // Kinetic Energy Integrals - already computed and stored in ORBE vector    
          if (i==j) { 
              Ke_p[i][j] = 0.5*ORBE->pointer()[i+offset];
          }
          //printf("%f",kinval);
          // Nuclear-attraction Integrals
          double dum = Vab_Int_new(n, x, w, mu, nu);
          Nu_p[i][j] = dum;
          Nu_p[j][i] = dum;

      }
  }
     //offset += nsopi_[h];
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
  unsigned long end = clock();
  outfile->Printf("done.\n");fflush(stdout);
  outfile->Printf("\n");
  outfile->Printf("    time for potential integral construction:   %6.1f s\n",(double)(end-start)/CLOCKS_PER_SEC); fflush(stdout);
  outfile->Printf("\n");

  // Compute self energy
  selfval = E0_Int(n, x, w);
  // Print to file
  //fprintf(selffp, "  %17.14f\n",selfval); 
  //free(x);
  //free(w);
  //free(mu);
  //free(nu);

}

double JelliumIntegrals::ERI_int(int a, int b, int c, int d){
    return ERI_unrolled(MO[a], MO[b], MO[c], MO[d], PQ->pointer(), PQmap);
}

double JelliumIntegrals::g_pq(int p, int q, double x) {
    int d = abs(p-q);
    double pi = M_PI;
    ////if(q < 0 || p < 0){
    ////   return 0;
    ////}
    //if (p == q && p == 0) {
    //  return 1.0 - x;
    //}
    //else if ( p == q && p > 0 ) {
    //  return (1.0 - x)*cos(p*pi*x)/2.0 - sin(p*pi*x)/(2*p*pi);
    //}
    //else if ( (d % 2)==0 && d!=0) {
    //  return (q*sin(q*pi*x) - p*sin(p*pi*x))/((p*p-q*q)*pi);
    //}
    //else 
    //  return 0.0;

    if ( p == q ) {
        if ( p == 0 ) {
            return 1.0 - x;
        }else {
            return (1.0 - x)*cos(p*pi*x)/2.0 - sin(p*pi*x)/(2*p*pi);
        }
    }else if ( (d % 2)==0 && d != 0 ) {
        return (q*sin(q*pi*x) - p*sin(p*pi*x))/((p*p-q*q)*pi);
    }   
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

double JelliumIntegrals::Vab_Int_new(int dim, double *xa, double *w, int *a, int *b) {

    int px, py, pz, qx, qy, qz;
    double Vab;
    px = abs(a[0] - b[0]);
    py = abs(a[1] - b[1]);
    pz = abs(a[2] - b[2]);
    

    qx = a[0] + b[0];
    qy = a[1] + b[1];
    qz = a[2] + b[2];
    if(px%2==1 || py %2==1 || pz%2==1 || qx%2==1 || qy%2==1 || qz%2==1){
       return 0.0;
    }
    Vab = 0.0;
 
    //make sure to check that these integrals are correct

    //    Cos[px x1] Cos[py y1] Cos[pz z1]
    Vab  += pq_int_new(dim, px, py, pz,  0,  0,  0);
         
    // -  Cos[qx x1] Cos[py y1] Cos[pz z1]
    Vab  -= pq_int_new(dim,  0, py, pz, qx,  0,  0);
         
    // -  Cos[px x1] Cos[qy y1] Cos[pz z1]
    Vab  -= pq_int_new(dim, px,  0, pz,  0, qy,  0);
         
    // +  Cos[qx x1] Cos[qy y1] Cos[pz z1]
    Vab  += pq_int_new(dim,  0,  0, pz, qx, qy,  0);
         
    // - Cos[px x1] Cos[py y1] Cos[qz z1]  
    Vab  -= pq_int_new(dim, px, py,  0,  0,  0, qz);
         
    // + Cos[qx x1] Cos[py y1] Cos[qz z1] 
    Vab  += pq_int_new(dim,  0, py,  0, qx,  0, qz);
         
    // +  Cos[px x1] Cos[qy y1] Cos[qz z1]
    Vab  += pq_int_new(dim, px,  0,  0,  0, qy, qz);
         
    // - Cos[qx x1] Cos[qy y1] Cos[qz z1]
    Vab  -= pq_int_new(dim,  0,  0,  0, qx, qy, qz);

            //if(px != py && py != pz && px != pz && px%2==0 && pz%2==0 && py%2==0){ printf("three different even\t : Vab %f",Vab);
            // if((mu[0] == mu[1] && mu[1] != mu[2] && mu[0]%2==0 && mu[2]%2==0) || (mu[0] != mu[1] && mu[1] == mu[2] && mu[0]%2==0 && mu[2]%2==0) || (mu[0] == mu[2] && mu[1] != mu[2] && mu[1]%2==0 && mu[2]%2==0)) printf("A1g + Eg\t");
            // if(mu[0] != mu[1] && mu[1] != mu[2] && mu[0]%2==0 && mu[1]%2==0 && mu[2]%2==0) printf("A1g + A2g + 2Eg\t");
            // if(mu[0] == mu[1] && mu[1] == mu[2] && mu[0]%2==0) printf("T2g\t");
            // if(mu[0] == mu[1] && mu[1] == mu[2] && mu[0]%2==0) printf("A1g\t");
            // if(mu[0] == mu[1] && mu[1] == mu[2] && mu[0]%2==0) printf("A1g\t");
            // if(mu[0] == mu[1] && mu[1] == mu[2] && mu[0]%2==0) printf("A1g\t");
            // if(mu[0] == mu[1] && mu[1] == mu[2] && mu[0]%2==0) printf("A1g\t");
    return -Vab;

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
//double JelliumIntegrals::ERI(int dim, double *xa, double *w, int *a, int *b, int *c, int *d) {
//
//  int i, j, k, l, m, n;
// // double *x1, *x2, *y1, *y2, *z1, *z2;
//  int faci, facj, fack, facl, facm, facn, fac;;
//  //char *cp, *cq, *cr, *cs;
//  double eri_val;
//  //static const char *cx1[] = {"px x1", "qx x1"};
//  //static const char *cx2[] = {"rx x2", "sx x2"};
//  //static const char *cy1[] = {"py y1", "qy y1"};
//  //static const char *cy2[] = {"ry y2", "sy y2"};
//  //static const char *cz1[] = {"pz z1", "qz z1"};
//  //static const char *cz2[] = {"rz z2", "sz z2"};  
//
//
//  //x1 = (double *)malloc(3*sizeof(double));
//  //x2 = (double *)malloc(3*sizeof(double));
//  //y1 = (double *)malloc(3*sizeof(double));
//  //y2 = (double *)malloc(3*sizeof(double));
//  //z1 = (double *)malloc(3*sizeof(double));
//  //z2 = (double *)malloc(3*sizeof(double));
//
//  //x1[0] = ax-bx, x1[1] = ax+bx
//  x1[0] = a[0] - b[0];
//  x1[1] = a[0] + b[0];
//  y1[0] = a[1] - b[1];
//  y1[1] = a[1] + b[1];
//  z1[0] = a[2] - b[2];
//  z1[1] = a[2] + b[2];
//
//  //x1[0] = cx-dx, x1[1] = cx+dx
//  x2[0] = c[0] - d[0];
//  x2[1] = c[0] + d[0];
//  y2[0] = c[1] - d[1];
//  y2[1] = c[1] + d[1];
//  z2[0] = c[2] - d[2];
//  z2[1] = c[2] + d[2];
//
//  double tempval = 0.0;
//  eri_val = 0.;
//  // Generate all combinations of phi_a phi_b phi_c phi_d in expanded cosine form
//  for (i=0; i<2; i++) {
//    faci = (int)pow(-1,i);
//    for (j=0; j<2; j++) {
//      facj = (int)pow(-1,j);
//      for (k=0; k<2; k++) {
//        fack = (int)pow(-1,k);
//        for (l=0; l<2; l++) { 
//          facl = (int)pow(-1,l);
//          for (m=0; m<2; m++) {
//            facm = (int)pow(-1,m);
//            for (n=0; n<2; n++) {
//              facn = (int)pow(-1,n);
//   
//              fac = faci*facj*fack*facl*facm*facn;          
//              // Uncomment to see the functions being integrated in each call to pq_int 
//              //printf(" + %f Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] \n",
//              //fac,cx1[n],cx2[m],cy1[l],cy2[k],cz1[j],cz2[i]);
//              // recall pq_int args are -> dim, *xa, *w, px, py, pz, qx, qy, qz
//              // order of indices to get these values is a bit strange, see print statement
//              // for example of ordering!
//              tempval = pq_int(dim, xa, w, x1[n], y1[l], z1[j], x2[m], y2[k], z2[i]);
//             //printf("%d %d %d | %d %d %d -> ",x1[n],y1[l],z1[j],x2[m],y2[k],z2[i]);
//             //printf("%f",tempval);
//
//              // TABLE IV DEBUG LINE!!!!!!
//              //printf("  (%d %d %d | %d %d %d) -> %17.14f\n",x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],tempval);
//              eri_val += fac*tempval;
//
//            }
//          } 
//        }
//      }
//    }
//  }
//
// 
//  //free(x1);
//  //free(x2);
//  //free(y1);
//  //free(y2);
//  //free(z1);
//  //free(z2);
//
//  return eri_val;
//
//}

double JelliumIntegrals::ERI_unrolled(int * a, int * b, int * c, int * d, double ** PQ, int *** PQmap) {
  
  //x1[0] = ax-bx, x1[1] = ax+bx
  int* x1 = (int *)malloc(3*sizeof(int));
  int* x2 = (int *)malloc(3*sizeof(int));
  int* y1 = (int *)malloc(3*sizeof(int));
  int* y2 = (int *)malloc(3*sizeof(int));
  int* z1 = (int *)malloc(3*sizeof(int));
  int* z2 = (int *)malloc(3*sizeof(int));
 
  x1[0] = abs(a[0] - b[0]);
  y1[0] = abs(a[1] - b[1]);
  z1[0] = abs(a[2] - b[2]);

  x1[1] = a[0] + b[0];
  y1[1] = a[1] + b[1];
  z1[1] = a[2] + b[2];

  //x1[0] abs(= cx-dx, x1)[1] = cx+dx
  x2[0] = abs(c[0] - d[0]);
  y2[0] = abs(c[1] - d[1]);
  z2[0] = abs(c[2] - d[2]);

  x2[1] = c[0] + d[0];
  y2[1] = c[1] + d[1];
  z2[1] = c[2] + d[2];
 
  //bool guess = false;
  ///if((z1[0]+z1[1])!=(z2[0]+z2[1]) && (x1[0]+x1[1])!=(x2[0]+x2[1]) && (y1[0]+y1[1])!=(y2[0]+y2[1])){
  //if(x1[0]*y1[0]*z1[0] == 0){
  //  if(x1[0]%2==1 || y1[0]%2==1 || y1[0]%2==1){
  //  return 0.0;
  //  guess = true; }
  //}
  //if(x1[0]%2!=0 || (y1[0])%2!=0){
  //  printf("0\n");
  //  return 0;
  //}
  // Generate all combinations of phi_a phi_b phi_c phi_d in expanded cosine form

  double eri_val = 0.0;

  int Q = PQmap[ x2[0] ][ y2[0] ][ z2[0] ];

  int P = PQmap[ x1[0] ][ y1[0] ][ z1[0] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[0] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[0] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[0] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[0] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[0] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[0] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[1] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[0] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[1] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[0] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[1] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[0] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[1] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[0] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[0] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[1] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[0] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[1] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[0] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[1] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[0] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[1] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[1] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[1] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[1] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[1] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[1] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[1] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[1] ][ z2[0] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[1] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[0] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[0] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[0] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[0] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[0] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[0] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[0] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[0] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[1] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[0] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[1] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[0] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[1] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[0] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[1] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[0] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[0] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[0] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[1] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[0] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[1] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[0] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[1] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[0] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[1] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[1] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[1] ];
  eri_val += PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[1] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[0] ][ z1[1] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[0] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[0] ][ y2[1] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[1] ];
  eri_val += PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  Q = PQmap[ x2[1] ][ y2[1] ][ z2[1] ];

  P = PQmap[ x1[0] ][ y1[1] ][ z1[1] ];
  eri_val -= PQ[P][Q];

  P = PQmap[ x1[1] ][ y1[1] ][ z1[1] ];
  eri_val += PQ[P][Q];

  //if(eri_val == 0)
  //  printf("a[0]: %d a[1]: %d a[2]: %d b[0]: %d b[1]: %d b[2]: %d c[0] %d c[1] %d c[2] %d d[0] %d d[1] %d d[2] %d value %f\n",a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2],d[0],d[1],d[2],eri_val);
     //printf("a: %d b: %d c: %d d: %d eri_val %f\n",(a[0]+a[1]+a[2]),(b[0]+b[1]+b[2]),(c[0]+c[1]+c[2]),(d[0]+d[1]+d[2]),eri_val);
  free(x1);
  free(x2);
  free(y1);
  free(y2);
  free(z1);
  free(z2);
  return eri_val;

}

//double JelliumIntegrals::ERI_new(std::shared_ptr<Vector> a, std::shared_ptr<Vector> b, std::shared_ptr<Vector> c, std::shared_ptr<Vector> d, double ** PQ, int *** PQmap) {
//double JelliumIntegrals::ERI_new(int * a, int * b, int * c, int * d, double ** PQ, int *** PQmap) {
//
//  //int * x1 = (int *)malloc(3*sizeof(int));
//  //int * x2 = (int *)malloc(3*sizeof(int));
//  //int * y1 = (int *)malloc(3*sizeof(int));
//  //int * y2 = (int *)malloc(3*sizeof(int));
//  //int * z1 = (int *)malloc(3*sizeof(int));
//  //int * z2 = (int *)malloc(3*sizeof(int));
//
//  //x1[0] = ax-bx, x1[1] = ax+bx
//  x1[0] = a[0] - b[0];
//  x1[1] = a[0] + b[0];
//  y1[0] = a[1] - b[1];
//  y1[1] = a[1] + b[1];
//  z1[0] = a[2] - b[2];
//  z1[1] = a[2] + b[2];
//
//  //x1[0] = cx-dx, x1[1] = cx+dx
//  x2[0] = c[0] - d[0];
//  x2[1] = c[0] + d[0];
//  y2[0] = c[1] - d[1];
//  y2[1] = c[1] + d[1];
//  z2[0] = c[2] - d[2];
//  z2[1] = c[2] + d[2];
//
// //x1[0] = (int)(a->pointer()[0] - b->pointer()[0]);
// //x1[1] = (int)(a->pointer()[0] + b->pointer()[0]);
// //y1[0] = (int)(a->pointer()[1] - b->pointer()[1]);
// //y1[1] = (int)(a->pointer()[1] + b->pointer()[1]);
// //z1[0] = (int)(a->pointer()[2] - b->pointer()[2]);
// //z1[1] = (int)(a->pointer()[2] + b->pointer()[2]);
//
// ////x1[0] = cx-dx, x1[1] = cx+dx
// //x2[0] = (int)(c->pointer()[0] - d->pointer()[0]);
// //x2[1] = (int)(c->pointer()[0] + d->pointer()[0]);
// //y2[0] = (int)(c->pointer()[1] - d->pointer()[1]);
// //y2[1] = (int)(c->pointer()[1] + d->pointer()[1]);
// //z2[0] = (int)(c->pointer()[2] - d->pointer()[2]);
// //z2[1] = (int)(c->pointer()[2] + d->pointer()[2]);
//
//  // Generate all combinations of phi_a phi_b phi_c phi_d in expanded cosine form
//
//  double eri_val = 0.0;
//
//  for (int i = 0; i < 2; i++) {
//      if ( z2[i] < 0 ) continue;
//      int faci = (int)pow(-1,i);
//      for (int j = 0; j < 2; j++) {
//          if ( z1[j] < 0 ) continue;
//          int facij = faci * (int)pow(-1,j);
//          for (int k = 0; k < 2; k++) {
//              if ( y2[k] < 0 ) continue;
//              int facijk = facij * (int)pow(-1,k);
//              for (int l = 0; l < 2; l++) { 
//                  if ( y1[l] < 0 ) continue;
//                  int facijkl = facijk * (int)pow(-1,l);
//                  for (int m = 0; m < 2; m++) {
//                      if ( x2[m] < 0 ) continue;
//                      int facijklm = facijkl * (int)pow(-1,m);
//                      for (int n = 0; n < 2; n++) {
//
//                          if ( x1[n] < 0 ) continue;
//                          int facijklmn = facijklm * (int)pow(-1,n);
//   
//                          // Uncomment to see the functions being integrated in each call to pq_int 
//                          //printf(" + %f Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] \n",
//                          //fac,cx1[n],cx2[m],cy1[l],cy2[k],cz1[j],cz2[i]);
//                          // recall pq_int args are -> dim, *xa, *w, px, py, pz, qx, qy, qz
//                          // order of indices to get these values is a bit strange, see print statement
//                          // for example of ordering!
//
//                          //double dum = pq_int(dim, xa, w, x1[n], y1[l], z1[j], x2[m], y2[k], z2[i]);
//
//                          int P = PQmap[ x1[n] ][ y1[l] ][ z1[j] ];
//                          int Q = PQmap[ x2[m] ][ y2[k] ][ z2[i] ];
//                          if ( P == -999 || Q == -999 ) {
//                              outfile->Printf("\n");
//                              outfile->Printf("    well, something is wrong with the indexing.\n");
//                              outfile->Printf("    %5i %5i\n",P,Q);
//                              outfile->Printf("    %5i %5i %5i; %5i %5i %5i\n",x1[n],y1[l],z1[j],x2[m],y2[k],z2[i]);
//                              outfile->Printf("\n");
//                              exit(1);
//                          }
//                          double dum = PQ[P][Q];
//
//                          //double dum = pq_int_new(dim, x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],g_tensor,orbitalMax,sqrt_tensor);
//
//                          // TABLE IV DEBUG LINE!!!!!!
//                         // printf("  (%d %d %d | %d %d %d) -> %17.14f\n",x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],dum);
//                          eri_val += facijklmn * dum;
//
//                      }
//                  } 
//              }
//          }
//      }
//  }
//
// 
//  //free(x1);
//  //free(x2);
//  //free(y1);
//  //free(y2);
//  //free(z1);
//  //free(z2);
//
//  return eri_val;
//
//}

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
double JelliumIntegrals::pq_int_new(int dim, int px, int py, int pz, int qx, int qy, int qz) {
    //if (px<0 || qx<0 || py<0 || qy<0 || pz<0 || qz<0){
    //    return 0.;
    //}
    double * s_p = sqrt_tensor->pointer();
    double * g_p = g_tensor->pointer();
    double sum = 0.0;
    double tmp_gxgygz = 0.0;
    //double tmp_gxgy = 0.0;
    for (int i = 0; i < dim; i++){
        double gx = g_p[i * 2 * nmax * 2 * nmax + px * 2 * nmax + qx];
        for (int j = 0; j < dim; j++){
            double gxgy = gx*g_p[j * 2 * nmax * 2 * nmax + py * 2 * nmax + qy];
            for (int k = 0; k < dim; k++){
                double gxgygz = g_p[k * 2 * nmax * 2 * nmax + pz * 2 * nmax + qz];
                tmp_gxgygz += gxgygz * s_p[i*dim*dim + j*dim + k];
                //printf("  sum %f  x %f  y %f  z %f\n",sum, x, y, z);
            }
            sum = tmp_gxgygz * gxgy + sum;
            tmp_gxgygz = 0.0;
        }
    }
    return 8.0 * sum / M_PI;
}

/* 
/ Function to Determine Energy Calculation
/ Take function and loop through n to keep all atomic orbital energy levels.
*/
void JelliumIntegrals::OrderPsis3D(int &norbs, double *E, int **MO) {

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
          //outfile->Printf("  Index is %i and Energy[%i,%i,%i] is %i\n",idx,i+1,j+1,k+1,l);
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
    //for (int i=0; i<(norbs*norbs*norbs); i++) {
    //  outfile->Printf(" E[%i] is %f \n",i,E[i]);
    //}
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
  
    // reset nmax to be actual maximum necessary to consider, given orbitalMax
  
    //outfile->Printf(" exit successful \n");
  
      for (i=0; i<(norbs*norbs*norbs); i++) {
       // outfile->Printf("  Psi( %i , %i, %i ) %i\n",MO[i][0],MO[i][1],MO[i][2],E[i]);
      }

    int new_nmax = 0;
    for (int i = 0; i < orbitalMax; i++) {
        if ( MO[i][0] > new_nmax ) new_nmax = MO[i][0];
        if ( MO[i][1] > new_nmax ) new_nmax = MO[i][1];
        if ( MO[i][2] > new_nmax ) new_nmax = MO[i][2];
    }
    //printf("%5i\n",nmax);
    //printf("%5i\n",new_nmax);
    //exit(0);
    norbs = new_nmax;
}
void JelliumIntegrals::Orderirrep(int &norbs, double *E, int **MO, int electrons) {
    int eee = 0;
    int eeo = 0;
    int eoe = 0;
    int eoo = 0;
    int oee = 0;
    int oeo = 0;
    int ooe = 0;
    int ooo = 0;
    for (int i = 0; i < orbitalMax; i++) {
        if ( MO[i][0]%2==0 ){
            if ( MO[i][1]%2==0 ){
                if( MO[i][2]%2==0 ){
                   eee++;
                } else {
                   eeo++;
                }
            }else{
                if( MO[i][2]%2==0 ){
                   eoe++;
                } else {
                   eoo++;
                }
            }
        } else {
            if ( MO[i][1]%2==0 ){
                if( MO[i][2]%2==0 ){
                   oee++;
                } else {
                   oeo++;
                }
            }else{
                if( MO[i][2]%2==0 ){
                   ooe++;
                } else {
                   ooo++;
                }
            } 
        }
    } 
    //printf("eee: %d \neeo: %d \neoe: %d \neoo: %d\n oee: %d\n oeo: %d\n ooe: %d\n ooo: %d\n",eee,eeo,eoe,eoo,oee,oeo,ooe,ooo);
    nirrep_ = 8;
    nsopi_ = (int*)malloc(nirrep_*sizeof(int));
    nsopi_[0] = eee;
    nsopi_[1] = eeo;
    nsopi_[2] = eoe;
    nsopi_[3] = eoo;
    nsopi_[4] = oee;
    nsopi_[5] = oeo;
    nsopi_[6] = ooe;
    nsopi_[7] = ooo;
    int tmp = 0;
    double tmp_energy = 0;
    double max_energy = E[electrons/2];
    int* tmp_swap = (int*)malloc(3*sizeof(int));
    for(int i = 0; i < orbitalMax; i++){
        if(MO[i][0]%2==0 && MO[i][1]%2==0 && MO[i][2]%2==0){
           tmp_swap[0]=MO[tmp][0];
           tmp_swap[1]=MO[tmp][1];
           tmp_swap[2]=MO[tmp][2];
           tmp_energy = E[tmp];
           MO[tmp][0]=MO[i][0];
           MO[tmp][1]=MO[i][1];
           MO[tmp][2]=MO[i][2];
           E[tmp] = E[i];
           tmp++;
           MO[i][0]=tmp_swap[0];
           MO[i][1]=tmp_swap[1];
           MO[i][2]=tmp_swap[2];
           E[i]=tmp_energy;
        }
   
    }
    for(int i = 0; i < orbitalMax; i++){
        if(MO[i][0]%2==0 && MO[i][1]%2==0 && MO[i][2]%2==1){
           tmp_swap[0]=MO[tmp][0];
           tmp_swap[1]=MO[tmp][1];
           tmp_swap[2]=MO[tmp][2];
           tmp_energy = E[tmp];
           MO[tmp][0]=MO[i][0];
           MO[tmp][1]=MO[i][1];
           MO[tmp][2]=MO[i][2];
           E[tmp] = E[i];
           tmp++;
           MO[i][0]=tmp_swap[0];
           MO[i][1]=tmp_swap[1];
           MO[i][2]=tmp_swap[2];
           E[i]=tmp_energy;
        }
   
    }
    for(int i = 0; i < orbitalMax; i++){
        if(MO[i][0]%2==0 && MO[i][1]%2==1 && MO[i][2]%2==0){
           tmp_swap[0]=MO[tmp][0];
           tmp_swap[1]=MO[tmp][1];
           tmp_swap[2]=MO[tmp][2];
           tmp_energy = E[tmp];
           MO[tmp][0]=MO[i][0];
           MO[tmp][1]=MO[i][1];
           MO[tmp][2]=MO[i][2];
           E[tmp] = E[i];
           tmp++;
           MO[i][0]=tmp_swap[0];
           MO[i][1]=tmp_swap[1];
           MO[i][2]=tmp_swap[2];
           E[i]=tmp_energy;
        }
   
    }
    for(int i = 0; i < orbitalMax; i++){
        if(MO[i][0]%2==0 && MO[i][1]%2==1 && MO[i][2]%2==1){
           tmp_swap[0]=MO[tmp][0];
           tmp_swap[1]=MO[tmp][1];
           tmp_swap[2]=MO[tmp][2];
           tmp_energy = E[tmp];
           MO[tmp][0]=MO[i][0];
           MO[tmp][1]=MO[i][1];
           MO[tmp][2]=MO[i][2];
           E[tmp] = E[i];
           tmp++;
           MO[i][0]=tmp_swap[0];
           MO[i][1]=tmp_swap[1];
           MO[i][2]=tmp_swap[2];
           E[i]=tmp_energy;
        }
   
    }
    for(int i = 0; i < orbitalMax; i++){
        if(MO[i][0]%2==1 && MO[i][1]%2==0 && MO[i][2]%2==0){
           tmp_swap[0]=MO[tmp][0];
           tmp_swap[1]=MO[tmp][1];
           tmp_swap[2]=MO[tmp][2];
           tmp_energy = E[tmp];
           MO[tmp][0]=MO[i][0];
           MO[tmp][1]=MO[i][1];
           MO[tmp][2]=MO[i][2];
           E[tmp] = E[i];
           tmp++;
           MO[i][0]=tmp_swap[0];
           MO[i][1]=tmp_swap[1];
           MO[i][2]=tmp_swap[2];
           E[i]=tmp_energy;
        }
   
    }
    for(int i = 0; i < orbitalMax; i++){
        if(MO[i][0]%2==1 && MO[i][1]%2==0 && MO[i][2]%2==1){
           tmp_swap[0]=MO[tmp][0];
           tmp_swap[1]=MO[tmp][1];
           tmp_swap[2]=MO[tmp][2];
           tmp_energy = E[tmp];
           MO[tmp][0]=MO[i][0];
           MO[tmp][1]=MO[i][1];
           MO[tmp][2]=MO[i][2];
           E[tmp] = E[i];
           tmp++;
           MO[i][0]=tmp_swap[0];
           MO[i][1]=tmp_swap[1];
           MO[i][2]=tmp_swap[2];
           E[i]=tmp_energy;
        }
   
    }
    for(int i = 0; i < orbitalMax; i++){
        if(MO[i][0]%2==1 && MO[i][1]%2==1 && MO[i][2]%2==0){
           tmp_swap[0]=MO[tmp][0];
           tmp_swap[1]=MO[tmp][1];
           tmp_swap[2]=MO[tmp][2];
           tmp_energy = E[tmp];
           MO[tmp][0]=MO[i][0];
           MO[tmp][1]=MO[i][1];
           MO[tmp][2]=MO[i][2];
           E[tmp] = E[i];
           tmp++;
           MO[i][0]=tmp_swap[0];
           MO[i][1]=tmp_swap[1];
           MO[i][2]=tmp_swap[2];
           E[i]=tmp_energy;
        }
   
    }
    Eirrep_ = (int*)malloc(nirrep_*sizeof(int));
    int offsetJ = 0;
    int ecounter = 0;
    for(int i = 0; i < nirrep_; i++){
        for(int j = 0; j < nsopi_[i]; j++){
            if(E[offsetJ + j] < max_energy && ecounter < electrons/2){
                Eirrep_[i]++;
                ecounter++;
            }
        }
        offsetJ += nsopi_[i];
    }
    offsetJ = 0;
    for(int i = 0; i < nirrep_; i++){
        for(int j = 0; j < nsopi_[i]; j++){
            if(E[offsetJ + j] == max_energy && ecounter < electrons/2){
                Eirrep_[i]++;
                ecounter++;
            }
        }
        offsetJ += nsopi_[i];
    }
    //printf("max energy%f\n",max_energy);
    for(int i = 0; i < nirrep_; i++){
        //printf("electrons[%d] with <= max energy %d\n",i,Eirrep_[i]);
    }
    offsetJ = 0;
    for(int i = 0; i < nirrep_; i++){
       for(int j = 0; j < nsopi_[i]; j++){
          for(int k = j+1; k < nsopi_[i]; k++){
             if(E[j+offsetJ]>E[k+offsetJ]){
                tmp_swap[0] = MO[j+offsetJ][0];     
                tmp_swap[1] = MO[j+offsetJ][1];     
                tmp_swap[2] = MO[j+offsetJ][2];
                tmp_energy = E[j+offsetJ];
                MO[j+offsetJ][0] = MO[k+offsetJ][0];    
                MO[j+offsetJ][1] = MO[k+offsetJ][1];    
                MO[j+offsetJ][2] = MO[k+offsetJ][2];
                E[j+offsetJ] = E[k+offsetJ];
                MO[k+offsetJ][0] = tmp_swap[0];    
                MO[k+offsetJ][1] = tmp_swap[1];    
                MO[k+offsetJ][2] = tmp_swap[2]; 
                E[k+offsetJ] = tmp_energy; 
             }
          }
       }     
       offsetJ += nsopi_[i];
    }
    for(int i = 0; i < orbitalMax; i++){
       printf("MO[%d][0]:%d\tMO[%d][1]:%d\tMO[%d][2]:%d energy: %f\n",i,MO[i][0],i,MO[i][1],i,MO[i][2],E[i]);
    }
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

int JelliumIntegrals::get_nmax(){
    return nmax;
}

}} // end of namespaces
