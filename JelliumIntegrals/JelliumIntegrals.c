# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>

//int main ( int argc, char *argv[] );
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
//const double pi=4*atan(1.0);

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
/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
 *   Purpose:
 *
 *   MAIN is the main program for LEGENDRE_RULE_FAST.
 *
 *   Discussion:
 *
 *   This program computes a Gauss-Legendre quadrature rule
 *   and writes it to a file.
 *
 *   Usage:
 *
 *   legendre_rule_fast n a b
 *
 *    where
 *
 *    * n is the number of points in the rule;
 *    * a is the left endpoint;
 *    * b is the right endpoint.
 *
 *    Licensing:
 *
 *    This code is distributed under the GNU LGPL license. 
 *
 *    Modified:
 *
 *    28 September 2013
 *
 *    Author:
 *
 *    John Burkardt
 */  
{
  double a;
  double b;
  int n;

  timestamp ( );

  printf ( "\n" );
  printf ( "LEGENDRE_RULE\n" );
  printf ( "  C version\n" );
  printf ( "\n" );
  printf ( "  Compiled on %s at %s.\n", __DATE__, __TIME__ );
  printf ( "\n" );
  printf ( "  Compute a Gauss-Legendre rule for approximating\n" );
  printf ( "\n" );
  printf ( "    Integral ( a <= x <= b ) f(x) dx\n" );
  printf ( "\n" );
  printf ( "  of order N.\n" );
  printf ( "\n" );
  printf ( "  The user specifies N, A, and B.\n" );
  printf ( "\n" );
  printf ( "  The rule is stored in 3 files:\n" );
  printf ( "\n" );
  printf ( "  * leg_oN_w.txt - the weight file\n" );
  printf ( "  * leg_oN_x.txt - the abscissa file.\n" );
  printf ( "  * leg_oN_r.txt - the region file.\n" );
  

/*
 *   Get N.
 *   */

  if ( 1 < argc )
  {
    n = atoi ( argv[1] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter N:\n" );
    scanf ( "%d", &n );
  }
  printf ( "\n" );
  printf ( "  N = %d\n", n );

/*
 *   Get A:
 *   */
  if ( 2 < argc )
  {
    a = atof ( argv[2] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter A.\n" );
    scanf ( "%lf", &a );
  }
  printf ( "\n" );
  printf ( "  A = %g\n", a );

/*
 *   Get B:
 *   */
  if ( 3 < argc )
  {
    b = atof ( argv[3] );
  }
  else
  {
    printf ( "\n" );
    printf ( "  Enter B.\n" );
    scanf ( "%lf", &b );
  }
  printf ( "\n" );
  printf ( "  B = %g\n", b );


/*

 *   Construct the rule and output it.
 *   */


 legendre_handle ( n, a, b );
/*
 *   Terminate.
 *   */
  printf ( "\n" );
  printf ( "LEGENDRE_RULE_FAST:\n" );
  printf ( "  Normal end of execution.\n" );

  printf ( "\n" );
  timestamp ( );


  double *x, *w;
  double *mu, *nu, *lam, *sig;

  x = (double *)malloc(n*sizeof(double));
  w = (double *)malloc(n*sizeof(double));
  
  mu = (double *)malloc(3*sizeof(double));
  nu = (double *)malloc(3*sizeof(double));
  lam = (double *)malloc(3*sizeof(double));
  sig = (double *)malloc(3*sizeof(double));

  
  int nmax=4;
  int *ORBE, **MO;

  ORBE = VEC_INT(3*nmax*nmax*nmax);
  MO = MAT_INT(3*nmax*nmax*nmax,3);

  int orbitalMax = 26;

  OrderPsis3D(nmax, ORBE, MO);

  //  Constructe grid and weights, store them to the vectors x and w, respectively.
  //  This is one of John Burkhardt's library functions
  legendre_compute_glr(n, x, w);
  // Scale the grid to start at value a and end at value b. 
  // We want our integration range to go from 0 to 1, so a = 0, b = 1
  // This is also one of John Burkhardt's library functions
  rescale( a, b, n, x, w);
 
  FILE *erifp, *nucfp, *kinfp, *selffp;
  double erival, nucval, kinval, selfval;
  erifp = fopen("ERI.dat","w");
  nucfp = fopen("NucAttraction.dat","w");
  kinfp = fopen("Kinetic.dat","w");
  selffp = fopen("SelfEnergy.dat","w");

  // build g tensor g[npq] * w[n]
  printf("\n");
  printf("    build g tensor......."); fflush(stdout);
  double * g_tensor = (double*)malloc(n * orbitalMax * orbitalMax * sizeof(double));
  memset((void*)g_tensor,'\0',n * orbitalMax * orbitalMax * sizeof(double));
  for (int pt = 0; pt < n; pt++) {
      double xval = x[pt];
      for (int p = 0; p < orbitalMax; p++) {
          for (int q = 0; q < orbitalMax; q++) {
              g_tensor[pt*orbitalMax*orbitalMax+p*orbitalMax+q] = g_pq(p, q, xval) * w[pt];
          }
      }
  }
  printf("done.\n");

  // build sqrt(x*x+y*y+z*z)
  printf("    build sqrt tensor...."); fflush(stdout);
  double * sqrt_tensor = (double*)malloc(n*n*n*sizeof(double));
  for (int i = 0; i < n; i++) {
      double xval = x[i];
      for (int j = 0; j < n; j++) {
          double yval = x[j];
          for (int k = 0; k < n; k++) {
              double zval = x[k];
              double val = sqrt(xval*xval+yval*yval+zval*zval);
              sqrt_tensor[i*n*n + j*n + k] = 1.0/val;
          }
      }
  }
  printf("done.\n");


  // now, compute (P|Q)
  printf("    build (P|Q).........."); fflush(stdout);
  int ***PQmap = (int ***)malloc((2*nmax+1)*sizeof(int**));
  for (int i = 0; i < 2*nmax+1; i++) {
      PQmap[i] = (int **)malloc((2*nmax+1)*sizeof(int*));
      for (int j = 0; j < 2*nmax+1; j++) {
          PQmap[i][j] = (int *)malloc((2*nmax+1)*sizeof(int));
          for (int k = 0; k < 2*nmax+1; k++) {
              PQmap[i][j][k] = -999;
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


  int start_pq = clock();
  double ** PQ = (double**)malloc(Pdim * sizeof(double*));
  for (int p = 0; p < Pdim; p++) {
      PQ[p] = (double*)malloc(Pdim * sizeof(double));
      for (int q = 0; q < Pdim; q++) {
          PQ[p][q] = 0.0;
      }
  }

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


                          //if ( P > Q ) continue;

                          double dum = pq_int_new(n, px, py, pz, qx, qy, qz, g_tensor,orbitalMax,sqrt_tensor);
                          
                          int P,Q;

                          // start 

                          P = PQmap[px][py][pz];
                          Q = PQmap[qx][qy][qz];
                          PQ[P][Q] = dum;

                          P = PQmap[qx][py][pz];
                          Q = PQmap[px][qy][qz];
                          PQ[P][Q] = dum;

                          P = PQmap[px][qy][pz];
                          Q = PQmap[qx][py][qz];
                          PQ[P][Q] = dum;

                          P = PQmap[qx][qy][pz];
                          Q = PQmap[px][py][qz];
                          PQ[P][Q] = dum;

                          P = PQmap[px][py][qz];
                          Q = PQmap[qx][qy][pz];
                          PQ[P][Q] = dum;

                          P = PQmap[qx][py][qz];
                          Q = PQmap[px][qy][pz];
                          PQ[P][Q] = dum;

                          P = PQmap[px][qy][qz];
                          Q = PQmap[qx][py][pz];
                          PQ[P][Q] = dum;

                          P = PQmap[qx][qy][qz];
                          Q = PQmap[px][py][pz];
                          PQ[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[py][px][pz];
                          Q = PQmap[qy][qx][qz];
                          PQ[P][Q] = dum;

                          P = PQmap[py][qx][pz];
                          Q = PQmap[qy][px][qz];
                          PQ[P][Q] = dum;

                          P = PQmap[qy][px][pz];
                          Q = PQmap[py][qx][qz];
                          PQ[P][Q] = dum;

                          P = PQmap[qy][qx][pz];
                          Q = PQmap[py][px][qz];
                          PQ[P][Q] = dum;

                          P = PQmap[py][px][qz];
                          Q = PQmap[qy][qx][pz];
                          PQ[P][Q] = dum;

                          P = PQmap[py][qx][qz];
                          Q = PQmap[qy][px][pz];
                          PQ[P][Q] = dum;

                          P = PQmap[qy][px][qz];
                          Q = PQmap[py][qx][pz];
                          PQ[P][Q] = dum;

                          P = PQmap[qy][qx][qz];
                          Q = PQmap[py][px][pz];
                          PQ[P][Q] = dum;

                          // now begins pxqx < pyqy < pzqz

                          P = PQmap[pz][px][py];
                          Q = PQmap[qz][qx][qy];
                          PQ[P][Q] = dum;

                          P = PQmap[pz][qx][py];
                          Q = PQmap[qz][px][qy];
                          PQ[P][Q] = dum;

                          P = PQmap[pz][px][qy];
                          Q = PQmap[qz][qx][py];
                          PQ[P][Q] = dum;

                          P = PQmap[pz][qx][qy];
                          Q = PQmap[qz][px][py];
                          PQ[P][Q] = dum;

                          P = PQmap[qz][px][py];
                          Q = PQmap[pz][qx][qy];
                          PQ[P][Q] = dum;

                          P = PQmap[qz][qx][py];
                          Q = PQmap[pz][px][qy];
                          PQ[P][Q] = dum;

                          P = PQmap[qz][px][qy];
                          Q = PQmap[pz][qx][py];
                          PQ[P][Q] = dum;

                          P = PQmap[qz][qx][qy];
                          Q = PQmap[pz][px][py];
                          PQ[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[pz][py][px];
                          Q = PQmap[qz][qy][qx];
                          PQ[P][Q] = dum;

                          P = PQmap[pz][py][qx];
                          Q = PQmap[qz][qy][px];
                          PQ[P][Q] = dum;

                          P = PQmap[pz][qy][px];
                          Q = PQmap[qz][py][qx];
                          PQ[P][Q] = dum;

                          P = PQmap[pz][qy][qx];
                          Q = PQmap[qz][py][px];
                          PQ[P][Q] = dum;

                          P = PQmap[qz][py][px];
                          Q = PQmap[pz][qy][qx];
                          PQ[P][Q] = dum;

                          P = PQmap[qz][py][qx];
                          Q = PQmap[pz][qy][px];
                          PQ[P][Q] = dum;

                          P = PQmap[qz][qy][px];
                          Q = PQmap[pz][py][qx];
                          PQ[P][Q] = dum;

                          P = PQmap[qz][qy][qx];
                          Q = PQmap[pz][py][px];
                          PQ[P][Q] = dum;

                          // now begins last set of 16

                          P = PQmap[px][pz][py];
                          Q = PQmap[qx][qz][qy];
                          PQ[P][Q] = dum;

                          P = PQmap[qx][pz][py];
                          Q = PQmap[px][qz][qy];
                          PQ[P][Q] = dum;

                          P = PQmap[px][pz][qy];
                          Q = PQmap[qx][qz][py];
                          PQ[P][Q] = dum;

                          P = PQmap[qx][pz][qy];
                          Q = PQmap[px][qz][py];
                          PQ[P][Q] = dum;

                          P = PQmap[px][qz][py];
                          Q = PQmap[qx][pz][qy];
                          PQ[P][Q] = dum;

                          P = PQmap[qx][qz][py];
                          Q = PQmap[px][pz][qy];
                          PQ[P][Q] = dum;

                          P = PQmap[px][qz][qy];
                          Q = PQmap[qx][pz][py];
                          PQ[P][Q] = dum;

                          P = PQmap[qx][qz][qy];
                          Q = PQmap[px][pz][py];
                          PQ[P][Q] = dum;

                          // pxqx - pyqy

                          P = PQmap[py][pz][px];
                          Q = PQmap[qy][qz][qx];
                          PQ[P][Q] = dum;

                          P = PQmap[py][pz][qx];
                          Q = PQmap[qy][qz][px];
                          PQ[P][Q] = dum;

                          P = PQmap[qy][pz][px];
                          Q = PQmap[py][qz][qx];
                          PQ[P][Q] = dum;

                          P = PQmap[qy][pz][qx];
                          Q = PQmap[py][qz][px];
                          PQ[P][Q] = dum;

                          P = PQmap[py][qz][px];
                          Q = PQmap[qy][pz][qx];
                          PQ[P][Q] = dum;

                          P = PQmap[py][qz][qx];
                          Q = PQmap[qy][pz][px];
                          PQ[P][Q] = dum;

                          P = PQmap[qy][qz][px];
                          Q = PQmap[py][pz][qx];
                          PQ[P][Q] = dum;

                          P = PQmap[qy][qz][qx];
                          Q = PQmap[py][pz][px];
                          PQ[P][Q] = dum;

                      }
                  }
              }
          }
      }
  }
  int end_pq = clock();
  printf("done.\n");fflush(stdout);
  printf("\n");
  printf("    time for (P|Q) construction: %12.6f\n",(double)(end_pq-start_pq)/CLOCKS_PER_SEC); fflush(stdout);
  printf("\n");
  

  // Four nested loops to compute lower triange of electron repulsion integrals - roughly have of the non-unique integrals
  // will not be computed, but this is still not exploiting symmetry fully
  printf("    build ERIs...........");fflush(stdout);
  int start = clock();
  for (int i=0; i<orbitalMax; i++) {
      mu[0] = MO[i][0];
      mu[1] = MO[i][1];
      mu[2] = MO[i][2];

      for (int j=i; j<orbitalMax; j++) {
          nu[0] = MO[j][0];
          nu[1] = MO[j][1];
          nu[2] = MO[j][2];

          // Lower triangle of 1-electron integrals will be computed, fully exploiting symmetry (I think!)
          // Kinetic Energy Integrals - already computed and stored in ORBE vector    
          if (i==j) { 
              kinval = 0.5*ORBE[i];
          }
          else {
              kinval = 0.;
          }
          // Print Kinetic Energy Integral to file
          fprintf(kinfp,"  %i  %i  %17.14f\n",i+1,j+1,kinval);

          // Nuclear-attraction Integrals
          nucval = Vab_Int(n, x, w, mu, nu);
          // Print Nuclear-attraction integral to file
          fprintf(nucfp, "  %i  %i  %17.14f\n",i+1,j+1,nucval);      

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
                  erival = ERI_new(n, x, mu, nu, lam, sig, g_tensor, orbitalMax, sqrt_tensor, PQ, PQmap);
                  //double dum = ERI(n, x, w, mu, nu, lam, sig);
                  //if ( fabs(erival - dum) > 1e-14 ) {
                  //    printf("uh-oh. %20.12lf %20.12lf\n",erival,dum);
                  //}else {
                  //    //printf("sweet! %20.12lf %20.12lf\n",erival,dum);
                  //}

                  // Print ERI to file
                  fprintf(erifp," %i  %i  %i  %i  %17.14f\n",i+1,j+1,k+1,l+1,erival);
              }
          }
      }
  }
  int end = clock();
  printf("done.\n");fflush(stdout);
  printf("\n");
  printf("    time for eri construction:   %12.6f\n",(double)(end-start)/CLOCKS_PER_SEC); fflush(stdout);
  printf("\n");

  // Compute self energy
  selfval = E0_Int(n, x, w);
  // Print to file
  fprintf(selffp, "  %17.14f\n",selfval); 
  

  free(x);
  free(w);
  free(mu);
  free(nu);
  free(lam);
  free(sig);
  fclose(erifp);
  fclose(nucfp);
  fclose(kinfp);
  fclose(selffp);
  return 0;



}
/******************************************************************************/

void legendre_compute_glr ( int n, double x[], double w[] )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       LEGENDRE_COMPUTE_GLR: Legendre quadrature by the Glaser-Liu-Rokhlin method.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license. 
 *
 *               Modified:
 *
 *                   19 October 2009
 *
 *                     Author:
 *
 *                         Original C version by Nick Hale.
 *                             This C version by John Burkardt.
 *
 *                               Reference:
 *
 *                                   Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
 *                                       A fast algorithm for the calculation of the roots of special functions, 
 *                                           SIAM Journal on Scientific Computing,
 *                                               Volume 29, Number 4, pages 1420-1438, 2007.
 *
 *                                                 Parameters:
 *
 *                                                     Input, int N, the order.
 *
 *                                                         Output, double X[N], the abscissas.
 *
 *                                                             Output, double W[N], the weights.
 *                                                             */
{
  int i;
  double p;
  double pp;
  double w_sum;
/*
 *   Get the value and derivative of the N-th Legendre polynomial at 0.
 *   */
  legendre_compute_glr0 ( n, &p, &pp );
/*
 *   Either zero is a root, or we have to call a function to find the first root.
 *   */  
  if ( n % 2 == 1 )
  {
    x[(n-1)/2] = p;
    w[(n-1)/2] = pp;
  }
  else
  {
    legendre_compute_glr2 ( p, n, &x[n/2], &w[n/2] );
  }
/*
 *   Get the complete set of roots and derivatives.
 *   */
  legendre_compute_glr1 ( n, x, w );
/*
 *   Compute the weights.
 *   */
  for ( i = 0; i < n; i++ )
  {
    w[i] = 2.0 / ( 1.0 - x[i] ) / ( 1.0 + x[i] ) / w[i] / w[i];
  }
  w_sum = 0.0;
  for ( i = 0; i < n; i++ )
  {
    w_sum = w_sum + w[i];
  }
  for ( i = 0; i < n; i++ )
  {
    w[i] = 2.0 * w[i] / w_sum;
  }
  return;
}
/******************************************************************************/

void legendre_compute_glr0 ( int n, double *p, double *pp )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       LEGENDRE_COMPUTE_GLR0 gets a starting value for the fast algorithm.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license. 
 *
 *               Modified:
 *
 *                   19 October 2009
 *
 *                     Author:
 *
 *                         Original C version by Nick Hale.
 *                             This C version by John Burkardt.
 *
 *                               Reference:
 *
 *                                   Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
 *                                       A fast algorithm for the calculation of the roots of special functions, 
 *                                           SIAM Journal on Scientific Computing,
 *                                               Volume 29, Number 4, pages 1420-1438, 2007.
 *
 *                                                 Parameters:
 *
 *                                                     Input, int N, the order of the Legendre polynomial.
 *
 *                                                         Output, double *P, *PP, the value of the N-th Legendre polynomial
 *                                                             and its derivative at 0.
 *                                                             */
{
  double dk;
  int k;
  double pm1;
  double pm2;
  double ppm1;
  double ppm2;

  pm2 = 0.0;
  pm1 = 1.0;
  ppm2 = 0.0;
  ppm1 = 0.0;

  for ( k = 0; k < n; k++ )
  {
    dk = ( double ) k;
    *p = - dk * pm2 / ( dk + 1.0 );
    *pp = ( ( 2.0 * dk + 1.0 ) * pm1 - dk * ppm2 ) / ( dk + 1.0 );
    pm2 = pm1;
    pm1 = *p;
    ppm2 = ppm1;
    ppm1 = *pp;
  }
  return;
}
/******************************************************************************/

void legendre_compute_glr1 ( int n, double *x, double *ders )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       LEGENDRE_COMPUTE_GLR1 gets the complete set of Legendre points and weights.
 *
 *         Discussion:
 *
 *             This routine requires that a starting estimate be provided for one
 *                 root and its derivative.  This information will be stored in entry
 *                     (N+1)/2 if N is odd, or N/2 if N is even, of ROOTS and DERS.
 *
 *                       Licensing:
 *
 *                           This code is distributed under the GNU LGPL license. 
 *
 *                             Modified:
 *
 *                                 19 October 2009
 *
 *                                   Author:
 *
 *                                       Original C version by Nick Hale.
 *                                           This C version by John Burkardt.
 *
 *                                             Reference:
 *
 *                                                 Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
 *                                                     A fast algorithm for the calculation of the roots of special functions, 
 *                                                         SIAM Journal on Scientific Computing,
 *                                                             Volume 29, Number 4, pages 1420-1438, 2007.
 *
 *                                                               Parameters:
 *
 *                                                                   Input, int N, the order of the Legendre polynomial.
 *
 *                                                                       Input/output, double X[N].  On input, a starting value
 *                                                                           has been set in one entry.  On output, the roots of the Legendre 
 *                                                                               polynomial.
 *
 *                                                                                   Input/output, double DERS[N].  On input, a starting value
 *                                                                                       has been set in one entry.  On output, the derivatives of the Legendre 
 *                                                                                           polynomial at the zeros.
 *
 *                                                                                             Local Parameters:
 *
 *                                                                                                 Local, int M, the number of terms in the Taylor expansion.
 *                                                                                                 */
{
  double dk;
  double dn;
  double h;
  int j;
  int k;
  int l;
  int m = 30;
  int n2;
  const double pi = 3.141592653589793;
  int s;
  double *u;
  double *up;
  double xp;

  if ( n % 2 == 1 )
  {
    n2 = ( n - 1 ) / 2;
    s = 1;
  }
  else
  {
    n2 = n / 2;
    s = 0;
  }

  u = ( double * ) malloc ( ( m + 2 ) * sizeof ( double ) );
  up = ( double * ) malloc ( ( m + 1 ) * sizeof ( double ) );

  dn = ( double ) n;

  for ( j = n2; j < n - 1; j++ )
  {
    xp = x[j];

    h = rk2_leg ( pi/2.0, -pi/2.0, xp, n ) - xp;

    u[0] = 0.0;
    u[1] = 0.0;
    u[2] = ders[j];

    up[0] = 0.0;
    up[1] = u[2];

    for ( k = 0; k <= m - 2; k++ )
    {
      dk = ( double ) k;

      u[k+3] = 
      ( 
        2.0 * xp * ( dk + 1.0 ) * u[k+2]
        + ( dk * ( dk + 1.0 ) - dn * ( dn + 1.0 ) ) * u[k+1] / ( dk + 1.0 )
      ) / ( 1.0 - xp ) / ( 1.0 + xp ) / ( dk + 2.0 );

      up[k+2] = ( dk + 2.0 ) * u[k+3];
    }

    for ( l = 0; l < 5; l++ )
    { 
      h = h - ts_mult ( u, h, m ) / ts_mult ( up, h, m-1 );
    }

    x[j+1] = xp + h;
    ders[j+1] = ts_mult ( up, h, m-1 );
  }

  free ( u );
  free ( up );

  for ( k = 0; k < n2 + s; k++ )
  {
    x[k] = - x[n-k-1];
    ders[k] = ders[n-k-1];
  }
  return;
}
/******************************************************************************/

void legendre_compute_glr2 ( double pn0, int n, double *x1,  double *d1 )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       LEGENDRE_COMPUTE_GLR2 finds the first real root.
 *
 *         Discussion:
 *
 *             This routine is only called if N is even.
 *
 *                 Thanks to Morten Welinder, for pointing out a typographical error
 *                     in indexing, 17 May 2013.
 *
 *                       Licensing:
 *
 *                           This code is distributed under the GNU LGPL license. 
 *
 *                             Modified:
 *
 *                                 17 May 2013
 *
 *                                   Author:
 *
 *                                       Original C version by Nick Hale.
 *                                           This C version by John Burkardt.
 *
 *                                             Reference:
 *
 *                                                 Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
 *                                                     A fast algorithm for the calculation of the roots of special functions, 
 *                                                         SIAM Journal on Scientific Computing,
 *                                                             Volume 29, Number 4, pages 1420-1438, 2007.
 *
 *                                                               Parameters:
 *
 *                                                                   Input, double PN0, the value of the N-th Legendre polynomial at 0.
 *
 *                                                                       Input, int N, the order of the Legendre polynomial.
 *
 *                                                                           Output, double *X1, the first real root.
 *
 *                                                                               Output, double *D1, the derivative at X1.
 *
 *                                                                                 Local Parameters:
 *
 *                                                                                     Local, int M, the number of terms in the Taylor expansion.
 *                                                                                     */
{
  double dk;
  double dn;
  int k;
  int l;
  int m = 30;
  const double pi = 3.141592653589793;
  double t;
  double *u;
  double *up;

  t = 0.0;
  *x1 = rk2_leg ( t, -pi/2.0, 0.0, n );

  u = ( double * ) malloc ( ( m + 2 ) * sizeof ( double ) );
  up = ( double * ) malloc ( ( m + 1 ) * sizeof ( double ) );

  dn = ( double ) n;
/*
 *   U[0] and UP[0] are never used.
 *     U[M+1] is set, but not used, and UP[M] is set and not used.
 *       What gives?
 *       */
  u[0] = 0.0;
  u[1] = pn0;

  up[0] = 0.0;
 
  for ( k = 0; k <= m - 2; k = k + 2 )
  {
    dk = ( double ) k;

    u[k+2] = 0.0;
    u[k+3] = ( dk * ( dk + 1.0 ) - dn * ( dn + 1.0 ) ) * u[k+1]
      / ( dk + 1.0 ) / ( dk + 2.0 );
 
    up[k+1] = 0.0;
    up[k+2] = ( dk + 2.0 ) * u[k+3];
  }
  
  for ( l = 0; l < 5; l++ )
  {
    *x1 = *x1 - ts_mult ( u, *x1, m ) / ts_mult ( up, *x1, m-1 );
  }
  *d1 = ts_mult ( up, *x1, m-1 );

  free ( u );
  free ( up) ;

  return;
}
/******************************************************************************/

void legendre_handle ( int n, double a, double b )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       LEGENDRE_HANDLE computes the requested Gauss-Legendre rule and outputs it.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license. 
 *
 *               Modified:
 *
 *                   22 October 2009
 *
 *                     Author:
 *
 *                         John Burkardt
 *
 *                           Parameters:
 *
 *                               Input, int N, the order of the rule.
 *
 *                                   Input, double A, B, the left and right endpoints.
 *                                   */ 
{
  int i;
  char output_r[255];
  char output_w[255];
  char output_x[255];
  double *r;
  double t;
  double *w;
  double *x;

  r = ( double * ) malloc ( 2 * sizeof ( double ) );
  w = ( double * ) malloc ( n * sizeof ( double ) );
  x = ( double * ) malloc ( n * sizeof ( double ) );

  r[0] = a;
  r[1] = b;
/*
 *   Compute the rule.
 *   */
  t = wtime ( );
  legendre_compute_glr ( n, x, w );
  t = wtime ( ) - t;

  printf ( "\n" );
  printf ( "  Elapsed time during computation was %g seconds.\n", t );
/*
 *   Rescale the rule to [A,B].
 *   */
  rescale ( a, b, n, x, w );
/*
 *   Uncomment to Write the rule to 3 files.
 *   

  sprintf ( output_w, "leg_o%d_w.txt", n );
  sprintf ( output_x, "leg_o%d_x.txt", n );
  sprintf ( output_r, "leg_o%d_r.txt", n );

  printf ( "\n" );
  printf ( "  Weight file will be   \"%s\".\n", output_w );
  printf ( "  Abscissa file will be \"%s\".\n", output_x );
  printf ( "  Region file will be   \"%s\".\n", output_r );
            
  r8mat_write ( output_w, 1, n, w );
  r8mat_write ( output_x, 1, n, x );
  r8mat_write ( output_r, 1, 2, r );
  */
  free ( r );
  free ( w );
  free ( x );

  return;
}
/******************************************************************************/

void r8mat_write ( char *output_filename, int m, int n, double table[] )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       R8MAT_WRITE writes an R8MAT file.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license. 
 *
 *               Modified:
 *
 *                   01 June 2009
 *
 *                     Author:
 *
 *                         John Burkardt
 *
 *                           Parameters:
 *
 *                               Input, char *OUTPUT_FILENAME, the output filename.
 *
 *                                   Input, int M, the spatial dimension.
 *
 *                                       Input, int N, the number of points.
 *
 *                                           Input, double TABLE[M*N], the table data.
 *                                           */
{
  int i;
  int j;
  FILE *output;
/*
 *   Open the file.
 *   */
  output = fopen ( output_filename, "wt" );

  if ( !output )
  {
    printf ( "\n" );
    printf ( "R8MAT_WRITE - Fatal error!\n" );
    printf ( "  Could not open the output file.\n" );
    return;
  }
/*
 *   Write the data.
 *   */
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      fprintf ( output, "  %24.16g", table[i+j*m] );
    }
    fprintf ( output, "\n" );
  }
/*
 *   Close the file.
 *   */
  fclose ( output );

  return;
}
/******************************************************************************/

void rescale ( double a, double b, int n, double x[], double w[] )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       RESCALE rescales a Legendre quadrature rule from [-1,+1] to [A,B].
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license. 
 *
 *               Modified:
 *
 *                   22 October 2009
 *
 *                     Author:
 *
 *                         Original MATLAB version by Nick Hale.
 *                             C version by John Burkardt.
 *
 *                               Reference:
 *
 *                                   Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
 *                                       A fast algorithm for the calculation of the roots of special functions, 
 *                                           SIAM Journal on Scientific Computing,
 *                                               Volume 29, Number 4, pages 1420-1438, 2007.
 *
 *                                                 Parameters:
 *
 *                                                     Input, double A, B, the endpoints of the new interval.
 *
 *                                                         Input, int N, the order.
 *
 *                                                             Input/output, double X[N], on input, the abscissas for [-1,+1].
 *                                                                 On output, the abscissas for [A,B].
 *
 *                                                                     Input/output, double W[N], on input, the weights for [-1,+1].
 *                                                                         On output, the weights for [A,B].
 *                                                                         */
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    x[i] = ( ( a + b ) + ( b - a ) * x[i] ) / 2.0;
  }
  for ( i = 0; i < n; i++ )
  {
    w[i] = ( b - a ) * w[i] / 2.0;
  }
  return;
}
/******************************************************************************/

double rk2_leg ( double t1, double t2, double x, int n )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       RK2_LEG advances the value of X(T) using a Runge-Kutta method.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license. 
 *
 *               Modified:
 *
 *                   22 October 2009
 *
 *                     Author:
 *
 *                         Original C version by Nick Hale.
 *                             This C version by John Burkardt.
 *
 *                               Parameters:
 *
 *                                   Input, double T1, T2, the range of the integration interval.
 *
 *                                       Input, double X, the value of X at T1.
 *
 *                                           Input, int N, the number of steps to take.
 *
 *                                               Output, double RK2_LEG, the value of X at T2.
 *                                               */
{
  double f;
  double h;
  int j;
  double k1;
  double k2;
  int m = 10;
  double snn1;
  double t;

  h = ( t2 - t1 ) / ( double ) m;
  snn1 = sqrt ( ( double ) ( n * ( n + 1 ) ) );

  t = t1;

  for ( j = 0; j < m; j++ )
  {
    f = ( 1.0 - x ) * ( 1.0 + x );
    k1 = - h * f / ( snn1 * sqrt ( f ) - 0.5 * x * sin ( 2.0 * t ) );
    x = x + k1;

    t = t + h;

    f = ( 1.0 - x ) * ( 1.0 + x );
    k2 = - h * f / ( snn1 * sqrt ( f ) - 0.5 * x * sin ( 2.0 * t ) );   
    x = x + 0.5 * ( k2 - k1 );
  }
  return x;
}
/******************************************************************************/

void timestamp ( void )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       TIMESTAMP prints the current YMDHMS date as a time stamp.
 *
 *         Example:
 *
 *             31 May 2001 09:45:54 AM
 *
 *               Licensing:
 *
 *                   This code is distributed under the GNU LGPL license. 
 *
 *                     Modified:
 *
 *                         24 September 2003
 *
 *                           Author:
 *
 *                               John Burkardt
 *
 *                                 Parameters:
 *
 *                                     None
 *                                     */
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
/******************************************************************************/

double ts_mult ( double *u, double h, int n )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       TS_MULT evaluates a polynomial.
 *
 *         Discussion:
 *
 *             TS_MULT = U[1] + U[2] * H + ... + U[N] * H^(N-1).
 *
 *               Licensing:
 *
 *                   This code is distributed under the GNU LGPL license. 
 *
 *                     Modified:
 *
 *                         17 May 2013
 *
 *                           Author:
 *
 *                               Original C version by Nick Hale.
 *                                   This C version by John Burkardt.
 *
 *                                     Parameters:
 *
 *                                         Input, double U[N+1], the polynomial coefficients.
 *                                             U[0] is ignored.
 *
 *                                                 Input, double H, the polynomial argument.
 *
 *                                                     Input, int N, the number of terms to compute.
 *
 *                                                         Output, double TS_MULT, the value of the polynomial.
 *                                                         */
{
  double hk;
  int k;
  double ts;
  
  ts = 0.0;
  hk = 1.0;
  for ( k = 1; k<= n; k++ )
  {
    ts = ts + u[k] * hk;
    hk = hk * h;
  }
  return ts;
}
/******************************************************************************/

double wtime ( void )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       WTIME estimates the elapsed wall clock time.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license. 
 *
 *               Modified:
 *
 *                   21 October 2009
 *
 *                     Author:
 *
 *                         John Burkardt
 *
 *                           Parameters:
 *
 *                               Output, double WTIME, the current elapsed wall clock time.
 *                               */
{
  double now;

  now = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC; 

  return now;
}

double g_pq(double p, double q, double x) {
  double pi=4*atan(1.0);
  int d;
  d = (int)(fabs(p-q));
  double g;
  g = 0.;
  if (p == q && p == 0) {

    g = 1 - x;

  }
  else if ( p == q && p > 0 ) {

    g = (1 - x)*cos(p*pi*x)/2. - sin(p*pi*x)/(2*p*pi);

  }
  else if ( (d % 2)==0) {

    g = (q*sin(q*pi*x) - p*sin(p*pi*x))/((p*p-q*q)*pi);
  }
  else g = 0.;

  return g;
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

double Vab_Int(int dim, double *xa, double *w, double *a, double *b){
  double px, py, pz, qx, qy, qz;
  double Vab;
  px = a[0] - b[0];
  py = a[1] - b[1];
  pz = a[2] - b[2];
  qx = a[0] + b[0];
  qy = a[1] + b[1];
  qz = a[2] + b[2];

  Vab = 0.;
  // Cos[px x1] Cos[py y1] Cos[pz z1]
  Vab += pq_int(dim, xa, w, px, py, pz, 0, 0, 0);
  // - Cos[qx x1] Cos[py y1] Cos[pz z1]
  Vab -= pq_int(dim, xa, w, 0,  py, pz, qx,0, 0);
  // - Cos[px x1] Cos[qy y1] Cos[pz z1]
  Vab -= pq_int(dim, xa, w, px, 0, pz, 0, qy, 0);
  // + Cos[qx x1] Cos[qy y1] Cos[pz z1]
  Vab += pq_int(dim, xa, w, 0, 0, pz, qx, qy, 0);   
  // -Cos[px x1] Cos[py y1] Cos[qz z1]  
  Vab -= pq_int(dim, xa, w, px, py, 0, 0, 0, qz);
  // +Cos[qx x1] Cos[py y1] Cos[qz z1] 
  Vab += pq_int(dim, xa, w, 0, py, 0, qx, 0, qz);
  // + Cos[px x1] Cos[qy y1] Cos[qz z1]
  Vab += pq_int(dim, xa, w, px, 0, 0, 0, qy, qz);
  // -Cos[qx x1] Cos[qy y1] Cos[qz z1]
  Vab -= pq_int(dim, xa, w, 0, 0, 0, qx, qy, qz);

  return -Vab;

}

// the E integral is 1/pi^6 \int 1/(|r1 - r2|) dr1 dr2
// which is equivalent to 
// 1/pi^6 \int cos(0 x1) cos(0 y1) cos(0 z1) cos(0 x2) cos(0 y2) cos(0 z2)/|r1-r2| dr1 dr2
//
double E0_Int(int dim, double *xa, double *w) {
  double Eint;

  Eint = pq_int(dim, xa, w, 0, 0, 0, 0, 0, 0);
  return Eint;

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
double ERI(int dim, double *xa, double *w, double *a, double *b, double *c, double *d) {

  int i, j, k, l, m, n;
  double *x1, *x2, *y1, *y2, *z1, *z2;
  double faci, facj, fack, facl, facm, facn, fac;;
  //char *cp, *cq, *cr, *cs;
  double eri_val;
  static const char *cx1[] = {"px x1", "qx x1"};
  static const char *cx2[] = {"rx x2", "sx x2"};
  static const char *cy1[] = {"py y1", "qy y1"};
  static const char *cy2[] = {"ry y2", "sy y2"};
  static const char *cz1[] = {"pz z1", "qz z1"};
  static const char *cz2[] = {"rz z2", "sz z2"};  


  x1 = (double *)malloc(3*sizeof(double));
  x2 = (double *)malloc(3*sizeof(double));
  y1 = (double *)malloc(3*sizeof(double));
  y2 = (double *)malloc(3*sizeof(double));
  z1 = (double *)malloc(3*sizeof(double));
  z2 = (double *)malloc(3*sizeof(double));

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

  double tempval = 0.;
  eri_val = 0.;
  // Generate all combinations of phi_a phi_b phi_c phi_d in expanded cosine form
  for (i=0; i<2; i++) {
    faci = pow(-1,i);
    for (j=0; j<2; j++) {
      facj = pow(-1,j);
      for (k=0; k<2; k++) {
        fack = pow(-1,k);
        for (l=0; l<2; l++) { 
          facl = pow(-1,l);
          for (m=0; m<2; m++) {
            facm = pow(-1,m);
            for (n=0; n<2; n++) {
              facn = pow(-1,n);
   
              fac = faci*facj*fack*facl*facm*facn;          
             
              // Uncomment to see the functions being integrated in each call to pq_int 
              //printf(" + %f Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] Cos[%s] \n",
              //fac,cx1[n],cx2[m],cy1[l],cy2[k],cz1[j],cz2[i]);
              // recall pq_int args are -> dim, *xa, *w, px, py, pz, qx, qy, qz
              // order of indices to get these values is a bit strange, see print statement
              // for example of ordering!
              tempval = pq_int(dim, xa, w, x1[n], y1[l], z1[j], x2[m], y2[k], z2[i]);

              // TABLE IV DEBUG LINE!!!!!!
              //printf("  (%f %f %f | %f %f %f) -> %17.14f\n",x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],tempval);
              eri_val += fac*tempval;
              

            }
          } 
        }
      }
    }
  }

 
  free(x1);
  free(x2);
  free(y1);
  free(y2);
  free(z1);
  free(z2);

  return eri_val;

}

double ERI_new(int dim, double *xa, double *a, double *b, double *c, double *d,double * g_tensor, int orbitalMax, double * sqrt_tensor, double ** PQ, int *** PQmap) {

  int * x1 = (int *)malloc(3*sizeof(int));
  int * x2 = (int *)malloc(3*sizeof(int));
  int * y1 = (int *)malloc(3*sizeof(int));
  int * y2 = (int *)malloc(3*sizeof(int));
  int * z1 = (int *)malloc(3*sizeof(int));
  int * z2 = (int *)malloc(3*sizeof(int));

  //x1[0] = ax-bx, x1[1] = ax+bx
  x1[0] = (int)(a[0] - b[0]);
  x1[1] = (int)(a[0] + b[0]);
  y1[0] = (int)(a[1] - b[1]);
  y1[1] = (int)(a[1] + b[1]);
  z1[0] = (int)(a[2] - b[2]);
  z1[1] = (int)(a[2] + b[2]);

  //x1[0] = cx-dx, x1[1] = cx+dx
  x2[0] = (int)(c[0] - d[0]);
  x2[1] = (int)(c[0] + d[0]);
  y2[0] = (int)(c[1] - d[1]);
  y2[1] = (int)(c[1] + d[1]);
  z2[0] = (int)(c[2] - d[2]);
  z2[1] = (int)(c[2] + d[2]);

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
                              printf("\n");
                              printf("    well, something is wrong with the indexing.\n");
                              printf("    %5i %5i\n",P,Q);
                              printf("    %5i %5i %5i; %5i %5i %5i\n",x1[n],y1[l],z1[j],x2[m],y2[k],z2[i]);
                              printf("\n");
                              exit(1);
                          }
                          double dum = PQ[P][Q];

                          //double dum = pq_int_new(dim, x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],g_tensor,orbitalMax,sqrt_tensor);

                          // TABLE IV DEBUG LINE!!!!!!
                          //printf("  (%f %f %f | %f %f %f) -> %17.14f\n",x1[n], y1[l], z1[j], x2[m], y2[k], z2[i],dum);
                          eri_val += facijklmn * dum;

                      }
                  } 
              }
          }
      }
  }

 
  free(x1);
  free(x2);
  free(y1);
  free(y2);
  free(z1);
  free(z2);

  return eri_val;

}

// This function implements Eq. 4.7 and 4.8 in Peter Gills paper on 2-electrons in a cube
// Gauss-Legendre quadrature is used for the 3d integral on the range 0->1 for x, y, and z
// int dim is the number of points on this grid, double *xa is a vector containing the actual points on this grid, and
// double *w is a vector containing the weights associated with this grid (analogous to differential length elements
// in rectangle rule integration).
// double px, py, pz, qx, qy, qz has the same interpretation as it does in the Gill paper.
double pq_int(int dim, double *xa, double *w, double px, double py, double pz, double qx, double qy, double qz) {
  double pi=4*atan(1.0);
  double sum = 0.;
  double num, denom;
  double x, y, z, dx, dy, dz;
  double gx, gy, gz;

  if (px<0 || qx<0 || py<0 || qy<0 || pz<0 || qz<0) {
  return 0.;
  } 

  else {

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
        //printf("  sum %f  x %f  y %f  z %f\n",sum, x, y, z);
      }
    }
  }

  return (8./pi)*sum;
  }
}

double pq_int_new(int dim, int px, int py, int pz, int qx, int qy, int qz, double * g_tensor, int orbitalMax, double * sqrt_tensor) {

    if (px<0 || qx<0 || py<0 || qy<0 || pz<0 || qz<0) {

        return 0.;

    }

    double sum = 0.0;

    for (int i = 0; i < dim; i++) {

        double gx = g_tensor[i * orbitalMax * orbitalMax + px * orbitalMax + qx];

        for (int j = 0; j < dim; j++) {

            double gxgy = gx * g_tensor[j * orbitalMax * orbitalMax + py * orbitalMax + qy];

            for (int k = 0; k < dim; k++) {

                double gxgygz = gxgy * g_tensor[k * orbitalMax * orbitalMax + pz * orbitalMax + qz];

                sum += gxgygz * sqrt_tensor[i*dim*dim + j*dim + k];

                //printf("  sum %f  x %f  y %f  z %f\n",sum, x, y, z);
            }
        }
    }

    return 8.0 * sum / M_PI;

}

/* 
/ Function to Determine Energy Calculation
/ Take function and loop through n to keep all atomic orbital energy levels.
*/
void OrderPsis3D(int norbs, int *E, int **MO) {

  int i, j, k, l, c, d, swap, idx, tx, ty, tz, ix, iy, iz;
  int **N, tidx[3];
  int cond, Ecur;
  N = MAT_INT(2*(norbs+1)*(norbs+1)*(norbs+1),3);

  // Going to start with i and j=0 because that's the beginning
  // of the array... nx=i+1, ny=j+1

  for (i=0; i<norbs; i++) {
    for (j=0; j<norbs; j++) {
      for (k=0; k<norbs; k++) {

        idx = i*norbs*norbs+j*norbs+k;
        // l is related to nx^2 + ny^2 + nz^2, aka, (i+1)^2 + (j+1)^2 (k+1)^2
        l = (i+1)*(i+1) + (j+1)*(j+1) + (k+1)*(k+1);
        E[idx] = l;
        // index is and energy is ...
        printf("  Index is %i and Energy[%i,%i,%i] is %i\n",idx,i+1,j+1,k+1,l);
        // element N[k][0] is nx = i+1
        N[l][0] = i+1;
        // element N[k][1] is ny = j+1
        N[l][1] = j+1;
        // element N[k][2] is nz = k+1
        N[l][2] = k+1;

      }
    }
  }

  for (c = 0 ; c < ( norbs*norbs*norbs-1 ); c++)
  {
    for (d = 0 ; d < norbs*norbs*norbs - c - 1; d++)
    {
      if (E[d] > E[d+1]) /* For decreasing order use < */
      {
        swap       = E[d];
        E[d]   = E[d+1];
        E[d+1] = swap;
      }
    }
  }

// print all energy values
for (i=0; i<(norbs*norbs*norbs); i++) {
  printf(" E[%i] is %i \n",i,E[i]);
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

  printf(" exit successful \n");

//  for (i=0; i<(norbs*norbs*norbs); i++) {
//    printf("  Psi( %i , %i, %i ) %i\n",MO[i][0],MO[i][1],MO[i][2],E[i]);
//  }

}

int *VEC_INT(int dim){
  int *v,i;
  v = (int *)malloc(dim*sizeof(int));
  if (v==NULL) {
     printf("\n\nVEC_INT: Memory allocation error\n\n");
     exit(0);
  }
  for (i=0; i<dim; i++) v[i] = 0;
  return v;
}

int **MAT_INT(int dim1, int dim2){
  int i,j,k;
  double sum=0.0;
  int **M;
  M = (int **)malloc(dim1*sizeof(int *));
  if (M==NULL) {
     printf("\n\nMAT_INT: Memory allocation error\n\n");
     exit(0);
  }
  for (i=0; i<dim1; i++){
      M[i] = (int *)malloc(dim2*sizeof(int));
      if (M[i]==NULL) {
         printf("\n\nMAT_INT: Memory allocation error\n\n");
         exit(0);
      }
      for (j=0; j<dim2; j++){
          M[i][j] = 0;
      }
  }
  return M;
}
