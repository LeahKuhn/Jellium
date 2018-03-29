#include"blas.h"
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>
#include<complex.h>
#define BIGNUM 1E100
#define MAXIT 1000
#define NORM_TOL 1.0E-5

/**
 * fortran dgetrf
 */
void DGETRF(integer n,integer m, doublereal *M,integer lda,integer *ipiv,integer info){
  DGETRF(n,m,M,lda,ipiv,info);
}


/**
 * fortran dgetri
 */
void DGETRI(integer n, doublereal *M,integer lda,integer *ipiv,doublereal *work,integer lwork,integer info) {
  DGETRI(n,M,lda,ipiv,work,lwork,info);
}

/**
 * fortran-ordered dgemv
 */
void F_DGEMV(char trans,integer m,integer n,doublereal alpha,doublereal*A,integer lda,
            doublereal*X,integer incx,doublereal beta,doublereal*Y,integer incy){
    DGEMV(trans,m,n,alpha,A,lda,X,incx,beta,Y,incy);
}
/**
 * fortran-ordered dgemm
 */
void F_DGEMM(char transa,char transb, integer m, integer n, integer k,
            doublereal alpha,doublereal*A,integer lda,doublereal*B,integer ldb,
            doublereal beta,doublereal*C,integer ldc){
    DGEMM(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}

/** 
 * fortran-ordered zgemm:  Subroutine: 

        int zgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, doublecomplex *alpha, doublecomplex *a, integer *lda, 
	doublecomplex *b, integer *ldb, doublecomplex *beta, doublecomplex *c,
	 integer *ldc)
*/
void F_ZGEMM(char transa, char transb, integer m, integer n, integer k, doublecomplex alpha,
             doublecomplex *A, integer lda, doublecomplex *B, integer ldb, doublecomplex beta,
             doublecomplex *C, integer ldc) {
     ZGEMM(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}

/**
 * dnrm2
 */
double F_DNRM2(integer n,doublereal*x,integer incx){
    return DNRM2(n,x,incx);
}
/**
 * ddot
 */
double F_DDOT(integer n,doublereal*dx,integer incx,doublereal*dy,integer incy){
    return DDOT(n,dx,incx,dy,incy);
}
/**
 * dcopy
 */
void F_DCOPY(integer n,doublereal*dx,integer incx,doublereal*dy,integer incy){
    DCOPY(n,dx,incx,dy,incy);
}

/**
 * daxpy
 */
void F_DAXPY(integer n,doublereal da,doublereal*dx,integer incx,doublereal*dy,
             integer incy){
    DAXPY(n,da,dx,incx,dy,incy);
}

/**
 * DSYEV
 */
void F_DSYEV(char &jobz, char &uplo, integer &n, doublereal *ap,integer &lda, doublereal *w, doublereal*work, integer &lwork, integer &info)
{
  DSYEV(jobz,uplo,n,ap,lda,w,work, lwork, info);
}
