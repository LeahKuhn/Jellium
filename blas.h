#ifndef BLAS_H
#define BLAS_H
#include<complex.h>
/**
 * fortran-ordered blas routines
 */

#define F77NAME(x) x##_
typedef long int integer;
typedef double doublereal;
typedef double complex doublecomplex;

//void F_DGEQRF(integer m, integer n, doublereal *A,integer LDA, doublereal *TAU, doublereal *work, integer lwork, integer info);

/**
 * fortran-ordered dgemv
 */
void F_DGEMV(char trans,integer m,integer n,doublereal alpha,doublereal*A,integer lda,
            doublereal*X,integer incx,doublereal beta,doublereal*Y,integer incy);
/**
 * fortran-ordered dgemm
 */

void F_DGEMM(char transa,char transb, integer m, integer n, integer k,
            doublereal alpha,doublereal*A,integer lda,doublereal*B,integer ldb,
            doublereal beta,doublereal*C,integer ldc);

/**
 * fortran-ordered zgemm
 */

void F_ZGEMM(char transa, char transb, integer m, integer n, integer k, doublecomplex alpha,
             doublecomplex *A, integer lda, doublecomplex *B, integer ldb, doublecomplex beta,
             doublecomplex *C, integer ldc);

/**
 * fortran-ordered dsyev
 */
void F_DSYEV(char jobz, char uplo, integer &n, doublereal *ap,integer &lda, doublereal *w, doublereal*work, integer &lwork, integer &info);


/**
 * ddot
 */
double F_DDOT(integer n,doublereal*dx,integer incx,doublereal*dy,integer incy);

/**
 * dnrm2
 */
double F_DNRM2(integer n,doublereal*x,integer incx);

/**
 * dcopy
 */
void F_DCOPY(integer n,doublereal*dx,integer incx,doublereal*dy,integer incy);

/**
 * daxpy
 */
void F_DAXPY(integer n,doublereal da,doublereal*dx,integer incx,doublereal*dy,
             integer incy);


/** 
 * name changing for fortran-ordered dgeqrf
 */
/*extern "C" {
    void F77NAME(dgeqrf)(integer &m, integer &n, doublereal *A, integer &LDA, doublereal *TAU, doublereal *work, integer &lwork, integer &info) {
    F77NAME(dgeqrf)(m, n, A, LDA, TAU, work, lwork, info);

};
*/

/**
 * name manging for fortra-ordered dsyev
 */
extern "C" {
void F77NAME(dsyev)(char &jobz,char &uplo,integer &n,doublereal*ap,integer &lda, doublereal*w,doublereal*work, integer &lwork,integer&info);
};
inline void DSYEV(char &jobz, char &uplo, integer &n, doublereal *ap,integer &lda, doublereal *w, doublereal*work, integer &lwork, integer &info)
{
  F77NAME(dsyev)(jobz,uplo,n,ap,lda,w,work, lwork, info);
}



/**
 * name manging for fortran-ordered dgemv
 */
extern "C" {
    void F77NAME(dgemv)(char&trans,integer&m,integer&n,doublereal&alpha,doublereal*A,integer&lda,
            doublereal*X,integer&incx,doublereal&beta,doublereal*Y,integer&incy);
};
inline void DGEMV(char&trans,integer&m,integer&n,doublereal&alpha,doublereal*A,integer&lda,
            doublereal*X,integer&incx,doublereal&beta,doublereal*Y,integer&incy){
    F77NAME(dgemv)(trans,m,n,alpha,A,lda,X,incx,beta,Y,incy);
}
/**
 * name manging for fortran-ordered dgemm
 */
extern "C" {
    void F77NAME(dgemm)(char&transa,char&transb,integer&m,integer&n,integer&k,
         doublereal&alpha,doublereal*A,integer&lda,doublereal*B,integer&ldb,
         doublereal&beta,doublereal*C,integer&ldc);
};
inline void DGEMM(char&transa,char&transb,integer&m,integer&n,integer&k,
         doublereal&alpha,doublereal*A,integer&lda,doublereal*B,integer&ldb,
         doublereal&beta,doublereal*C,integer&ldc)
{
    F77NAME(dgemm)(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
};

extern "C" {
     void F77NAME(zgemm)(char&transa, char&transb, integer&m, integer&n, integer&k, doublecomplex&alpha,
             doublecomplex*A, integer&lda, doublecomplex*B, integer&ldb, doublecomplex&beta,
             doublecomplex*C, integer&ldc);
};
inline void ZGEMM(char&transa, char&transb, integer&m, integer&n, integer&k, doublecomplex&alpha,
             doublecomplex*A, integer&lda, doublecomplex*B, integer&ldb, doublecomplex&beta,
             doublecomplex*C, integer&ldc)
{
    F77NAME(zgemm)(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
};
/**
 * name manging dcopy
 */
extern "C" {
    void F77NAME(dcopy)(integer&n,doublereal*dx,integer&incx,doublereal*dy,
         integer&incy);
};
inline void DCOPY(integer&n,doublereal*dx,integer&incx,doublereal*dy,
            integer&incy){
    F77NAME(dcopy)(n,dx,incx,dy,incy);
}
/**
 * name manging daxpy
 */
extern "C" {
   void F77NAME(daxpy)(integer&n,doublereal&da,doublereal*dx,integer&incx,
        doublereal*dy,integer&incy);
};
inline void DAXPY(integer&n,doublereal&da,doublereal*dx,integer&incx,
            doublereal*dy,integer&incy)
{
    F77NAME(daxpy)(n,da,dx,incx,dy,incy);
};
/**
 * name manging dnrm2
 */
extern"C"{
    double F77NAME(dnrm2)(integer&N,doublereal*X,integer&INCX);
};
inline double DNRM2(integer&N,doublereal*X,integer&INCX){
    return F77NAME(dnrm2)(N,X,INCX);
};
/**
 * name manging dspev
 */
extern "C" {
void F77NAME(dspev)(char &jobz,char &uplo,integer &n,doublereal*ap,doublereal*w,
                    doublereal*z__,integer&ldz,doublereal*work,integer&info);
};
inline void DSPEV(char &jobz, char &uplo, integer &n, doublereal *
           ap, doublereal *w, doublereal *z__, integer &ldz, doublereal *work,
           integer &info)
{
  F77NAME(dspev)(jobz,uplo,n,ap,w,z__,ldz,work,info);
}

extern "C" {
void F77NAME(dgeqrf)(integer &m, integer &n,doublereal *a,integer &lda,doublereal *tau, doublereal *work,integer &lwork,integer &info);
};
inline void DGEQRF(integer &m, integer &n, doublereal *a,integer &lda,doublereal *tau,doublereal *work,integer &lwork,integer &info) 
{
  F77NAME(dgeqrf)(m, n, a, lda, tau, work, lwork, info);
}  


/**
 * name manging dgesv
 */
extern"C" {
    void F77NAME(dgesv)(integer &N,integer &NRHS,doublereal*A,integer &LDA,integer*IPIV,doublereal*B,integer &LDB,integer &INFO);
};
inline void DGESV(integer &N,integer &NRHS,doublereal*A,integer &LDA,integer*IPIV,doublereal*B,integer &LDB,integer &INFO){
    F77NAME(dgesv)(N,NRHS,A,LDA,IPIV,B,LDB,INFO);
};
/**
 * name manging ddot
 */
extern "C" {
    double F77NAME(ddot)(integer&n,doublereal*dx,integer&incx,doublereal*dy,integer&incy);
};
inline double DDOT(integer&n,doublereal*dx,integer&incx,doublereal*dy,integer&incy){
    return F77NAME(ddot)(n,dx,incx,dy,incy);
}

#endif
