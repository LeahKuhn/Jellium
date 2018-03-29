LIB        = -L. -lm -llapack -lblas -Lg2c
INCLUDE    = -I.
CFLAGS     = -O2 -g -pg
EXEC       = JPIC.x
CXX        = g++

${EXEC}: JPIC.c  blas.o
	${CXX} ${CFLAGS} ${INCLUDE} ${LIB} JPIC.c blas.o -o ${EXEC}


blas.o: blas.c blas.h
	${CXX} ${LIB} -c blas.c ${CFLAGS}

clean:
	rm -f *.o

%.o: $.cpp
	${CXX} -c ${CFLAGS} ${INCL} -cpp -o $*.o $<

