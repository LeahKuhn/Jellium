LIB        = -L. 
INCLUDE    = -I.
CFLAGS     = -O3
EXEC       = JelliumIntegrals.exe
CXX        = g++

${EXEC}: JelliumIntegrals.c
	${CXX} ${CFLAGS} ${INCLUDE} ${LIB} JelliumIntegrals.c -o ${EXEC}

clean:
	rm -f *.o

%.o: $.cpp
	${CXX} -c ${CFLAGS} ${INCL} -cpp -o $*.o $<
