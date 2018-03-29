- Contains a program which will (hopefully) compute all 1- and 2-electron integrals + the self energy and store them in files

- To compile, type `make`

- Closely follows reference by Peter Gills here: https://github.com/WillyPChem/Jellium/blob/master/Papers/Particle_in_Cube.pdf

- Integrals are computed in the basis of 3D particle in a cube energy eigenstates (see Eq. 2.2)	

- All 1- and 2-electron integrals can be expressed in terms of canonical integrals of the form shown in Eq. 4.2

- Eq. 4.7 and Eq. 4.8 are used to numerically evaluate the cononical integrals

	- Gauss-Legendre quadrature is used to compute Integral in 4.7 over the range 0 to 1
	- The number of grid points between 0 and 1 is selected at run time by the first argument passed to the program
	- The lower and upper limits of integration (0 and 1) are selected at runtime by the second and third arguments passed to the program at runtime
	- `./JelliumIntegrals.x 20 0 1` will evaluate integral (Eq. 4.7) between 0 and 1 using 20 grid points
	- More gridpoints will lead to tighter convergence of the integrals, but will also increase the runtime
	- Note that Eq. 4.7 is a 3-D integral, so 2x as many gridpoints leads to 2^3 x as many operations!

- Outputs from the program are written to file
	- 2-electron repulsion integrals ~ half of all integrals are computed -> ERI.dat
	- 1-electron nuclear attraction integrals - symmetry fully exploited  -> NucAttraction.dat
	- 1-electron kinetic energy integrals     - symmetry fully exploited  -> Kinetic.dat
	- Nuclear Self-Energy - treat same as nuclear repulsion in H2O case   -> SelfEnergy.dat


