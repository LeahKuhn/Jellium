Particle in Cube HF Jellium

Hartree-Fock Jellium provides a mean-field treatment of valence electrons constrained to a cubic volume and subject to a uniform background positive charge and electron-electron repulsion.  

Current worflow is as follows:

- Generate all 1- and 2-electron integrals using the program Jellium_Integrals.x in the folder /JelliumIntegrals  

- Integrals will be written to files:
	- ERI.dat:           2-electron repulsion integrals
	- Kinetoc/dat:       1-electron kinetic energy integrals
	- NucAttraction.dat: 1-electron nuclear attraction integrals
	- SelfEnergy.dat:    Average repulsion on positive background charge with itself (independant of electron coordinates, just a single number)

- More details on Integral Code can be found here: https://github.com/WillyPChem/Jellium/blob/master/JelliumIntegrals/README.md 
 
- Self-consistent field calculation is done using the program JPIC.x in the current directory, which will read the integrals from the JelliumIntegrals folder.

	- To compile JPIC code, type `make`
	- To run JPIC.x, type `./JPIC.x`


- SCF calculation closely follows Daniel Crawford's tutorial here: http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project3 so much of the notation is consistent
- The definition of the Fock operator and the SCF Energy follow the convention in Peter Gill's paper on Jellium RHF here: https://github.com/WillyPChem/Jellium/blob/master/Papers/Particle_in_Cube.pdf
	- Fock operator -> Equation 3.8
	- SCF Energy ->    Equation 3.3

- Current target for SCF calculation is to reproduce the Nelectrons=2, Norbs=26, total energy from Table V in Peter Gill's paper (17.14416 Hartrees)


