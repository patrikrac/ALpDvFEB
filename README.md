# ALpDvFEB
Collection of Programms using different Finite-Element Libraries to solve partial differential equations.
This is the Uniform refinement branch used to create reference data to compare to the adaptive methods. All programs feature the same functionality as in the main branch but uniformly refine the mesh for each iteration.
The used libraries include 
- deal.ii
  - For more information and installation instructions visit <https://www.dealii.org>
- Netgen/NGSolve
  - For more information and installation instructions visit <https://ngsolve.org>
- MFEM
  - For more information and installation instructions visit <https://mfem.org> and <https://mfem.org/howto/build-systems/> for instructions on how to build using cmake
  - The parallel build of MFEM will require additional libraries

The programms build the bases to solve PDEs using an adaptive mesh refinement loop.

The programms are written as the main work of my Bachelor Thesis at Friedrich-Alexander Universität Erlangen-Nürnberg. 

